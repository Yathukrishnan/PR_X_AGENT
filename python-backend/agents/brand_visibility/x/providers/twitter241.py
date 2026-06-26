"""
twitter241 provider — RapidAPI scraper by davethebeast.
Endpoints:
  GET /search-v2          (keyword search, type=Latest|Top, count=20)
  GET /user               (handle -> user ID resolution)
  GET /user-tweets        (timeline by numeric user ID)
  GET /user-replies-v2    (replies by numeric user ID)

Important: /user-tweets and /user-replies-v2 take numeric user IDs, NOT handles.
We cache handle->ID in Turso to avoid double-billing the resolution call.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import requests

from agents.brand_visibility.x.providers import NormalizedTweet, ProviderCapabilities, ScraperProvider
from agents.brand_visibility.x.providers._http import safe_get

logger = logging.getLogger(__name__)

BASE_URL = "https://twitter241.p.rapidapi.com"
DEFAULT_COUNT = 20


# Same operator-stripping pattern as twitter-api45 — twitter241 doesn't support
# X v2 search operators either, so we strip and re-apply post-hoc.
_FILTER_PATTERNS = {
    "min_faves": re.compile(r"\s*min_faves:(\d+)"),
    "is_retweet": re.compile(r"\s*-is:retweet"),
    "is_reply": re.compile(r"\s*-is:reply"),
    "is_nullcast": re.compile(r"\s*-is:nullcast"),
    "lang": re.compile(r"\s*lang:(\w+)"),
}


def _strip_unsupported_operators(query: str) -> tuple[str, dict]:
    filters: dict[str, Any] = {}
    for name, pattern in _FILTER_PATTERNS.items():
        match = pattern.search(query)
        if match:
            filters[name] = match.group(1) if match.groups() else True
            query = pattern.sub("", query)
    return query.strip(), filters


def _is_reply_to_other(tweet) -> bool:
    """
    True if tweet is a reply to a DIFFERENT user.
    False if it's standalone OR a self-thread continuation (same user replying to themselves).
    """
    legacy = tweet.raw.get("legacy", {}) or {}
    in_reply_to_user_id = legacy.get("in_reply_to_user_id_str")
    own_user_id = legacy.get("user_id_str")
    if not in_reply_to_user_id:
        return False  # Not a reply at all
    if str(in_reply_to_user_id) == str(own_user_id):
        return False  # Self-thread continuation — keep
    return True  # Reply to someone else — drop


def _apply_posthoc_filters(tweets: list[NormalizedTweet], filters: dict) -> list[NormalizedTweet]:
    out = tweets
    if "min_faves" in filters:
        threshold = int(filters["min_faves"])
        out = [t for t in out if t.like_count >= threshold]
    if filters.get("is_retweet"):
        out = [t for t in out if not t.text.startswith("RT @")]
    if filters.get("is_reply"):
        out = [t for t in out if not _is_reply_to_other(t)]
    if "lang" in filters:
        target_lang = filters["lang"]
        out = [t for t in out if (t.lang or "").startswith(target_lang)]
    return out


def _parse_date(s: str | None) -> datetime:
    """twitter241 returns RFC 822 format: 'Tue Mar 12 04:12:30 +0000 2024'."""
    if not s:
        return datetime.now(timezone.utc)
    try:
        return parsedate_to_datetime(s)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        logger.warning("Could not parse date %r — using now()", s)
        return datetime.now(timezone.utc)


def _extract_tweets_from_response(payload: dict) -> list[dict]:
    """
    twitter241 wraps tweets in a GraphQL "instructions/entries" envelope.
    Walk: payload -> result -> timeline -> instructions -> [...] -> entries -> [...]
    Each entry's tweet lives at: entry.content.itemContent.tweet_results.result
    """
    # Find the instructions list — try a few known paths
    instructions = None
    for path in [
        ["result", "timeline", "instructions"],
        ["data", "search_by_raw_query", "search_timeline", "timeline", "instructions"],
        ["data", "user", "result", "timeline_v2", "timeline", "instructions"],
        ["timeline", "instructions"],
    ]:
        cursor: Any = payload
        for key in path:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                cursor = None
                break
        if isinstance(cursor, list):
            instructions = cursor
            break

    if instructions is None:
        logger.warning(
            "twitter241: no instructions list found. Top keys: %s",
            list(payload.keys()) if isinstance(payload, dict) else type(payload),
        )
        return []

    # Walk every entry across every instruction
    tweet_objects = []
    for instr in instructions:
        if not isinstance(instr, dict):
            continue

        entries = instr.get("entries", [])

        # If instr IS the entry itself (no wrapper), treat it as a one-element list
        if not entries and "content" in instr:
            entries = [instr]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content", {})
            item_content = content.get("itemContent", {})
            tweet_results = item_content.get("tweet_results", {})
            result = tweet_results.get("result")
            if not result:
                continue

            # Unwrap TweetWithVisibilityResults wrapper if present
            if result.get("__typename") == "TweetWithVisibilityResults":
                result = result.get("tweet", result)

            # Only accept actual Tweet objects (not promoted, not deleted, etc)
            if result.get("__typename") in (None, "Tweet"):
                tweet_objects.append(result)

    if not tweet_objects:
        logger.warning(
            "twitter241: found %d instructions but extracted 0 tweets",
            len(instructions),
        )

    return tweet_objects


def _extract_bottom_cursor(payload: dict) -> str | None:
    """Find the 'Bottom' cursor value in twitter241's response (for fetching next page)."""
    # Walk the same response shapes _extract_tweets_from_response handles
    instructions_paths = [
        ["result", "timeline", "instructions"],
        ["data", "search_by_raw_query", "search_timeline", "timeline", "instructions"],
        ["data", "user", "result", "timeline_v2", "timeline", "instructions"],
        ["timeline", "instructions"],
    ]

    instructions = None
    for path in instructions_paths:
        cursor = payload
        for key in path:
            if isinstance(cursor, dict):
                cursor = cursor.get(key)
            else:
                cursor = None
                break
        if cursor:
            instructions = cursor
            break

    if not instructions:
        return None

    for instr in instructions:
        for entry in instr.get("entries", []):
            content = entry.get("content", {})
            if (content.get("__typename") == "TimelineTimelineCursor"
                    and content.get("cursorType") == "Bottom"):
                return content.get("value")
    return None


def _normalize(raw: dict) -> Optional[NormalizedTweet]:
    """
    Convert a twitter241 Tweet result object to NormalizedTweet.
    Expected shape (after _extract_tweets_from_response unwrapping):
      raw.legacy.{id_str, full_text, created_at, favorite_count, ...}
      raw.core.user_results.result.{rest_id, legacy: {description, followers_count, ...}, core: {name, screen_name}}
      raw.views.count (string like "103")
      raw.rest_id (tweet ID, redundant with legacy.id_str)
      raw.note_tweet.note_tweet_results.result.text (long-form text if present)
    """
    legacy = raw.get("legacy", {})

    tweet_id = legacy.get("id_str") or raw.get("rest_id")
    if not tweet_id:
        logger.debug("Skipping tweet with no id_str/rest_id")
        return None

    # Prefer note_tweet text (full long-form) if present, else full_text
    text = legacy.get("full_text") or ""
    note_tweet = raw.get("note_tweet", {}).get("note_tweet_results", {}).get("result", {})
    if note_tweet.get("text"):
        text = note_tweet["text"]

    # User info — nested under core.user_results.result
    user_result = raw.get("core", {}).get("user_results", {}).get("result", {})
    user_legacy = user_result.get("legacy", {})
    user_core = user_result.get("core", {})

    author_id = user_result.get("rest_id") or legacy.get("user_id_str") or ""
    # screen_name lives under core in twitter241 responses
    author_handle = user_core.get("screen_name") or user_legacy.get("screen_name") or ""
    author_followers = int(user_legacy.get("followers_count") or 0)
    author_bio = user_legacy.get("description") or ""

    created_at = _parse_date(legacy.get("created_at"))

    def _to_int(v: Any) -> int:
        if v is None or v == "":
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    like_count = _to_int(legacy.get("favorite_count"))
    reply_count = _to_int(legacy.get("reply_count"))
    retweet_count = _to_int(legacy.get("retweet_count"))
    quote_count = _to_int(legacy.get("quote_count"))

    # views.count is a string like "103"
    views_obj = raw.get("views", {})
    impression_count = _to_int(views_obj.get("count")) if views_obj.get("count") else None

    lang = legacy.get("lang")
    conversation_id = legacy.get("conversation_id_str")

    return NormalizedTweet(
        tweet_id=str(tweet_id),
        author_id=str(author_id) if author_id else None,
        author_handle=author_handle,
        author_followers=author_followers,
        author_bio=author_bio,
        text=text,
        created_at=created_at,
        like_count=like_count,
        reply_count=reply_count,
        retweet_count=retweet_count,
        quote_count=quote_count,
        impression_count=impression_count,
        lang=lang,
        conversation_id=str(conversation_id) if conversation_id else None,
        raw=raw,
    )


class Twitter241Provider(ScraperProvider):
    """RapidAPI twitter241 (davethebeast)."""

    capabilities = ProviderCapabilities(
        supports_or_grouping_in_query=True,
        supports_min_faves=False,         # post-hoc filtered
        supports_exclude_retweets=False,  # post-hoc filtered
        supports_exclude_replies=False,   # post-hoc filtered
        supports_lang_in_query=False,     # post-hoc filtered
        supports_from_handle_in_query=False,
        supports_conversation_id_in_query=False,
        supports_since_id=False,
        rate_limit_header_remaining="x-ratelimit-requests-remaining",
        rate_limit_header_reset="x-ratelimit-requests-reset",
        notes=(
            "twitter241 by davethebeast on RapidAPI. Higher rate limits than "
            "twitter-api45. User-tweets and user-replies-v2 require numeric user "
            "IDs — resolve via /user endpoint and cache."
        ),
    )

    def __init__(self) -> None:
        self.api_key = os.environ.get("RAPIDAPI_KEY", "")
        if not self.api_key:
            raise RuntimeError("RAPIDAPI_KEY is empty — set in .env")
        self.headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": "twitter241.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    # ------------------------------------------------------------------
    # search-v2
    # ------------------------------------------------------------------

    def search_recent(
        self,
        query: str,
        since_id: str | None = None,
        max_results: int = DEFAULT_COUNT,
        max_pages: int = 1,
        sweep_type: str = "Latest",
    ) -> list[NormalizedTweet]:
        cleaned, filters = _strip_unsupported_operators(query)

        all_raw_entries = []
        cursor = None

        for page_num in range(max_pages):
            params = {
                "type": sweep_type,
                "count": str(min(max_results, 20)),
                "query": cleaned,
            }
            if cursor:
                params["cursor"] = cursor

            resp = safe_get(
                self.session,
                f"{BASE_URL}/search-v2",
                params=params,
            )
            data = resp.json()

            page_entries = _extract_tweets_from_response(data)
            if not page_entries:
                logger.info(f"twitter241: pagination stopped at page {page_num + 1} (empty response)")
                break

            all_raw_entries.extend(page_entries)

            # Get cursor for the next page (only if we have more pages to fetch)
            if page_num + 1 < max_pages:
                cursor = _extract_bottom_cursor(data)
                if not cursor:
                    logger.info(f"twitter241: pagination stopped at page {page_num + 1} (no cursor available)")
                    break

        # Normalize and apply filters (same as before, but on combined entries)
        normalized = []
        for entry in all_raw_entries:
            t = _normalize(entry)
            if t is None:
                continue
            # since_id post-filter (twitter241 doesn't support it natively)
            if since_id and t.tweet_id <= since_id:
                continue
            normalized.append(t)
        return _apply_posthoc_filters(normalized, filters)

    # ------------------------------------------------------------------
    # user handle -> user ID resolution
    # ------------------------------------------------------------------

    def resolve_user_id(self, handle: str) -> str | None:
        username = handle.lstrip("@")
        resp = safe_get(
            self.session,
            f"{BASE_URL}/user",
            params={"username": username},
        )
        data = resp.json()
        # Common paths for user ID in Twitter responses
        for path in [
            ["result", "data", "user", "result", "rest_id"],
            ["data", "user", "result", "rest_id"],
            ["user", "rest_id"],
            ["rest_id"],
            ["id_str"],
            ["id"],
        ]:
            cursor: Any = data
            for key in path:
                if isinstance(cursor, dict) and key in cursor:
                    cursor = cursor[key]
                else:
                    cursor = None
                    break
            if cursor is not None:
                return str(cursor)

        logger.warning("Could not resolve user ID for @%s in response", username)
        return None

    # ------------------------------------------------------------------
    # user-tweets (timeline)
    # ------------------------------------------------------------------

    def user_timeline(self, handle: str, max_results: int = 10) -> list[NormalizedTweet]:
        user_id = self.resolve_user_id(handle)
        if not user_id:
            return []
        resp = safe_get(
            self.session,
            f"{BASE_URL}/user-tweets",
            params={"user": user_id, "count": str(max_results)},
        )
        raw_entries = _extract_tweets_from_response(resp.json())
        tweets = [t for t in (_normalize(e) for e in raw_entries) if t is not None]
        return tweets

    # ------------------------------------------------------------------
    # user-replies-v2 — used for reply-tree expansion
    # ------------------------------------------------------------------

    def conversation_replies(
        self, tweet_id: str, max_pages: int = 3
    ) -> list[NormalizedTweet]:
        # twitter241's user-replies-v2 takes a user ID, not a tweet ID, and returns
        # the user's replies — which isn't quite the same as "all replies to one tweet"
        # but is what's available.
        logger.warning(
            "twitter241 conversation_replies is approximate — "
            "returns user replies, not full conversation tree"
        )
        resp = safe_get(
            self.session,
            f"{BASE_URL}/user-replies-v2",
            params={"user": tweet_id, "count": "20"},
        )
        raw_entries = _extract_tweets_from_response(resp.json())
        return [t for t in (_normalize(e) for e in raw_entries) if t is not None]

    # ------------------------------------------------------------------
    # rate limit headers
    # ------------------------------------------------------------------

    def get_rate_limit_status(
        self, response_headers: dict
    ) -> tuple[int | None, datetime | None]:
        remaining = response_headers.get(self.capabilities.rate_limit_header_remaining)
        reset = response_headers.get(self.capabilities.rate_limit_header_reset)
        try:
            remaining_int = int(remaining) if remaining else None
        except ValueError:
            remaining_int = None
        try:
            reset_dt = (
                datetime.fromtimestamp(int(reset), tz=timezone.utc)
                if reset else None
            )
        except (ValueError, OSError):
            reset_dt = None
        return remaining_int, reset_dt