"""
twitter-api45 adapter — RapidAPI provider by alexanderxbx.

Base URL: https://twitter-api45.p.rapidapi.com
Auth headers (lowercase, both required):
    x-rapidapi-host: twitter-api45.p.rapidapi.com
    x-rapidapi-key:  <RAPIDAPI_KEY>

Endpoint reference (verified):

GET /search.php
  Params: query (required), cursor (optional), search_type ("Top"|"Latest")
  Response: { "timeline": [...tweets...], "prev_cursor": "...", "next_cursor": "..." }

GET /timeline.php
  Params: screenname (required, no @), cursor (optional)
  Response: same tweet shape under "timeline" key.

GET /screenname.php
  Params: screenname (required), rest_id (optional)
  Response: user profile object.

GET /latest_replies.php
  Params: id (required — tweet_id), cursor (optional)
  Response: same tweet shape, returns replies. Substitute for conversation_id: in X v2.

Tweet object shape:
  tweet_id, screen_name, user_id, text, created_at (RFC 822), favorites (int),
  retweets (int), replies (int), quotes (int), bookmarks (int),
  views (STRING — impressions, sometimes empty), lang,
  user_info (may be absent): { name, screen_name, description, followers_count }
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import requests

from agents.brand_visibility.x.providers import NormalizedTweet, ProviderCapabilities, ScraperProvider
from agents.brand_visibility.x.providers._http import safe_get

logger = logging.getLogger(__name__)

BASE_URL = "https://twitter-api45.p.rapidapi.com"

# ---------------------------------------------------------------------------
# Operator stripping — X v2 operators that twitter-api45 doesn't support
# ---------------------------------------------------------------------------

_FILTER_PATTERNS = {
    "min_faves": re.compile(r"\s*min_faves:(\d+)"),
    "is_retweet": re.compile(r"\s*-is:retweet"),
    "is_reply": re.compile(r"\s*-is:reply"),
    "is_nullcast": re.compile(r"\s*-is:nullcast"),
    "lang": re.compile(r"\s*lang:(\w+)"),
}


def _strip_unsupported_operators(query: str) -> tuple[str, dict]:
    """Strip X v2 operators from query; return (cleaned_query, extracted_filters)."""
    filters: dict = {}
    for name, pattern in _FILTER_PATTERNS.items():
        match = pattern.search(query)
        if match:
            filters[name] = match.group(1) if match.groups() else True
            query = pattern.sub("", query)
    return query.strip(), filters


def _apply_posthoc_filters(tweets: list[dict], filters: dict) -> list[dict]:
    """Re-apply the operators that were stripped before the API call."""
    result = tweets

    if "min_faves" in filters:
        threshold = int(filters["min_faves"])
        result = [t for t in result if t.get("favorites", 0) >= threshold]

    if filters.get("is_retweet"):
        result = [t for t in result if not t.get("text", "").startswith("RT @")]

    if filters.get("is_reply"):
        result = [t for t in result if not t.get("in_reply_to_status_id")]

    if "lang" in filters:
        wanted = filters["lang"]
        result = [t for t in result if t.get("lang") == wanted]

    return result


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _safe_int(v: object, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _normalize(raw: dict) -> NormalizedTweet:
    user_info = raw.get("user_info") or {}
    created_at_str = raw.get("created_at", "")
    try:
        created_at = parsedate_to_datetime(created_at_str) if created_at_str else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        created_at = datetime.now(timezone.utc)

    views_raw = raw.get("views")
    impression_count: Optional[int] = _safe_int(views_raw) if views_raw else None

    return NormalizedTweet(
        tweet_id=str(raw.get("tweet_id", "")),
        author_id=str(raw.get("user_id", "")) or None,
        author_handle=raw.get("screen_name", "") or user_info.get("screen_name", ""),
        author_followers=_safe_int(user_info.get("followers_count")),
        author_bio=user_info.get("description", "") or "",
        text=raw.get("text", ""),
        created_at=created_at,
        like_count=_safe_int(raw.get("favorites")),
        reply_count=_safe_int(raw.get("replies")),
        retweet_count=_safe_int(raw.get("retweets")),
        quote_count=_safe_int(raw.get("quotes")),
        impression_count=impression_count,
        lang=raw.get("lang"),
        conversation_id=None,  # twitter-api45 doesn't expose this directly
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class TwitterApi45Provider(ScraperProvider):
    BASE_URL = BASE_URL

    def __init__(self) -> None:
        self.host = os.environ.get("RAPIDAPI_HOST", "twitter-api45.p.rapidapi.com")
        self.key = os.environ.get("RAPIDAPI_KEY", "")
        if not self.key:
            raise RuntimeError("RAPIDAPI_KEY is empty. Set it in .env.")
        self._session = requests.Session()
        self._session.headers.update({
            "x-rapidapi-host": self.host,
            "x-rapidapi-key": self.key,
        })

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_or_grouping_in_query=True,
            supports_min_faves=False,
            supports_exclude_retweets=False,
            supports_exclude_replies=False,
            supports_lang_in_query=False,
            supports_from_handle_in_query=False,
            supports_conversation_id_in_query=False,
            supports_since_id=False,
            rate_limit_header_remaining="x-ratelimit-requests-remaining",
            rate_limit_header_reset="x-ratelimit-requests-reset",
            notes=(
                "Raw query string is passed to X's underlying search. "
                "OR grouping like (kw1 OR kw2) works; filter operators "
                "(min_faves, -is:retweet, lang:en, conversation_id, from:) "
                "DO NOT work and must be applied post-hoc in Python. "
                "Cursor-based pagination only — no since_id."
            ),
        )

    def search_recent(
        self,
        query: str,
        since_id: Optional[str] = None,
        max_results: int = 100,
    ) -> list[NormalizedTweet]:
        cleaned_query, filters = _strip_unsupported_operators(query)
        cursor: str | None = None
        all_tweets: list[dict] = []
        pages = 0

        while len(all_tweets) < max_results and pages < 5:
            params: dict = {"query": cleaned_query, "search_type": "Latest"}
            if cursor:
                params["cursor"] = cursor

            resp = safe_get(self._session, f"{self.BASE_URL}/search.php", params)
            data = resp.json()

            page_tweets = data.get("timeline") or []
            if not page_tweets:
                break

            # since_id emulation: filter by tweet_id comparison (IDs are snowflake-ordered)
            if since_id:
                page_tweets = [
                    t for t in page_tweets
                    if str(t.get("tweet_id", "")) > since_id
                ]

            all_tweets.extend(page_tweets)
            cursor = data.get("next_cursor")
            if not cursor:
                break
            pages += 1

        filtered = _apply_posthoc_filters(all_tweets, filters)
        return [_normalize(t) for t in filtered[:max_results]]

    def user_timeline(self, handle: str, max_results: int = 10) -> list[NormalizedTweet]:
        handle = handle.lstrip("@")
        resp = safe_get(
            self._session,
            f"{self.BASE_URL}/timeline.php",
            params={"screenname": handle},
        )
        data = resp.json()
        tweets = (data.get("timeline") or [])[:max_results]
        return [_normalize(t) for t in tweets]

    def conversation_replies(
        self, tweet_id: str, max_pages: int = 3
    ) -> list[NormalizedTweet]:
        cursor: str | None = None
        all_replies: list[dict] = []

        for _ in range(max_pages):
            params: dict = {"id": str(tweet_id)}
            if cursor:
                params["cursor"] = cursor

            resp = safe_get(
                self._session,
                f"{self.BASE_URL}/latest_replies.php",
                params,
            )
            data = resp.json()
            replies = data.get("timeline") or data.get("replies") or []
            if not replies:
                break
            all_replies.extend(replies)
            cursor = data.get("next_cursor")
            if not cursor:
                break

        return [_normalize(t) for t in all_replies]

    def get_rate_limit_status(
        self, response_headers: dict
    ) -> tuple[Optional[int], Optional[datetime]]:
        remaining = response_headers.get("x-ratelimit-requests-remaining")
        reset = response_headers.get("x-ratelimit-requests-reset")
        try:
            remaining = int(remaining) if remaining else None
        except (TypeError, ValueError):
            remaining = None
        try:
            reset_dt = datetime.fromtimestamp(int(reset), tz=timezone.utc) if reset else None
        except (TypeError, ValueError):
            reset_dt = None
        return remaining, reset_dt
