"""
X API v2 (official) provider — reference implementation.
All supports_* capabilities are True.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from agents.brand_visibility.x.providers import NormalizedTweet, ProviderCapabilities, ScraperProvider
from agents.brand_visibility.x.providers._http import safe_get

logger = logging.getLogger(__name__)

BASE_URL = "https://api.x.com/2"
TWEET_FIELDS = "id,text,created_at,public_metrics,author_id,conversation_id,lang"
USER_FIELDS = "id,username,name,public_metrics,description"
EXPANSIONS = "author_id"


class XOfficialProvider(ScraperProvider):
    def __init__(self) -> None:
        from shared.config.settings import X_BEARER_TOKEN
        self._bearer_token = X_BEARER_TOKEN
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._bearer_token}"})

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_or_grouping_in_query=True,
            supports_min_faves=True,
            supports_exclude_retweets=True,
            supports_exclude_replies=True,
            supports_lang_in_query=True,
            supports_from_handle_in_query=True,
            supports_conversation_id_in_query=True,
            supports_since_id=True,
            rate_limit_header_remaining="x-rate-limit-remaining",
            rate_limit_header_reset="x-rate-limit-reset",
            notes="Full X API v2 native operator support.",
        )

    def _normalize_response(self, data: dict) -> list[NormalizedTweet]:
        tweets_raw = data.get("data") or []
        users_by_id: dict[str, dict] = {}
        for user in (data.get("includes") or {}).get("users") or []:
            users_by_id[user["id"]] = user

        results: list[NormalizedTweet] = []
        for tw in tweets_raw:
            author = users_by_id.get(tw.get("author_id", ""), {})
            metrics = tw.get("public_metrics") or {}

            created_str = tw.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)

            results.append(
                NormalizedTweet(
                    tweet_id=tw["id"],
                    author_id=tw.get("author_id"),
                    author_handle=author.get("username", ""),
                    author_followers=(author.get("public_metrics") or {}).get("followers_count", 0),
                    author_bio=author.get("description", "") or "",
                    text=tw.get("text", ""),
                    created_at=created_at,
                    like_count=metrics.get("like_count", 0),
                    reply_count=metrics.get("reply_count", 0),
                    retweet_count=metrics.get("retweet_count", 0),
                    quote_count=metrics.get("quote_count", 0),
                    impression_count=metrics.get("impression_count"),
                    lang=tw.get("lang"),
                    conversation_id=tw.get("conversation_id"),
                    raw=tw,
                )
            )
        return results

    def search_recent(
        self,
        query: str,
        since_id: Optional[str] = None,
        max_results: int = 100,
    ) -> list[NormalizedTweet]:
        params: dict = {
            "query": query,
            "tweet.fields": TWEET_FIELDS,
            "user.fields": USER_FIELDS,
            "expansions": EXPANSIONS,
            "max_results": min(max_results, 100),
        }
        if since_id:
            params["since_id"] = since_id

        resp = safe_get(self._session, f"{BASE_URL}/tweets/search/recent", params)
        return self._normalize_response(resp.json())

    def user_timeline(self, handle: str, max_results: int = 10) -> list[NormalizedTweet]:
        handle = handle.lstrip("@")

        # Step 1: resolve handle to user ID
        resp = safe_get(
            self._session,
            f"{BASE_URL}/users/by/username/{handle}",
            params={"user.fields": USER_FIELDS},
        )
        user_data = resp.json().get("data") or {}
        user_id = user_data.get("id")
        if not user_id:
            logger.warning("Could not resolve handle @%s to a user ID", handle)
            return []

        # Step 2: fetch their recent non-retweet posts
        resp = safe_get(
            self._session,
            f"{BASE_URL}/users/{user_id}/tweets",
            params={
                "tweet.fields": TWEET_FIELDS,
                "user.fields": USER_FIELDS,
                "expansions": EXPANSIONS,
                "max_results": min(max_results, 100),
                "exclude": "retweets,replies",
            },
        )
        data = resp.json()
        # Inject author info into includes so _normalize_response can find it
        if "includes" not in data:
            data["includes"] = {}
        if "users" not in data["includes"]:
            data["includes"]["users"] = [user_data | {"id": user_id}]
        return self._normalize_response(data)

    def conversation_replies(
        self, tweet_id: str, max_pages: int = 3
    ) -> list[NormalizedTweet]:
        query = f"conversation_id:{tweet_id} is:reply"
        all_tweets: list[NormalizedTweet] = []
        next_token: str | None = None

        for _ in range(max_pages):
            params: dict = {
                "query": query,
                "tweet.fields": TWEET_FIELDS,
                "user.fields": USER_FIELDS,
                "expansions": EXPANSIONS,
                "max_results": 100,
            }
            if next_token:
                params["next_token"] = next_token

            resp = safe_get(self._session, f"{BASE_URL}/tweets/search/recent", params)
            data = resp.json()
            all_tweets.extend(self._normalize_response(data))

            next_token = (data.get("meta") or {}).get("next_token")
            if not next_token:
                break

        return all_tweets

    def get_rate_limit_status(
        self, response_headers: dict
    ) -> tuple[Optional[int], Optional[datetime]]:
        remaining = response_headers.get("x-rate-limit-remaining")
        reset_ts = response_headers.get("x-rate-limit-reset")
        try:
            remaining = int(remaining) if remaining is not None else None
        except (TypeError, ValueError):
            remaining = None
        try:
            reset_dt = (
                datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                if reset_ts
                else None
            )
        except (TypeError, ValueError):
            reset_dt = None
        return remaining, reset_dt
