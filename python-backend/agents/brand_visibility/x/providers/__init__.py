from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NormalizedTweet:
    """Provider-agnostic tweet shape. Every provider maps to this."""
    tweet_id: str
    author_id: Optional[str]
    author_handle: str
    author_followers: int
    author_bio: str
    text: str
    created_at: datetime
    like_count: int
    reply_count: int
    retweet_count: int
    quote_count: int
    impression_count: Optional[int]   # twitter-api45 returns this as 'views' (string)
    lang: Optional[str]
    conversation_id: Optional[str]    # may be None for twitter-api45
    raw: dict                          # original API response for debugging


@dataclass
class ProviderCapabilities:
    supports_or_grouping_in_query: bool
    supports_min_faves: bool
    supports_exclude_retweets: bool
    supports_exclude_replies: bool
    supports_lang_in_query: bool
    supports_from_handle_in_query: bool   # `from:HANDLE` as a search operator
    supports_conversation_id_in_query: bool
    supports_since_id: bool
    rate_limit_header_remaining: str       # name of remaining-quota header
    rate_limit_header_reset: str           # name of reset-timestamp header
    notes: str


class ScraperProvider(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def search_recent(
        self,
        query: str,
        since_id: Optional[str] = None,
        max_results: int = 100,
    ) -> list[NormalizedTweet]: ...

    @abstractmethod
    def user_timeline(
        self,
        handle: str,
        max_results: int = 10,
    ) -> list[NormalizedTweet]: ...

    @abstractmethod
    def conversation_replies(
        self,
        tweet_id: str,
        max_pages: int = 3,
    ) -> list[NormalizedTweet]: ...

    @abstractmethod
    def get_rate_limit_status(
        self,
        response_headers: dict,
    ) -> tuple[Optional[int], Optional[datetime]]: ...
