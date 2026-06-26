"""
Provider abstraction tests — no API calls, no keys required.
Run: pytest tests/test_provider_abstraction.py -v
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# test_provider_factory_validates_env
# ---------------------------------------------------------------------------

def test_provider_factory_validates_env():
    """Bad SCRAPER_PROVIDER value should raise ValueError at settings import."""
    import importlib
    import sys

    # Evict the cached module so the next import re-runs validation
    sys.modules.pop("shared.config.settings", None)
    with patch.dict(os.environ, {"SCRAPER_PROVIDER": "made_up_provider"}):
        with pytest.raises(ValueError, match="SCRAPER_PROVIDER"):
            import shared.config.settings  # noqa: F401

    # Restore a valid cached module for subsequent tests
    sys.modules.pop("shared.config.settings", None)
    with patch.dict(os.environ, {"SCRAPER_PROVIDER": "x_official"}):
        import shared.config.settings  # noqa: F401


# ---------------------------------------------------------------------------
# test_x_official_capabilities
# ---------------------------------------------------------------------------

def test_x_official_capabilities():
    """XOfficialProvider declares full native operator support."""
    with patch.dict(os.environ, {"X_BEARER_TOKEN": "fake_token_for_test"}):
        from agents.brand_visibility.x.providers.x_official import XOfficialProvider
        provider = XOfficialProvider()
        caps = provider.capabilities
        assert caps.supports_or_grouping_in_query is True
        assert caps.supports_min_faves is True
        assert caps.supports_exclude_retweets is True
        assert caps.supports_exclude_replies is True
        assert caps.supports_lang_in_query is True
        assert caps.supports_from_handle_in_query is True
        assert caps.supports_conversation_id_in_query is True
        assert caps.supports_since_id is True


# ---------------------------------------------------------------------------
# test_twitter_api45_capabilities
# ---------------------------------------------------------------------------

def test_twitter_api45_capabilities():
    """TwitterApi45Provider correctly declares its limitations."""
    with patch.dict(os.environ, {"RAPIDAPI_KEY": "fake_key_for_test"}):
        from agents.brand_visibility.x.providers.twitter_api45 import TwitterApi45Provider
        provider = TwitterApi45Provider()
        caps = provider.capabilities
        assert caps.supports_or_grouping_in_query is True
        assert caps.supports_min_faves is False
        assert caps.supports_exclude_retweets is False
        assert caps.supports_exclude_replies is False
        assert caps.supports_lang_in_query is False
        assert caps.supports_from_handle_in_query is False
        assert caps.supports_conversation_id_in_query is False
        assert caps.supports_since_id is False


# ---------------------------------------------------------------------------
# test_strip_unsupported_operators
# ---------------------------------------------------------------------------

def test_strip_unsupported_operators():
    """Operators are stripped cleanly; core query survives intact."""
    from agents.brand_visibility.x.providers.twitter_api45 import _strip_unsupported_operators

    query = "voice AI latency min_faves:1 -is:retweet -is:reply lang:en"
    cleaned, filters = _strip_unsupported_operators(query)

    assert cleaned == "voice AI latency"
    assert "min_faves" in filters
    assert filters["min_faves"] == "1"
    assert "is_retweet" in filters
    assert "is_reply" in filters
    assert "lang" in filters
    assert filters["lang"] == "en"


def test_strip_unsupported_operators_no_operators():
    """Query with no unsupported operators is returned unchanged."""
    from agents.brand_visibility.x.providers.twitter_api45 import _strip_unsupported_operators

    query = "(voice AI OR speech synthesis) developer"
    cleaned, filters = _strip_unsupported_operators(query)

    assert cleaned == query
    assert filters == {}


# ---------------------------------------------------------------------------
# test_posthoc_filters_min_faves
# ---------------------------------------------------------------------------

def test_posthoc_filters_min_faves():
    """Tweets with favorites below threshold are dropped."""
    from agents.brand_visibility.x.providers.twitter_api45 import _apply_posthoc_filters

    tweets = [
        {"tweet_id": "1", "favorites": 0, "text": "low engagement"},
        {"tweet_id": "2", "favorites": 5, "text": "below threshold"},
        {"tweet_id": "3", "favorites": 10, "text": "at threshold"},
        {"tweet_id": "4", "favorites": 100, "text": "above threshold"},
    ]
    result = _apply_posthoc_filters(tweets, {"min_faves": "10"})

    ids = [t["tweet_id"] for t in result]
    assert "1" not in ids
    assert "2" not in ids
    assert "3" in ids
    assert "4" in ids


# ---------------------------------------------------------------------------
# test_posthoc_filters_retweets
# ---------------------------------------------------------------------------

def test_posthoc_filters_retweets():
    """Tweets starting with 'RT @' are dropped when is_retweet filter active."""
    from agents.brand_visibility.x.providers.twitter_api45 import _apply_posthoc_filters

    tweets = [
        {"tweet_id": "1", "text": "RT @someone: this is a retweet"},
        {"tweet_id": "2", "text": "Original thought about voice AI"},
        {"tweet_id": "3", "text": "RT @other: another retweet"},
        {"tweet_id": "4", "text": "Not a retweet"},
    ]
    result = _apply_posthoc_filters(tweets, {"is_retweet": True})

    ids = [t["tweet_id"] for t in result]
    assert "1" not in ids
    assert "3" not in ids
    assert "2" in ids
    assert "4" in ids


# ---------------------------------------------------------------------------
# test_normalize_handles_string_views
# ---------------------------------------------------------------------------

def test_normalize_handles_string_views():
    """'views' as a numeric string becomes impression_count int; empty string → None."""
    from agents.brand_visibility.x.providers.twitter_api45 import _normalize

    base = {
        "tweet_id": "9999",
        "screen_name": "testuser",
        "user_id": "111",
        "text": "test tweet",
        "created_at": "Tue, 12 Mar 2024 04:12:30 +0000",
        "favorites": 5,
        "retweets": 1,
        "replies": 2,
        "quotes": 0,
        "lang": "en",
    }

    tweet_with_views = {**base, "views": "12345"}
    nt = _normalize(tweet_with_views)
    assert nt.impression_count == 12345

    tweet_empty_views = {**base, "views": ""}
    nt2 = _normalize(tweet_empty_views)
    assert nt2.impression_count is None

    tweet_no_views = {**base}
    nt3 = _normalize(tweet_no_views)
    assert nt3.impression_count is None


# ---------------------------------------------------------------------------
# test_normalize_handles_rfc822_dates
# ---------------------------------------------------------------------------

def test_normalize_handles_rfc822_dates():
    """RFC 822 created_at string is parsed into a timezone-aware datetime."""
    from agents.brand_visibility.x.providers.twitter_api45 import _normalize

    raw = {
        "tweet_id": "8888",
        "screen_name": "testuser",
        "user_id": "222",
        "text": "date test",
        "created_at": "Tue, 12 Mar 2024 04:12:30 +0000",
        "favorites": 0,
        "retweets": 0,
        "replies": 0,
        "quotes": 0,
        "lang": "en",
    }
    nt = _normalize(raw)

    assert nt.created_at.year == 2024
    assert nt.created_at.month == 3
    assert nt.created_at.day == 12
    assert nt.created_at.hour == 4
    assert nt.created_at.minute == 12
    assert nt.created_at.second == 30
    assert nt.created_at.tzinfo is not None


def test_normalize_handles_bad_date_fallback():
    """Unparseable created_at falls back to a timezone-aware now()."""
    from agents.brand_visibility.x.providers.twitter_api45 import _normalize

    raw = {
        "tweet_id": "7777",
        "screen_name": "testuser",
        "user_id": "333",
        "text": "bad date test",
        "created_at": "not-a-date",
        "favorites": 0,
        "retweets": 0,
        "replies": 0,
        "quotes": 0,
    }
    before = datetime.now(timezone.utc)
    nt = _normalize(raw)
    after = datetime.now(timezone.utc)

    assert nt.created_at.tzinfo is not None
    assert before <= nt.created_at <= after
