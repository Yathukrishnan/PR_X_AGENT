"""
X scraper — provider-agnostic. Backend selected via SCRAPER_PROVIDER env var.

Usage:
  python -m agents.brand_visibility.x.x_scraper --mode keywords
  python -m agents.brand_visibility.x.x_scraper --mode influencers
  python -m agents.brand_visibility.x.x_scraper --mode replies
  python -m agents.brand_visibility.x.x_scraper --mode keywords --dry-run
  python -m agents.brand_visibility.x.x_scraper --mode keywords --limit 3
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

# Force load the .env file
load_dotenv()

from shared.config.settings import (
    MAX_REPLY_EXPANSIONS_PER_TICK,
    MAX_REPLY_TREE_PAGES,
    SCRAPE_SLEEP_SECONDS,
    SCRAPER_PROVIDER,
    TIER_1_EVERY_N_TICKS,
    TIER_2_EVERY_N_TICKS,
    TIER_3_EVERY_N_TICKS,
)
from agents.brand_visibility.x.db import Database
from agents.brand_visibility.x.lexicon import load as load_lexicon
from agents.brand_visibility.x.providers import NormalizedTweet, ScraperProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider factory — lazy singleton
# ---------------------------------------------------------------------------

_provider_instance: ScraperProvider | None = None

_call_count_this_run: int = 0
_call_budget: int | None = None  # None = unlimited


def reset_call_budget(budget: int) -> None:
    global _call_count_this_run, _call_budget
    _call_count_this_run = 0
    _call_budget = budget


def _budget_ok() -> bool:
    if _call_budget is None:
        return True
    return _call_count_this_run < _call_budget


def _record_api_call() -> None:
    global _call_count_this_run
    _call_count_this_run += 1


def _load_provider() -> ScraperProvider:
    provider_name = os.environ.get("SCRAPER_PROVIDER", SCRAPER_PROVIDER)
    logger.info(f"DEBUG: Attempting to load provider: {provider_name}")
    if provider_name == "x_official":
        from agents.brand_visibility.x.providers.x_official import XOfficialProvider
        return XOfficialProvider()
    if provider_name == "twitter_api45":
        from agents.brand_visibility.x.providers.twitter_api45 import TwitterApi45Provider
        return TwitterApi45Provider()
    if provider_name == "twitter241":
        from agents.brand_visibility.x.providers.twitter241 import Twitter241Provider
        return Twitter241Provider()
    raise ValueError(f"Unknown SCRAPER_PROVIDER={provider_name!r}")


def _get_provider() -> ScraperProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = _load_provider()
    return _provider_instance


# ---------------------------------------------------------------------------
# Velocity + priority (provider-agnostic — uses NormalizedTweet fields)
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def calculate_velocity(
    like_count: int,
    retweet_count: int,
    reply_count: int,
    quote_count: int,
    created_at: datetime,
) -> float:
    minutes_elapsed = max(1.0, (_now_utc() - created_at).total_seconds() / 60.0)
    engagement = like_count + 2 * retweet_count + 3 * reply_count + 4 * quote_count
    return engagement / minutes_elapsed


def assign_priority(
    source_type: str,
    matched_class: str,
    velocity: float,
    created_at: datetime,
) -> str:
    if source_type == "INFLUENCER_REPLY":
        return "URGENT_INFLUENCER_REPLY"
    minutes_elapsed = max(1.0, (_now_utc() - created_at).total_seconds() / 60.0)
    if velocity > 50 and minutes_elapsed < 120:
        return "URGENT_VIRAL"
    if matched_class == "G":
        return "LOW_PRIORITY_CONTENT"
    return "STANDARD"


def _to_db_dict(
    tweet: NormalizedTweet,
    source_type: str,
    matched_class: str = "",
    matched_query: str = "",
    source_handle: str = "",
    override_priority: str | None = None,
) -> dict[str, Any]:
    """Convert NormalizedTweet + context metadata to the dict expected by db.upsert_tweet."""
    velocity = calculate_velocity(
        tweet.like_count,
        tweet.retweet_count,
        tweet.reply_count,
        tweet.quote_count,
        tweet.created_at,
    )
    priority = override_priority or assign_priority(source_type, matched_class, velocity, tweet.created_at)
    created_at_str = tweet.created_at.isoformat()
    if tweet.created_at.tzinfo is None:
        created_at_str = tweet.created_at.replace(tzinfo=timezone.utc).isoformat()

    return {
        "tweet_id": tweet.tweet_id,
        "created_at": created_at_str,
        "author_id": tweet.author_id or "",
        "author_handle": tweet.author_handle,
        "author_followers": tweet.author_followers,
        "author_bio": tweet.author_bio,
        "text": tweet.text,
        "like_count": tweet.like_count,
        "reply_count": tweet.reply_count,
        "retweet_count": tweet.retweet_count,
        "quote_count": tweet.quote_count,
        "impression_count": tweet.impression_count,
        "lang": tweet.lang or "",
        "source_type": source_type,
        "matched_class": matched_class,
        "matched_query": matched_query,
        "source_handle": source_handle,
        "conversation_id": tweet.conversation_id or "",
        "velocity": velocity,
        "priority_flag": priority,
    }


import time as _time
import re as _re


def _apply_since_time(query: str) -> str:
    """Substitute {{since_time}} placeholder per env var, or strip the clause.

    Reads KA017_SINCE_HOURS env var. If set to a positive integer N, replaces
    {{since_time}} with the current Unix timestamp minus N*3600 seconds.
    If unset or invalid, strips the entire `since_time:{{since_time}}` clause
    so queries without time-filter intent still work.
    """
    if "{{since_time}}" not in query:
        return query

    hours_str = os.environ.get("KA017_SINCE_HOURS", "").strip()
    if hours_str:
        try:
            hours = int(hours_str)
            if hours > 0:
                cutoff = int(_time.time()) - (hours * 3600)
                return query.replace("{{since_time}}", str(cutoff))
        except ValueError:
            logger.warning("KA017_SINCE_HOURS=%r is not a valid integer; stripping placeholder", hours_str)

    # Strip the entire `since_time:{{since_time}}` clause + any leading space
    return _re.sub(r"\s*since_time:\{\{since_time\}\}", "", query)


# ---------------------------------------------------------------------------
# Mode 1: Keyword sweep
# ---------------------------------------------------------------------------

def sweep_keywords(db: Database, dry_run: bool = False, limit: int | None = None, run_id: int | None = None) -> dict[str, int]:
    provider = _get_provider()
    lexicon = load_lexicon(db)
    stats = {"new": 0, "updated": 0, "api_calls": 0}
    if run_id:
        db.log_activity(run_id, phase="scrape", event="sweep_keywords_start",
                        message=f"source={lexicon.get('source', 'turso')}")

    count = 0
    class_filter_env = os.environ.get("KA017_CLASS_FILTER", "").strip()
    allowed_classes = (
        {c.strip() for c in class_filter_env.split(",") if c.strip()}
        if class_filter_env else None
    )
    if allowed_classes is not None:
        logger.info("sweep_keywords class filter active: %s", sorted(allowed_classes))

    for cls_id, cls_data in lexicon["Keyword_Classes"].items():
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue
        if not dry_run and not _budget_ok():
            logger.info(
                "API budget exhausted (%d/%d) -- stopping keyword sweep",
                _call_count_this_run, _call_budget,
            )
            break
        for query_str in cls_data["queries"]:
            if limit is not None and count >= limit:
                break
            count += 1

            query_hash = hashlib.sha256(query_str.encode()).hexdigest()

            if dry_run:
                if SCRAPER_PROVIDER == "twitter_api45":
                    from agents.brand_visibility.x.providers.twitter_api45 import _strip_unsupported_operators
                    cleaned, filters = _strip_unsupported_operators(query_str)
                    logger.info(
                        "[DRY-RUN][%s] class %s | cleaned: %s... | stripped: %s",
                        SCRAPER_PROVIDER, cls_id, cleaned[:60], list(filters.keys()),
                    )
                else:
                    logger.info(
                        "[DRY-RUN][%s] class %s: %s...",
                        SCRAPER_PROVIDER, cls_id, query_str[:80],
                    )
                continue

            state = db.get_query_state(query_hash)
            since_id = state["last_since_id"] if state else None

            if not _budget_ok():
                logger.info(
                    "API budget exhausted (%d/%d) -- stopping keyword sweep",
                    _call_count_this_run, _call_budget,
                )
                break
            _record_api_call()

            try:
                # DEBUG LOGGING ADDED HERE
                logger.info(f"DEBUG: Initiating API request to provider for class {cls_id} with query starting: {query_str[:30]}...")

                # Execute the API call
                sweep_type = os.environ.get("KA017_SWEEP_TYPE", "Latest")
                max_pages_env = int(os.environ.get("KA017_MAX_PAGES", "1"))
                final_query = _apply_since_time(query_str)
                if final_query != query_str:
                    logger.debug("Applied since_time substitution: %s -> %s", query_str[:80], final_query[:80])
                tweets = provider.search_recent(
                    final_query, since_id=since_id,
                    sweep_type=sweep_type, max_pages=max_pages_env,
                )
                
                # Success log
                logger.info(f"DEBUG: Successfully received {len(tweets)} tweets from provider.")
                
            except Exception as exc:
                logger.error("keyword query failed (class %s): %s", cls_id, exc)
                db.log_api_call(
                    endpoint="search_recent",
                    query_text=query_str[:500],
                    status_code=0,
                    tweets_returned=0,
                    rate_remaining=None,
                    rate_reset_at=None,
                    notes=str(exc)[:200],
                )
                time.sleep(SCRAPE_SLEEP_SECONDS)
                continue

            stats["api_calls"] += 1
            new_since_id = ""

            for tw in tweets:
                db_dict = _to_db_dict(tw, source_type="KEYWORD",
                                       matched_class=cls_id, matched_query=query_str)
                db.upsert_tweet(db_dict)
                if tw.tweet_id > new_since_id:
                    new_since_id = tw.tweet_id
                stats["new"] += 1

            db.log_api_call(
                endpoint="search_recent",
                query_text=query_str[:500],
                status_code=200,
                tweets_returned=len(tweets),
                rate_remaining=None,
                rate_reset_at=None,
            )

            if new_since_id:
                db.set_query_state(query_hash, query_str, new_since_id, len(tweets))

            logger.info("DEBUG: Sleeping to respect rate limits...")
            time.sleep(SCRAPE_SLEEP_SECONDS)

    if run_id:
        db.log_activity(run_id, phase="scrape", event="sweep_keywords_done",
                        message=str(stats), meta=stats)
    logger.info("keyword sweep done [%s]: %s", SCRAPER_PROVIDER, stats)
    return stats


# ---------------------------------------------------------------------------
# Mode 2: Influencer sweep
# ---------------------------------------------------------------------------

def sweep_influencers(db: Database, tick_number: int = 0, dry_run: bool = False, run_id: int | None = None) -> dict[str, int]:
    provider = _get_provider()
    lexicon = load_lexicon(db)
    handles = lexicon["Tracked_Handles"]
    stats = {"new": 0, "api_calls": 0}
    if run_id:
        db.log_activity(run_id, phase="scrape", event="sweep_influencers_start",
                        message=f"tick={tick_number} source={lexicon.get('source', 'turso')}")

    active_handles: list[str] = []
    if tick_number % TIER_1_EVERY_N_TICKS == 0:
        active_handles += handles.get("tier_1", [])
    if tick_number % TIER_2_EVERY_N_TICKS == 0:
        active_handles += handles.get("tier_2", [])
    if tick_number % TIER_3_EVERY_N_TICKS == 0:
        active_handles += handles.get("tier_3", [])

    for handle in active_handles:
        if dry_run:
            logger.info("[DRY-RUN][%s] Would fetch timeline: %s", SCRAPER_PROVIDER, handle)
            continue

        if not _budget_ok():
            logger.info(
                "API budget exhausted (%d/%d) -- stopping influencer sweep",
                _call_count_this_run, _call_budget,
            )
            break
        _record_api_call()

        try:
            cached_uid = db.get_user_id(handle)
            if cached_uid and hasattr(provider, "user_timeline_by_id"):
                logger.info("DEBUG: Using cached user_id for %s: %s", handle, cached_uid)
                tweets = provider.user_timeline_by_id(cached_uid, max_results=10)
            else:
                logger.info("DEBUG: Fetching timeline for influencer: %s...", handle)
                tweets = provider.user_timeline(handle, max_results=10)
                if not cached_uid and hasattr(provider, "resolve_user_id"):
                    resolved = provider.resolve_user_id(handle)
                    if resolved:
                        db.set_user_id(handle, resolved)
            logger.info("DEBUG: Successfully fetched timeline for %s.", handle)
        except Exception as exc:
            logger.error("influencer timeline failed for %s: %s", handle, exc)
            time.sleep(SCRAPE_SLEEP_SECONDS)
            continue

        stats["api_calls"] += 1
        for tw in tweets:
            db_dict = _to_db_dict(tw, source_type="INFLUENCER_POST", source_handle=handle)
            db.upsert_tweet(db_dict)
            stats["new"] += 1

        db.log_api_call(
            endpoint="user_timeline",
            query_text=handle,
            status_code=200,
            tweets_returned=len(tweets),
            rate_remaining=None,
            rate_reset_at=None,
        )
        time.sleep(SCRAPE_SLEEP_SECONDS)

    if run_id:
        db.log_activity(run_id, phase="scrape", event="sweep_influencers_done",
                        message=str(stats), meta=stats)
    logger.info("influencer sweep done (tick %d) [%s]: %s", tick_number, SCRAPER_PROVIDER, stats)
    return stats


# ---------------------------------------------------------------------------
# Mode 3: Reply-tree expansion
# ---------------------------------------------------------------------------

def expand_reply_trees(db: Database, dry_run: bool = False, run_id: int | None = None) -> dict[str, int]:
    provider = _get_provider()
    rows = db.query(
        """
        SELECT tweet_id, source_handle FROM scraped_tweets
        WHERE source_type = 'INFLUENCER_POST'
          AND confirmed_class IN ('A','B','C','D','E','F')
          AND quality_score >= 6
          AND status = 'CLASSIFIED'
        ORDER BY quality_score DESC, velocity DESC
        LIMIT ?
        """,
        (MAX_REPLY_EXPANSIONS_PER_TICK,),
    )

    stats = {"reply_tweets": 0, "api_calls": 0}
    if run_id:
        db.log_activity(run_id, phase="scrape", event="expand_reply_trees_start",
                        message=f"candidates={len(rows)}")

    for row in rows:
        tweet_id = row["tweet_id"]
        source_handle = row["source_handle"]

        if dry_run:
            logger.info("[DRY-RUN][%s] Would expand replies for %s", SCRAPER_PROVIDER, tweet_id)
            continue

        if not _budget_ok():
            logger.info(
                "API budget exhausted (%d/%d) -- stopping reply tree sweep",
                _call_count_this_run, _call_budget,
            )
            break
        _record_api_call()

        try:
            logger.info(f"DEBUG: Expanding reply tree for tweet: {tweet_id}...")
            tweets = provider.conversation_replies(tweet_id, max_pages=MAX_REPLY_TREE_PAGES)
            logger.info(f"DEBUG: Successfully expanded reply tree for {tweet_id}.")
        except Exception as exc:
            logger.error("reply expansion failed for %s: %s", tweet_id, exc)
            continue

        stats["api_calls"] += 1
        for tw in tweets:
            db_dict = _to_db_dict(
                tw,
                source_type="INFLUENCER_REPLY",
                source_handle=source_handle,
                override_priority="URGENT_INFLUENCER_REPLY",
            )
            db.upsert_tweet(db_dict)
            stats["reply_tweets"] += 1

        db.log_api_call(
            endpoint="conversation_replies",
            query_text=tweet_id,
            status_code=200,
            tweets_returned=len(tweets),
            rate_remaining=None,
            rate_reset_at=None,
        )
        time.sleep(SCRAPE_SLEEP_SECONDS)

    if run_id:
        db.log_activity(run_id, phase="scrape", event="expand_reply_trees_done",
                        message=str(stats), meta=stats)
    logger.info("reply expansion done [%s]: %s", SCRAPER_PROVIDER, stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KA017 X scraper")
    parser.add_argument("--mode", choices=["keywords", "influencers", "replies"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run (keywords mode)")
    parser.add_argument("--tick", type=int, default=0, help="Tick number (influencer mode)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Read the provider directly from environment to ensure it uses the .env file
    current_provider = os.environ.get("SCRAPER_PROVIDER", SCRAPER_PROVIDER)
    logger.info("Active provider: %s", current_provider)

    if not args.dry_run:
        # Validate credentials before spending time on anything
        try:
            _load_provider()
        except Exception as exc:
            logger.error("Provider init failed: %s", exc)
            raise SystemExit(1)

    db = Database()

    if args.mode == "keywords":
        stats = sweep_keywords(db, dry_run=args.dry_run, limit=args.limit)
    elif args.mode == "influencers":
        stats = sweep_influencers(db, tick_number=args.tick, dry_run=args.dry_run)
    else:
        stats = expand_reply_trees(db, dry_run=args.dry_run)

    print(f"\nScrape complete [{current_provider}]: {stats}")


if __name__ == "__main__":
    main()