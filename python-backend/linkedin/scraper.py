"""
Fresh LinkedIn Scraper API (Apidojo) provider — KA018.

RapidAPI host: fresh-linkedin-scraper-api.p.rapidapi.com
Endpoint:      GET /api/v1/search/posts
Auth:          x-rapidapi-key + x-rapidapi-host headers

The exact response shape isn't documented yet, so extraction is intentionally
defensive: we probe several plausible keys for the posts array and each field,
logging top-level keys when the array can't be located. Mirrors KA017's
twitter241 conventions: stdlib logging, tenacity retries on transient errors
only (429/5xx/timeout, never 4xx), env-driven config, conservative sleeps.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven, sensible defaults)
# ---------------------------------------------------------------------------

DEFAULT_HOST = "fresh-linkedin-scraper-api.p.rapidapi.com"
RAPIDAPI_KEY = os.getenv("LINKEDIN_RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("LINKEDIN_RAPIDAPI_HOST", DEFAULT_HOST)
BASE_URL = f"https://{RAPIDAPI_HOST}"
SEARCH_POSTS_PATH = "/api/v1/search/posts"

# Conservative sleeps (seconds) — configurable
SLEEP_BETWEEN_PAGES = float(os.getenv("LINKEDIN_SLEEP_PAGINATED", "2"))
SLEEP_BETWEEN_QUERIES = float(os.getenv("LINKEDIN_SLEEP_QUERY", "3"))
HTTP_TIMEOUT = int(os.getenv("LINKEDIN_HTTP_TIMEOUT", "30"))

_VALID_DATE_POSTED = {"past_24h", "past_week", "past_month"}
_VALID_SORT_BY = {"date_posted", "relevance"}

# Keys we probe, in order, when hunting for the posts array in a response.
_POSTS_ARRAY_KEYS = ("posts", "data", "results", "items", "elements")
# Nested containers a posts array might hide under (e.g. {"data": {"posts": [...]}})
_NESTED_CONTAINER_KEYS = ("data", "result", "response")


# ---------------------------------------------------------------------------
# HTTP — transient-only retry (never retry 4xx)
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    pass


class ServerError(Exception):
    pass


_session = requests.Session()


@retry(
    retry=retry_if_exception_type((RateLimitError, ServerError, requests.exceptions.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=900),
    reraise=True,
)
def _safe_get(url: str, params: dict, headers: dict) -> requests.Response:
    """GET with retries on transient errors only. 4xx (incl. auth) raises immediately."""
    resp = _session.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)

    if resp.status_code == 429:
        reset = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-requests-reset")
        if reset:
            try:
                sleep_for = max(5, int(reset) - int(time.time()) + 5)
                logger.warning("Rate limited. Sleeping %d s.", min(sleep_for, 900))
                time.sleep(min(sleep_for, 900))
            except (TypeError, ValueError):
                pass
        raise RateLimitError(f"429 from {url}")

    if 500 <= resp.status_code < 600:
        raise ServerError(f"{resp.status_code} from {url}")

    resp.raise_for_status()  # other 4xx -> HTTPError, not retried
    return resp


def _headers() -> dict:
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }


# ---------------------------------------------------------------------------
# Response extraction (defensive)
# ---------------------------------------------------------------------------


def _extract_posts_array(payload: Any) -> list[dict]:
    """Find the list of post objects in an undocumented response shape.

    Probes top-level keys, then a few nested containers. Logs a warning with
    the available top-level keys if nothing matches so we can wire the real
    path later.
    """
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]

    if not isinstance(payload, dict):
        logger.warning("LinkedIn posts: unexpected payload type %s", type(payload))
        return []

    # Direct top-level array
    for key in _POSTS_ARRAY_KEYS:
        val = payload.get(key)
        if isinstance(val, list):
            return [p for p in val if isinstance(p, dict)]

    # One level of nesting (e.g. {"data": {"posts": [...]}})
    for container in _NESTED_CONTAINER_KEYS:
        inner = payload.get(container)
        if isinstance(inner, dict):
            for key in _POSTS_ARRAY_KEYS:
                val = inner.get(key)
                if isinstance(val, list):
                    return [p for p in val if isinstance(p, dict)]
        elif isinstance(inner, list):
            return [p for p in inner if isinstance(p, dict)]

    logger.warning(
        "LinkedIn posts array not found. Top-level keys: %s",
        list(payload.keys()),
    )
    return []


def _first(raw: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among keys."""
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


def _to_int(v: Any) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0


def normalize_post(
    raw: dict,
    matched_keyword: str,
    query_string: str,
    source_class: str,
    matched_category: str,
) -> dict:
    """Map one undocumented LinkedIn post object to KA018's row shape.

    Uses defensive .get()/_first() across multiple plausible key names because
    the provider's response schema isn't documented yet.
    """
    author = raw.get("author") or raw.get("actor") or raw.get("user") or {}
    if not isinstance(author, dict):
        author = {}
    # Engagement counts are nested under "activity" in this provider's shape.
    activity = raw.get("activity") or {}
    if not isinstance(activity, dict):
        activity = {}

    post_urn = _first(raw, "urn", "post_urn", "id", "activity_urn", "entity_urn")

    return {
        "post_urn": str(post_urn) if post_urn is not None else None,
        "author_name": _first(
            raw, "author_name", default=_first(author, "name", "full_name", "title", default="")
        ),
        # Headline lives in author.description for this provider.
        "author_headline": _first(
            raw, "author_headline",
            default=_first(author, "description", "headline", "subtitle", default=""),
        ),
        "author_urn": _first(
            raw, "author_urn", "member_urn",
            default=_first(author, "urn", "id", "member_urn", default=None),
        ),
        "author_profile_url": _first(
            raw, "author_url", "profile_url",
            default=_first(author, "profile_url", "url", "public_profile_url", default=""),
        ),
        "author_followers": _to_int(
            _first(raw, "author_followers", default=_first(author, "followers", "follower_count"))
        ),
        # Post body lives in "title" for this provider; keep other keys as fallbacks.
        "text": _first(raw, "title", "text", "commentary", "content", "body", default=""),
        "posted_at": _first(
            raw, "posted_at", "published_at", "created_at", "date_posted", "time", default=None
        ),
        # Prefer nested activity.*, fall back to any top-level variants.
        "like_count": _to_int(_first(
            activity, "num_likes", "num_reactions",
            default=_first(raw, "like_count", "likes", "num_likes", "reactions", "num_reactions"),
        )),
        "comment_count": _to_int(_first(
            activity, "num_comments",
            default=_first(raw, "comment_count", "comments", "num_comments"),
        )),
        "repost_count": _to_int(_first(
            activity, "num_shares", "num_reposts",
            default=_first(raw, "repost_count", "reposts", "shares", "num_shares"),
        )),
        "post_url": _first(raw, "post_url", "url", "permalink", "share_url", default=""),
        "matched_keyword": matched_keyword,
        "query_string": query_string,
        "source_class": source_class,
        "matched_category": matched_category,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_posts(
    keyword: str,
    date_posted: Optional[str] = None,
    sort_by: str = "date_posted",
    max_pages: int = 1,
    from_member: Optional[str] = None,
) -> list[dict]:
    """Search LinkedIn posts for a keyword.

    Args:
        keyword: search term.
        date_posted: one of 'past_24h', 'past_week', 'past_month', or None.
        sort_by: 'date_posted' or 'relevance'.
        max_pages: number of result pages to fetch (1 page per API call).
        from_member: comma-separated LinkedIn member IDs (watchlist filter).

    Returns:
        A list of raw post dicts (as returned by the provider). Call
        normalize_post() on each to map into KA018's row shape.

    Raises:
        RuntimeError if the API key is missing.
        requests.HTTPError on non-transient (4xx) responses.
    """
    if not RAPIDAPI_KEY:
        raise RuntimeError("LINKEDIN_RAPIDAPI_KEY is empty — set it in .env")

    if date_posted is not None and date_posted not in _VALID_DATE_POSTED:
        raise ValueError(
            f"date_posted must be one of {sorted(_VALID_DATE_POSTED)} or None, got {date_posted!r}"
        )
    if sort_by not in _VALID_SORT_BY:
        raise ValueError(f"sort_by must be one of {sorted(_VALID_SORT_BY)}, got {sort_by!r}")

    url = f"{BASE_URL}{SEARCH_POSTS_PATH}"
    all_posts: list[dict] = []

    for page in range(1, max_pages + 1):
        params: dict[str, Any] = {
            "keyword": keyword,
            "sort_by": sort_by,
            "page": page,
        }
        if date_posted:
            params["date_posted"] = date_posted
        if from_member:
            params["from_member"] = from_member

        logger.info("LinkedIn search: keyword=%r page=%d sort_by=%s date=%s",
                    keyword, page, sort_by, date_posted)
        resp = _safe_get(url, params=params, headers=_headers())

        try:
            payload = resp.json()
        except ValueError:
            logger.warning("LinkedIn search: non-JSON response (status %s)", resp.status_code)
            break

        page_posts = _extract_posts_array(payload)
        if not page_posts:
            logger.info("LinkedIn search: page %d returned 0 posts — stopping pagination", page)
            break

        all_posts.extend(page_posts)
        if page < max_pages:
            time.sleep(SLEEP_BETWEEN_PAGES)

    return all_posts


def sleep_between_queries() -> None:
    """Conservative sleep between different keyword queries (configurable via env)."""
    time.sleep(SLEEP_BETWEEN_QUERIES)
