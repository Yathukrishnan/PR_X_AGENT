"""
Read-only Turso queries for the KA017 demo dashboard.

Every function is cached (st.cache_data, 30s TTL) and returns plain Python
dicts/lists so Streamlit can serialise them. The single libSQL connection is
shared via st.cache_resource so we don't re-sync the replica on every call.

This module NEVER writes, NEVER triggers the agent, NEVER calls external APIs.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from shared.config.settings import MAX_API_CALLS_PER_RUN
from agents.brand_visibility.x.db import Database


@st.cache_resource
def get_db() -> Database:
    """One shared, read-only Turso connection for the whole app session."""
    return Database()


def _q(sql: str, params: tuple = ()) -> list[dict]:
    return get_db().query(sql, params)


def format_count(n: int | None) -> str:
    """Format an engagement count with k/m suffix for readability."""
    if n is None:
        return "—"
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}m".replace(".0m", "m")


def _scalar(sql: str, params: tuple = (), default: Any = 0) -> Any:
    rows = _q(sql, params)
    if not rows:
        return default
    val = next(iter(rows[0].values()))
    return val if val is not None else default


# --------------------------------------------------------------------------
# Connection status (sidebar footer)
# --------------------------------------------------------------------------

@st.cache_data(ttl=30)
def connection_ok() -> tuple[bool, str]:
    try:
        n = _scalar("SELECT COUNT(*) FROM scraped_tweets")
        return True, f"{n} tweets"
    except Exception as exc:  # pragma: no cover - demo safety
        return False, str(exc)[:80]


# --------------------------------------------------------------------------
# Page 1 — Overview
# --------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_metrics_overview() -> dict:
    scraped_7d = _scalar(
        "SELECT COUNT(*) FROM scraped_tweets "
        "WHERE ingested_at >= datetime('now','-7 days')"
    )
    scraped_prior_7d = _scalar(
        "SELECT COUNT(*) FROM scraped_tweets "
        "WHERE ingested_at >= datetime('now','-14 days') "
        "AND ingested_at < datetime('now','-7 days')"
    )
    classified_7d = _scalar(
        "SELECT COUNT(*) FROM scraped_tweets "
        "WHERE classified_at >= datetime('now','-7 days')"
    )
    backlog = _scalar("SELECT COUNT(*) FROM scraped_tweets WHERE status = 'PENDING'")
    avg_rel = _scalar(
        "SELECT AVG(relevance_score) FROM scraped_tweets "
        "WHERE classified_at >= datetime('now','-7 days') "
        "AND relevance_score IS NOT NULL",
        default=None,
    )
    # Budget remaining = cap - calls used by the most recent run
    last_calls = _scalar(
        "SELECT calls_used FROM agent_runs ORDER BY id DESC LIMIT 1", default=0
    )
    return {
        "scraped_7d": int(scraped_7d or 0),
        "scraped_delta": int((scraped_7d or 0) - (scraped_prior_7d or 0)),
        "classified_7d": int(classified_7d or 0),
        "backlog": int(backlog or 0),
        "avg_relevance": round(avg_rel) if avg_rel is not None else None,
        "budget_cap": int(MAX_API_CALLS_PER_RUN),
        "budget_remaining": max(0, int(MAX_API_CALLS_PER_RUN) - int(last_calls or 0)),
        "last_calls_used": int(last_calls or 0),
    }


@st.cache_data(ttl=30)
def get_recent_runs(limit: int = 5) -> list[dict]:
    rows = _q(
        "SELECT id, started_at, ended_at, triggered_by, status, calls_used, "
        "records_new, records_updated FROM agent_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    for r in rows:
        tb = r.get("triggered_by") or ""
        r["mode"] = tb.split(":", 1)[1] if ":" in tb else tb
    return rows


@st.cache_data(ttl=30)
def get_top_signals(limit: int = 5) -> list[dict]:
    return _q(
        "SELECT tweet_id, author_handle, text, relevance_score, confirmed_class, "
        "intent_signal, like_count, reply_count, retweet_count, quote_count, "
        "impression_count FROM scraped_tweets WHERE status = 'CLASSIFIED' "
        "AND relevance_score IS NOT NULL "
        "ORDER BY relevance_score DESC, classified_at DESC LIMIT ?",
        (limit,),
    )


# --------------------------------------------------------------------------
# Page 2 — Signal Feed
# --------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_distinct(column: str) -> list[str]:
    allowed = {"confirmed_class", "intent_signal", "source_type"}
    if column not in allowed:
        return []
    rows = _q(
        f"SELECT DISTINCT {column} AS v FROM scraped_tweets "
        f"WHERE {column} IS NOT NULL AND {column} != '' ORDER BY v"
    )
    return [r["v"] for r in rows]


@st.cache_data(ttl=30)
def get_signal_feed(
    classes: tuple[str, ...],
    intents: tuple[str, ...],
    rel_min: int,
    rel_max: int,
    date_from: str,
    date_to: str,
    limit: int = 200,
) -> tuple[list[dict], int]:
    where = ["status = 'CLASSIFIED'"]
    params: list[Any] = []
    if classes:
        where.append(f"confirmed_class IN ({','.join('?' * len(classes))})")
        params += list(classes)
    if intents:
        where.append(f"intent_signal IN ({','.join('?' * len(intents))})")
        params += list(intents)
    where.append("(relevance_score IS NULL OR (relevance_score >= ? AND relevance_score <= ?))")
    params += [rel_min, rel_max]
    where.append("date(created_at) >= ? AND date(created_at) <= ?")
    params += [date_from, date_to]

    clause = " AND ".join(where)
    total = _scalar(f"SELECT COUNT(*) FROM scraped_tweets WHERE {clause}", tuple(params))
    rows = _q(
        f"SELECT tweet_id, author_handle, author_followers, text, created_at, "
        f"confirmed_class, matched_class, matched_query, relevance_score, "
        f"quality_score, intent_signal, is_builder, theme_tags, competitor_mentioned, "
        f"summary_one_line, noise_reason, like_count, reply_count, retweet_count, "
        f"quote_count, impression_count, "
        f"(COALESCE(like_count,0) + COALESCE(reply_count,0)*3 + "
        f"COALESCE(retweet_count,0)*2 + COALESCE(quote_count,0)*2) AS engagement_score "
        f"FROM scraped_tweets WHERE {clause} "
        f"ORDER BY COALESCE(relevance_score, -1) DESC, "
        f"(COALESCE(like_count,0) + COALESCE(reply_count,0)*3 + "
        f"COALESCE(retweet_count,0)*2 + COALESCE(quote_count,0)*2) DESC "
        f"LIMIT ?",
        tuple(params) + (limit,),
    )
    return rows, int(total or 0)


# --------------------------------------------------------------------------
# Page 3 — Workflow (live config counts)
# --------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_lexicon_counts() -> dict:
    return {
        "keywords_enabled": int(_scalar("SELECT COUNT(*) FROM keywords WHERE enabled = 1")),
        "keywords_total": int(_scalar("SELECT COUNT(*) FROM keywords")),
        "classes_enabled": int(
            _scalar("SELECT COUNT(*) FROM keyword_classes WHERE enabled = 1")
        ),
        "influencers_enabled": int(
            _scalar("SELECT COUNT(*) FROM influencers WHERE enabled = 1")
        ),
    }


# --------------------------------------------------------------------------
# Page 5 — Keywords
# --------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_class_summary() -> list[dict]:
    """One row per class: enabled keyword count + tweet stats by matched_class."""
    classes = _q(
        "SELECT class_key, name, enabled, display_order, color_hex "
        "FROM keyword_classes ORDER BY display_order, class_key"
    )
    kw_counts = {
        r["class_key"]: r["n"]
        for r in _q(
            "SELECT class_key, COUNT(*) AS n FROM keywords WHERE enabled = 1 "
            "GROUP BY class_key"
        )
    }
    tweet_stats = {
        r["matched_class"]: r
        for r in _q(
            "SELECT matched_class, COUNT(*) AS total, AVG(relevance_score) AS avg_rel, "
            "SUM(CASE WHEN ingested_at >= datetime('now','-7 days') THEN 1 ELSE 0 END) "
            "AS scraped_7d FROM scraped_tweets WHERE matched_class IS NOT NULL "
            "GROUP BY matched_class"
        )
    }
    out = []
    for c in classes:
        key = c["class_key"]
        stat = tweet_stats.get(key, {})
        avg_rel = stat.get("avg_rel")
        out.append(
            {
                "class_key": key,
                "name": c.get("name"),
                "enabled": c.get("enabled"),
                "keywords": int(kw_counts.get(key, 0)),
                "tweets_total": int(stat.get("total") or 0),
                "avg_relevance": round(avg_rel) if avg_rel is not None else None,
                "scraped_7d": int(stat.get("scraped_7d") or 0),
            }
        )
    return out


@st.cache_data(ttl=300)
def get_keywords() -> list[dict]:
    return _q(
        "SELECT class_key, keyword, enabled, hit_count FROM keywords "
        "ORDER BY hit_count DESC, keyword ASC"
    )


# --------------------------------------------------------------------------
# Page 6 — Costs & Health
# --------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_daily_spend() -> list[dict]:
    return _q(
        "SELECT date(called_at) AS day, purpose, SUM(estimated_cost_usd) AS cost, "
        "COUNT(*) AS calls FROM llm_costs "
        "WHERE called_at >= datetime('now','-30 days') "
        "GROUP BY day, purpose ORDER BY day"
    )


@st.cache_data(ttl=60)
def get_spend_totals() -> dict:
    total = _scalar(
        "SELECT SUM(estimated_cost_usd) FROM llm_costs "
        "WHERE called_at >= datetime('now','-30 days')",
        default=0.0,
    )
    calls = _scalar(
        "SELECT COUNT(*) FROM llm_costs WHERE called_at >= datetime('now','-30 days')"
    )
    by_purpose = _q(
        "SELECT purpose, SUM(estimated_cost_usd) AS cost FROM llm_costs "
        "WHERE called_at >= datetime('now','-30 days') GROUP BY purpose"
    )
    return {
        "total": float(total or 0.0),
        "calls": int(calls or 0),
        "avg_per_call": float(total or 0.0) / calls if calls else 0.0,
        "by_purpose": by_purpose,
    }


@st.cache_data(ttl=30)
def get_api_log(limit: int = 50) -> list[dict]:
    # NB: Turso api_log has columns id, agent_id, called_at, service, endpoint,
    # status_code, month_key — no tweets_returned / rate_remaining.
    return _q(
        "SELECT called_at, service, endpoint, status_code FROM api_log "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )


@st.cache_data(ttl=30)
def get_status_breakdown() -> list[dict]:
    return _q(
        "SELECT status_code, COUNT(*) AS n FROM api_log "
        "WHERE called_at >= datetime('now','-7 days') "
        "GROUP BY status_code ORDER BY n DESC"
    )


@st.cache_data(ttl=30)
def get_failed_runs() -> list[dict]:
    rows = _q(
        "SELECT id, started_at, triggered_by, error_message FROM agent_runs "
        "WHERE status = 'failed' AND started_at >= datetime('now','-30 days') "
        "ORDER BY id DESC"
    )
    for r in rows:
        tb = r.get("triggered_by") or ""
        r["mode"] = tb.split(":", 1)[1] if ":" in tb else tb
    return rows
