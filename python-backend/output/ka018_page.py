"""
KA018 — LinkedIn Voice AI Signals (read-only Streamlit page).

Rendered as one page inside output/dashboard.py. Self-contained: it must NOT
import dashboard.py (that would be circular). Data access reuses
linkedin/db.py's LinkedInDatabase.query() — no new DB/connection logic.

Read-only: no API calls, no agent triggers. (LinkedInDatabase.__init__ does run
idempotent CREATE TABLE IF NOT EXISTS against its OWN replica, ka018_replica.db,
which is separate from KA017's replica.)

Schema notes — linkedin_posts uses `ingested_at` (not `scraped_at`) and
`repost_count` (not `share_count`); this page maps the requested labels onto
those real columns.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from agents.brand_visibility.linkedin.db import LinkedInDatabase

MONTHLY_BUDGET = int(os.getenv("LINKEDIN_MONTHLY_BUDGET", "50"))


# --------------------------------------------------------------------------
# Data access (reuse LinkedInDatabase.query); cached so we don't re-sync.
# --------------------------------------------------------------------------

@st.cache_resource
def _db() -> LinkedInDatabase:
    # Read-mostly, short-lived per render: background sync (60s) is fine here.
    return LinkedInDatabase(sync_interval=60)


@st.cache_data(ttl=30)
def _rows(sql: str, params: tuple = ()) -> list[dict]:
    return _db().query(sql, params)


def _scalar(sql: str, params: tuple = (), default=0):
    rows = _rows(sql, params)
    if not rows:
        return default
    val = next(iter(rows[0].values()))
    return val if val is not None else default


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _relative(ts) -> str:
    d = _parse_ts(ts)
    if not d:
        return "—"
    now = datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    secs = (now - d).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 60:
        return f"{secs:.0f}s ago"
    if secs < 3600:
        return f"{secs / 60:.0f}m ago"
    if secs < 86400:
        return f"{secs / 3600:.0f}h ago"
    return f"{secs / 86400:.0f}d ago"


def _fmt_ts(ts) -> str:
    d = _parse_ts(ts)
    return d.strftime("%b %d, %H:%M") if d else "—"


def _trunc(text, n: int) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def render() -> None:
    # 1. Header
    st.title("KA018 — LinkedIn Voice AI Signals")
    st.caption("Voice AI builder signals scraped from LinkedIn")

    # 2. KPI strip — row 1 = totals, row 2 = tier counts
    total_posts = _scalar("SELECT COUNT(*) FROM linkedin_posts")
    unique_authors = _scalar("SELECT COUNT(DISTINCT author_name) FROM linkedin_posts")
    last_scrape = _scalar("SELECT MAX(ingested_at) FROM linkedin_posts", default=None)
    used_this_month = _scalar(
        "SELECT COALESCE(SUM(api_calls_made), 0) FROM linkedin_runs "
        "WHERE strftime('%Y-%m', started_at) = strftime('%Y-%m', 'now')"
    )
    remaining = max(0, MONTHLY_BUDGET - int(used_this_month or 0))

    tier_counts = {r["classification_class"]: r["c"] for r in _rows(
        "SELECT classification_class, COUNT(*) AS c FROM linkedin_posts "
        "WHERE classification_class IS NOT NULL GROUP BY classification_class"
    )}
    t1 = tier_counts.get("TIER_1_ENGAGE", 0)
    classified_total = sum(tier_counts.values())

    r1 = st.columns(4)
    r1[0].metric("Total posts scraped", f"{total_posts:,}")
    r1[1].metric("Unique authors", f"{unique_authors:,}")
    r1[2].metric("Most recent scrape", _relative(last_scrape))
    r1[3].metric("API budget remaining", f"{remaining} / {MONTHLY_BUDGET}")

    r2 = st.columns(4)
    pct = f"{round(100 * t1 / classified_total)}% of classified" if classified_total else "—"
    r2[0].metric("🟢 TIER 1 · Engage", t1, delta=pct, delta_color="normal")  # highlighted
    r2[1].metric("🟡 TIER 2 · Watch", tier_counts.get("TIER_2_WATCH", 0))
    r2[2].metric("🔵 TIER 3 · Signal", tier_counts.get("TIER_3_SIGNAL", 0))
    r2[3].metric("⚪ TIER 4 · Noise", tier_counts.get("TIER_4_NOISE", 0))

    # 3. Filters (sidebar)
    TIER_OPTS = ["TIER_1_ENGAGE", "TIER_2_WATCH", "TIER_3_SIGNAL", "TIER_4_NOISE", "(unclassified)"]
    with st.sidebar:
        st.markdown("### KA018 filters")
        sel_tiers = st.multiselect(
            "Tier", TIER_OPTS, default=["TIER_1_ENGAGE", "TIER_2_WATCH"],  # actionable
        )

        cat_opts = [r["category"] for r in _rows(
            "SELECT DISTINCT category FROM linkedin_keywords "
            "WHERE category IS NOT NULL ORDER BY category"
        )]
        sel_cats = st.multiselect("Matched category", cat_opts, default=[])

        sc_opts = [r["source_class"] for r in _rows(
            "SELECT DISTINCT source_class FROM linkedin_posts "
            "WHERE source_class IS NOT NULL ORDER BY source_class"
        )]
        sel_sc = st.multiselect("Source class", sc_opts, default=[])

        sort_label = st.selectbox(
            "Sort by",
            ["Tier (T1 first)", "Scraped (newest)", "Most liked", "Posted (newest)"],
        )

    sort_sql = {
        "Tier (T1 first)": "classification_class ASC, commercial_fit_score DESC",
        "Scraped (newest)": "ingested_at DESC",
        "Most liked": "like_count DESC",
        "Posted (newest)": "posted_at DESC",
    }[sort_label]

    # 4. Posts table — limit 100 (applying filters + chosen sort)
    where: list[str] = []
    params: list = []

    real_tiers = [t for t in sel_tiers if t != "(unclassified)"]
    tier_clauses: list[str] = []
    if real_tiers:
        tier_clauses.append(f"classification_class IN ({', '.join('?' for _ in real_tiers)})")
    if "(unclassified)" in sel_tiers:
        tier_clauses.append("classification_class IS NULL")
    if tier_clauses:
        where.append("(" + " OR ".join(tier_clauses) + ")")
        params.extend(real_tiers)  # IS NULL contributes no param

    if sel_cats:
        where.append(f"matched_category IN ({', '.join('?' for _ in sel_cats)})")
        params.extend(sel_cats)
    if sel_sc:
        where.append(f"source_class IN ({', '.join('?' for _ in sel_sc)})")
        params.extend(sel_sc)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    posts = _rows(
        "SELECT classification_class, commercial_fit_score, relationship_value_score, "
        "intent_signal, summary_one_line, matched_keyword, author_name, author_headline, "
        "text, like_count, comment_count, repost_count, posted_at, post_url, ingested_at "
        f"FROM linkedin_posts{where_sql} ORDER BY {sort_sql} LIMIT 100",
        tuple(params),
    )

    st.subheader(f"Posts ({len(posts)} shown, 100 max)")
    if not posts:
        st.info("No posts match the current filters.")
    else:
        df = pd.DataFrame([{
            "tier": p["classification_class"] or "(unclassified)",
            "commercial_fit": p["commercial_fit_score"],
            "relationship_value": p["relationship_value_score"],
            "keyword": p["matched_keyword"],
            "author": p["author_name"],
            "headline": _trunc(p["author_headline"], 80),
            "reason": _trunc(p["summary_one_line"], 80),
            "noise_flags": p["intent_signal"] or "",   # comma-joined flags
            "likes": p["like_count"],
            "comments": p["comment_count"],
            "reposts": p["repost_count"],   # repost_count == requested "share_count"
            "text": _trunc(p["text"], 200),
            "posted_at": _fmt_ts(p["posted_at"]),
            "post_url": p["post_url"],
        } for p in posts])

        # Clickable link column when this Streamlit build supports it.
        col_config = None
        try:
            col_config = {"post_url": st.column_config.LinkColumn("Post", display_text="Open ↗")}
        except Exception:
            pass

        # Color-code the tier column (Styler). Falls back to plain df if the
        # installed Streamlit/pandas can't render a Styler with column_config.
        _tier_bg = {
            "TIER_1_ENGAGE": "background-color:#1b5e20;color:#fff",
            "TIER_2_WATCH": "background-color:#8d6e00;color:#fff",
            "TIER_3_SIGNAL": "background-color:#0d47a1;color:#fff",
            "TIER_4_NOISE": "background-color:#3a3a3a;color:#fff",
        }
        rendered = False
        try:
            styler = df.style.map(lambda v: _tier_bg.get(v, ""), subset=["tier"])
            st.dataframe(styler, use_container_width=True, hide_index=True, column_config=col_config)
            rendered = True
        except Exception:
            rendered = False
        if not rendered:
            st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_config)

    # 5. Footer (subtle, single line)
    runs_total = _scalar("SELECT COUNT(*) FROM linkedin_runs")
    last_run = _scalar("SELECT MAX(started_at) FROM linkedin_runs", default=None)
    ingested_month = _scalar(
        "SELECT COUNT(*) FROM linkedin_posts "
        "WHERE strftime('%Y-%m', ingested_at) = strftime('%Y-%m', 'now')"
    )
    st.caption(
        f"KA018 sweep runs total: {runs_total} | Last run: {_fmt_ts(last_run)} | "
        f"Posts ingested this month: {ingested_month}"
    )
