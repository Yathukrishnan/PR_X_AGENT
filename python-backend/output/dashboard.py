"""
KA017 — Demo Dashboard (read-only).

A 6-page Streamlit viewer over the agent's Turso database. Safe to run live in
front of leadership: it makes NO external API calls, NEVER triggers the agent,
and NEVER writes to Turso. Action buttons only display the CLI command to run.

    streamlit run output/dashboard.py   ->   http://localhost:8501
"""
from __future__ import annotations

import html
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# streamlit run puts the script's dir (output/) on sys.path, not the repo root.
# Ensure the repo root is importable so config/ and ingestion/ resolve.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from shared.config.settings import (
    CLASSIFIER_MODEL,
    DRAFTER_MODEL,
    MAX_API_CALLS_PER_RUN,
    SCRAPE_SLEEP_SECONDS,
    TURSO_SYNC_INTERVAL,
)

import output.dashboard_queries as q
import output.ka018_page as ka018
import output.prompt_manager as pm
import output.scheduler_manager as sm
from output import dashboard_styles as sty

st.set_page_config(
    page_title="KA017 — Market Intelligence Agent",
    page_icon="🪁",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(sty.get_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _esc(text) -> str:
    return html.escape(str(text or ""))


def _trunc(text, n: int = 140) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _duration(started: str | None, ended: str | None) -> str:
    a, b = _parse_ts(started), _parse_ts(ended)
    if not a or not b:
        return "—"
    secs = (b - a).total_seconds()
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs / 60:.1f}m"


def _fmt_ts(ts: str | None) -> str:
    d = _parse_ts(ts)
    return d.strftime("%b %d, %H:%M") if d else "—"


def _json_tags(raw) -> list[str]:
    import json
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [t.strip() for t in str(raw).strip("[]").split(",") if t.strip()]


def _tweet_url(handle, tweet_id) -> str | None:
    """Status URL for a tweet, or None if we can't build a valid one."""
    if not handle or not tweet_id:
        return None
    return f"https://x.com/{handle}/status/{tweet_id}"


def _engagement_strip(row) -> str:
    """Compact one-line engagement strip (overview cards)."""
    fc = q.format_count
    bits = [
        f"♥ {fc(row.get('like_count'))}",
        f"💬 {fc(row.get('reply_count'))}",
        f"🔁 {fc(row.get('retweet_count'))}",
    ]
    if row.get("impression_count") is not None:
        bits.append(f"👁 {fc(row.get('impression_count'))}")
    return f"<div class='engagement-strip'>{' · '.join(bits)}</div>"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='ka-logo'>Kite<span>AI</span></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ka-tag'>KA017 — Market Intelligence Agent</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Signal Feed",
                "Workflow",
                "Run Agent",
                "Classifier Prompt",
                "Scheduler",
                "Keywords",
                "Costs & Health",
                "KA018 LinkedIn",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        ok, detail = q.connection_ok()
        if ok:
            st.markdown(
                f"<div class='ka-conn-ok'>● Turso connected<br>{_esc(detail)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='ka-conn-bad'>● Turso unreachable<br>{_esc(detail)}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='ka-tag' style='margin-top:1rem'>Agent control plane · Run Agent and Classifier Prompt pages make live API calls</div>",
            unsafe_allow_html=True,
        )
    return page


# ---------------------------------------------------------------------------
# Class code -> human-readable label mapping. The DB stores single-letter
# codes ("A", "B", ..., "NOISE"); the UI presents friendly labels. Filter
# dropdowns show labels; queries use codes — convert at the UI boundary via
# `class_label` and `code_from_label`.
# ---------------------------------------------------------------------------
CLASS_LABELS: dict[str, str] = {
    "A": "A — Frontier Models",
    "B": "B — Agent Frameworks",
    "C": "C — Voice AI Stack",
    "D": "D — Unit Economics",
    "E": "E — Multilingual AI",
    "F": "F — Vertical Builders",
    "G": "G — AI Terminology",
    "K": "K — Buying Intent",
    "NOISE": "NOISE",
}


def class_label(code: str) -> str:
    """Return the friendly label for a class code, or the code itself if unknown."""
    if not code:
        return ""
    return CLASS_LABELS.get(code, code)


def code_from_label(label: str) -> str:
    """Reverse-lookup: given a friendly label, return the underlying class code.

    Returns the input unchanged if the label is already a code or unknown.
    """
    if not label:
        return ""
    for code, lbl in CLASS_LABELS.items():
        if lbl == label:
            return code
    return label  # fall back to raw input (handles 'All' and bare codes)


# --------------------------------------------------------------------------
# Page 1 — Overview
# --------------------------------------------------------------------------

def page_overview() -> None:
    st.title("KA017 — Overview")
    st.markdown(
        "<div class='ka-subtitle'>What the agent has been doing this week.</div>",
        unsafe_allow_html=True,
    )

    m = q.get_metrics_overview()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tweets scraped (7d)", m["scraped_7d"], delta=m["scraped_delta"])
    c2.metric(
        "Tweets classified (7d)",
        m["classified_7d"],
        delta=f"-{m['backlog']} backlog" if m["backlog"] else "0 backlog",
        delta_color="inverse",
    )
    c3.metric(
        "Avg relevance (7d)",
        f"{m['avg_relevance']}/100" if m["avg_relevance"] is not None else "—",
    )
    c4.metric(
        "API budget remaining",
        f"{m['budget_remaining']}/{m['budget_cap']}",
        delta=f"{m['last_calls_used']} used last run",
        delta_color="off",
    )

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Recent runs")
        runs = q.get_recent_runs(5)
        if runs:
            df = pd.DataFrame(
                [
                    {
                        "Run": r["id"],
                        "When": _fmt_ts(r["started_at"]),
                        "Mode": r.get("mode") or "—",
                        "Status": (r.get("status") or "—").capitalize(),
                        "Duration": _duration(r["started_at"], r["ended_at"]),
                        "New": r.get("records_new") or 0,
                        "Calls": r.get("calls_used") or 0,
                    }
                    for r in runs
                ]
            )
            st.dataframe(
                df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Run": st.column_config.NumberColumn(width="small"),
                    "New": st.column_config.NumberColumn(width="small"),
                    "Calls": st.column_config.NumberColumn(width="small"),
                },
            )
        else:
            st.info("No runs recorded yet. Trigger one from the Run Agent page.")

    with right:
        st.subheader("Top signals this week")
        signals = q.get_top_signals(5)
        if not signals:
            st.info("No classified tweets yet.")
        for s in signals:
            handle = s.get("author_handle") or "unknown"
            url = _tweet_url(s.get("author_handle"), s.get("tweet_id"))
            if url:
                head = (
                    f"<a class='ka-handle-link' href='{url}' target='_blank'>"
                    f"@{_esc(handle)} →</a>"
                )
            else:
                head = f"<span class='h'>@{_esc(handle)}</span>"
            with st.container(border=True):
                st.markdown(
                    f"<div class='ka-signal'>{head} "
                    f"{sty.relevance_badge(s.get('relevance_score'))} "
                    f"{sty.class_chip(s.get('confirmed_class'))}<br>"
                    f"{_esc(_trunc(s.get('text'), 140))}</div>"
                    f"{_engagement_strip(s)}",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Page 2 — Signal Feed
# --------------------------------------------------------------------------

def page_signal_feed() -> None:
    st.title("Signal Feed")
    st.markdown(
        "<div class='ka-subtitle'>Filterable view of every classified tweet.</div>",
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        classes = st.multiselect("Class", q.get_distinct("confirmed_class"))
    with f2:
        intents = st.multiselect("Intent", q.get_distinct("intent_signal"))
    with f3:
        rel = st.slider("Relevance", 0, 100, (40, 100))
    with f4:
        today = date.today()
        drange = st.date_input("Created", (today - timedelta(days=7), today))

    if isinstance(drange, (list, tuple)) and len(drange) == 2:
        date_from, date_to = drange[0].isoformat(), drange[1].isoformat()
    else:
        date_from = (today - timedelta(days=7)).isoformat()
        date_to = today.isoformat()

    rows, total = q.get_signal_feed(
        tuple(classes), tuple(intents), rel[0], rel[1], date_from, date_to
    )
    st.caption(f"Showing {len(rows)} of {total} tweets matching filters.")

    if not rows:
        st.info("No tweets match these filters. Widen the relevance range or date window.")
        return

    df = pd.DataFrame(
        [
            {
                "Created": _fmt_ts(r["created_at"]),
                "Handle": f"@{r.get('author_handle')}" if r.get("author_handle") else "—",
                # confirmed_class = what the AI decided; matched_class = which
                # keyword class surfaced it. They can legitimately differ.
                "AI Class": r.get("confirmed_class"),
                "Matched": r.get("matched_class"),
                "Matched Query": r.get("matched_query") or "",
                "Tweet": r.get("text") or "",
                "Score": r.get("relevance_score"),
                "Intent": r.get("intent_signal"),
                "Engagement": r.get("engagement_score") or 0,
                "Likes": r.get("like_count") or 0,
                "Replies": r.get("reply_count") or 0,
                "Retweets": r.get("retweet_count") or 0,
                "Quotes": r.get("quote_count") or 0,
                "Views": r.get("impression_count") or 0,
                "Open": _tweet_url(r.get("author_handle"), r.get("tweet_id")),
            }
            for r in rows
        ]
    )
    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Handle": st.column_config.TextColumn("Handle", width="small"),
            "AI Class": st.column_config.TextColumn(
                "AI Class", width="small",
                help="Class the AI classifier assigned after reading the tweet.",
            ),
            "Matched": st.column_config.TextColumn(
                "Matched", width="small",
                help="Keyword class whose query surfaced this tweet.",
            ),
            "Matched Query": st.column_config.TextColumn(
                "Matched Query",
                width="medium",
                help="The keyword query that surfaced this tweet (hover for full text).",
            ),
            "Tweet": st.column_config.TextColumn("Tweet", width="large"),
            "Score": st.column_config.NumberColumn("Score", width="small"),
            "Engagement": st.column_config.NumberColumn(
                "Engagement", width="small", format="%d",
                help="Weighted score: replies*3 + RTs*2 + quotes*2 + likes*1",
            ),
            "Likes": st.column_config.NumberColumn("Likes", width="small", format="%d"),
            "Replies": st.column_config.NumberColumn("Replies", width="small", format="%d"),
            "Retweets": st.column_config.NumberColumn("Retweets", width="small", format="%d"),
            "Quotes": st.column_config.NumberColumn("Quotes", width="small", format="%d"),
            "Views": st.column_config.NumberColumn("Views", width="small", format="%d"),
            # Dedicated link column -> opens the specific tweet (not the profile).
            "Open": st.column_config.LinkColumn(
                "Open", display_text="↗", width="small", help="Open the tweet on X",
            ),
        },
    )

    sel = event.selection.rows if event and event.selection else []
    if sel:
        _signal_detail(rows[sel[0]])


def _signal_detail(r: dict) -> None:
    """Expanded panel for a selected tweet — full text + engagement + match."""
    handle = r.get("author_handle") or "unknown"
    url = _tweet_url(r.get("author_handle"), r.get("tweet_id"))
    with st.container(border=True):
        head, link = st.columns([4, 1])
        with head:
            st.markdown(
                f"**@{_esc(handle)}**  ·  {_fmt_ts(r.get('created_at'))}"
                + (f"  ·  {r.get('author_followers'):,} followers"
                   if r.get("author_followers") else ""),
            )
        with link:
            if url:
                st.link_button("Open on X →", url)

        # Full tweet text — plain markdown, no HTML (avoids escaping/quote traps)
        st.markdown(r.get("text") or "")

        # Engagement as native metric widgets (no unsafe_allow_html needed)
        fc = q.format_count
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("♥ Likes", fc(r.get("like_count")))
        e2.metric("💬 Replies", fc(r.get("reply_count")))
        e3.metric("🔁 Retweets", fc(r.get("retweet_count")))
        e4.metric("❝ Quotes", fc(r.get("quote_count")))
        e5.metric(
            "👁 Views",
            fc(r.get("impression_count")) if r.get("impression_count") is not None else "—",
        )

        # Matched class + query (full query in hover title)
        mc = r.get("matched_class")
        mq = r.get("matched_query") or ""
        color = sty.class_color(mc)
        st.markdown(
            f"<div class='matched-keyword-line' title='{_esc(mq)}'>"
            f"<span class='class-badge' style='background:{color}22;color:{color};"
            f"border:1px solid {color}55'>{_esc(mc) or '—'}</span>"
            f"{_esc(_trunc(mq, 50))}</div>",
            unsafe_allow_html=True,
        )

        # Scores line
        st.markdown(
            f"Relevance: **{r.get('relevance_score') if r.get('relevance_score') is not None else '—'}/100**"
            f"  ·  Class: {sty.class_chip(r.get('confirmed_class'))}"
            f"  ·  Intent: **{_esc(r.get('intent_signal') or '—')}**"
            f"  ·  Quality: **{r.get('quality_score') if r.get('quality_score') is not None else '—'}/10**",
            unsafe_allow_html=True,
        )

        tags = _json_tags(r.get("theme_tags"))
        comps = _json_tags(r.get("competitor_mentioned"))
        if tags:
            st.markdown(
                "Theme tags: "
                + " ".join(sty.pill(_esc(t), sty.class_color(r.get("confirmed_class"))) for t in tags),
                unsafe_allow_html=True,
            )
        if comps:
            st.markdown(
                "Competitors: " + " ".join(sty.pill(_esc(c), "#EC4899") for c in comps),
                unsafe_allow_html=True,
            )
        if r.get("summary_one_line"):
            st.caption(f"AI summary: {r['summary_one_line']}")
        if (r.get("relevance_score") or 100) < 40 and r.get("noise_reason"):
            st.markdown(
                f"<span class='ka-muted ka-pill'>noise: {_esc(r['noise_reason'])}</span>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Page 3 — Workflow
# --------------------------------------------------------------------------

_FLOW = """\
TRIGGER (manual / orchestrator)
        |
        v
KEYWORD LOADING ---- Turso `keywords` (enabled=1) + `influencers` (enabled=1)
        |
        v
QUERY CHUNKING ----- ~60 OR'd queries, each <500 chars
        |
        v
SWEEP (twitter241) -- /search-v2 (Latest, count=20)
        |              -- /user (handle -> user_id cache)
        |              -- /user-tweets (timeline by user_id)
        |
        v
HARD STOP (12 calls)
        |
        v
AI CLASSIFY -------- Gemini Flash 2.5 via OpenRouter
        |             output: relevance_score (0-100) + class + noise_reason
        v
DB UPSERT ---------- Turso scraped_tweets (status='CLASSIFIED')
        |
        v
CLUSTER + DRAFT ---- Sonnet 4.5 (weekly, not every run)
        |
        v
HUMAN REVIEW ------- Dashboard / manual posting
"""


def _step(color: str, title: str, bullets: list[str], ref: str | None = None) -> None:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    ref_html = f"<div class='ka-ref'>↳ {ref}</div>" if ref else ""
    st.markdown(
        f"<div class='ka-step' style='border-left-color:{color}'>"
        f"<h4 style='color:{color}'>{title}</h4><ul>{items}</ul>{ref_html}</div>",
        unsafe_allow_html=True,
    )


def page_workflow() -> None:
    st.title("KA017 — System Workflow")
    st.markdown(
        "<div class='ka-subtitle'>Complete pipeline from keyword loading to draft "
        "generation, with current configuration values.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Data flow")
    st.code(_FLOW, language="text")

    lex = q.get_lexicon_counts()
    st.subheader("Pipeline stages")

    _step(
        "#3B82F6",
        "1 · KEYWORD LOADING",
        [
            "Source: Turso tables <code>keywords</code> and <code>influencers</code>.",
            f"Current state: <b>{lex['keywords_enabled']} keywords</b> across "
            f"<b>{lex['classes_enabled']} enabled classes</b>, "
            f"<b>{lex['influencers_enabled']} influencers</b>.",
            "Filter: only rows with <code>enabled = 1</code>.",
            "Fallback: <code>config/genesis_lexicon.json</code> if Turso is unreachable.",
        ],
        "ingestion/lexicon.py",
    )
    _step(
        "#06B6D4",
        "2 · QUERY CHUNKING",
        [
            "Groups keywords into OR'd batches of max 18 keywords per chunk.",
            "Adds post-hoc operator suffix: "
            "<code>min_faves:1 -is:retweet -is:reply -is:nullcast lang:en</code> "
            "(Class E omits <code>lang</code>).",
            "Each chunk stays under 500 chars (X's query limit).",
        ],
    )
    _step(
        "#10B981",
        "3 · TWITTER241 SCRAPE",
        [
            "Provider: <code>twitter241</code> (RapidAPI by davethebeast).",
            "<code>GET /search-v2?type=Latest&count=20&query=…</code>",
            "<code>GET /user?username=…</code> (handle → numeric id)",
            "<code>GET /user-tweets?user=&lt;id&gt;&count=20</code>",
            "<code>GET /user-replies-v2?user=&lt;id&gt;&count=20</code>",
            "User-ID cache: <code>user_id_cache</code> table avoids double-billing resolution.",
        ],
        "ingestion/providers/twitter241.py",
    )
    _step(
        "#F59E0B",
        "4 · RATE LIMITING & BUDGET",
        [
            f"Hard cap per run: <code>MAX_API_CALLS_PER_RUN = {MAX_API_CALLS_PER_RUN}</code>.",
            f"Sleep between calls: <code>SCRAPE_SLEEP_SECONDS = {SCRAPE_SLEEP_SECONDS}s</code>.",
            "On 429: exponential backoff + retry.",
            "On budget exhaustion: stop new scrape calls, finish in-memory classification.",
        ],
        "ingestion/x_scraper.py",
    )
    _step(
        "#8B5CF6",
        "5 · AI CLASSIFICATION",
        [
            f"Model: Gemini Flash 2.5 (<code>{CLASSIFIER_MODEL}</code>) via OpenRouter.",
            "Endpoint: <code>https://openrouter.ai/api/v1/chat/completions</code>.",
            "Cost per call: ~$0.001 (Gemini Flash 2.5 pricing).",
            "Output: relevance_score (0-100), class (A-K/NOISE), noise_reason when &lt;40.",
        ],
        "processing/classifier.py",
    )
    st.code(
        """{
  "relevance_score": 0-100,
  "confirmed_class": "A-K or NOISE",
  "intent_signal": "BUILDER_PAIN | BUILDER_QUESTION | RECOMMENDATION | OBSERVATION | MARKETING",
  "is_builder": 0 | 1,
  "quality_score": 0-10,
  "theme_tags": ["voice-latency", ...],
  "competitor_mentioned": ["Vapi", ...],
  "summary_one_line": "Brief one-liner",
  "noise_reason": "Why low-scoring (if score < 40)"
}""",
        language="json",
    )
    _step(
        "#6B7280",
        "6 · STORAGE",
        [
            "Database: Turso (libSQL embedded replica at <code>data/ka017_replica.db</code>).",
            "Tables written: <code>scraped_tweets, agent_runs, agent_activity, "
            "llm_costs, api_log, user_id_cache</code>.",
            f"Sync interval: <code>TURSO_SYNC_INTERVAL = {TURSO_SYNC_INTERVAL}s</code>.",
            "Shared with: the Next.js team dashboard (same Turso DB).",
        ],
        "ingestion/db.py",
    )
    _step(
        "#EF4444",
        "7 · CLUSTER + DRAFT",
        [
            "Runs only ~every 24h (not per scrape).",
            "Clusters classified tweets by overlapping <code>theme_tags</code>.",
            f"Drafts X posts using Sonnet 4.5 (<code>{DRAFTER_MODEL}</code>) from clusters of 5+ tweets.",
            "Output table: <code>content_themes</code> (status='DRAFT').",
            "<b>Drafts are never auto-posted</b> — human review required.",
        ],
        "output/post_drafter.py",
    )

    st.markdown(
        "<div class='ka-warn'><h4>What KA017 does NOT do</h4><ul>"
        "<li>Does not post to X</li>"
        "<li>Does not reply to other users' tweets</li>"
        "<li>Does not impersonate anyone</li>"
        "<li>Does not modify keyword / rule / influencer config (humans only, via the Next.js dashboard)</li>"
        "<li>Does not run autonomously without human review of draft content</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Page 4 — Run Agent
# --------------------------------------------------------------------------

# Project root for spawning the orchestrator subprocess.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _pending_count() -> int:
    """Short-lived count read. Caller must ensure no subprocess is writing."""
    from agents.brand_visibility.x.db import Database

    db = Database()
    try:
        rows = db.query("SELECT COUNT(*) AS c FROM scraped_tweets WHERE status = 'PENDING'")
        return rows[0]["c"] if rows else 0
    finally:
        # Drop the replica handle so the dashboard holds nothing during a run.
        del db


def _state_counts() -> tuple[int, int]:
    """Read (classified, pending) in ONE short-lived connection to limit churn."""
    from agents.brand_visibility.x.db import Database

    db = Database()
    try:
        rows = db.query("SELECT status, COUNT(*) AS c FROM scraped_tweets GROUP BY status")
        m = {r["status"]: r["c"] for r in rows}
        return m.get("CLASSIFIED", 0), m.get("PENDING", 0)
    finally:
        del db


def _drop_dashboard_db_handle() -> None:
    """Evict the cached Turso connection so the dashboard process holds NO replica
    handle while a subprocess writes. This is the core WAL-corruption guard: the
    embedded replica is fragile under concurrent access (audit P0-2)."""
    try:
        q.get_db.clear()  # st.cache_resource.clear()
    except Exception:
        pass


def run_orchestrator_command(cmd_args, log_state_key, max_seconds=1200):
    """Run an orchestrator subprocess, streaming stdout into session_state.

    Returns (exit_code, lines_received). Uses sys.executable so the subprocess
    runs under the SAME venv as Streamlit (the global python lacks libsql).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_placeholder = st.empty()
    try:
        proc = subprocess.Popen(
            cmd_args,
            cwd=_PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as e:
        st.session_state[log_state_key].append(f"ERROR: failed to start subprocess: {e}")
        log_placeholder.code("\n".join(st.session_state[log_state_key][-50:]))
        return -1, 0

    start_time = time.time()
    lines_received = 0
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        st.session_state[log_state_key].append(line)
        lines_received += 1
        log_placeholder.code("\n".join(st.session_state[log_state_key][-50:]))
        if time.time() - start_time > max_seconds:
            proc.terminate()
            st.session_state[log_state_key].append(
                f"TIMEOUT: exceeded {max_seconds}s, process terminated"
            )
            log_placeholder.code("\n".join(st.session_state[log_state_key][-50:]))
            break

    exit_code = proc.wait()
    elapsed = int(time.time() - start_time)
    st.session_state[log_state_key].append(
        f"--- Done in {elapsed}s with exit code {exit_code} ---"
    )
    log_placeholder.code("\n".join(st.session_state[log_state_key][-50:]))
    return exit_code, lines_received


def show_current_state() -> None:
    with st.container(border=True):
        st.markdown("**Current state**")
        # Skip the live DB read while a run is active — querying the replica
        # during a subprocess write is the exact concurrency hazard we avoid.
        if st.session_state.get("collect_running") or st.session_state.get("classify_running"):
            st.info("Run in progress — refresh state after it completes.")
            return
        try:
            classified, pending = _state_counts()
            last_run = q.get_recent_runs(1)
        except Exception as e:
            st.error(f"Could not read state: {e}")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Classified", classified)
        c2.metric("Pending", pending)
        lr = last_run[0] if last_run else None
        c3.metric(
            "Last run",
            f"{lr.get('mode') or '—'}" if lr else "Never",
            delta=_fmt_ts(lr["started_at"]) if lr else None,
            delta_color="off",
        )
        if st.button("🔄 Refresh state", key="refresh_state_btn"):
            st.rerun()


def show_collection_panel() -> None:
    with st.container(border=True):
        st.subheader("Run Collection")
        st.markdown(
            "Fetch new tweets from keyword search **and** influencer timelines. "
            "Cost: ~129 RapidAPI calls (87 keyword chunks + 42 influencer handles). "
            "Time: ~15-20 min depending on rate limits."
        )

        # Sweep configuration (applies to keyword sweep only)
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            sweep_type = st.radio(
                "Sweep type",
                ["Latest", "Top"],
                horizontal=True,
                help="Latest = real-time chronological. Top = X algorithmic ranking.",
                key="sweep_type_radio",
            )
        with ctrl_col2:
            count_label = st.selectbox(
                "Tweets per query",
                ["20 (1 page)", "40 (2 pages)", "60 (3 pages)"],
                index=0,
                help="twitter241 caps at 20 tweets per page. More pages = deeper capture at proportional API cost.",
                key="sweep_count_select",
            )
            max_pages = {"20 (1 page)": 1, "40 (2 pages)": 2, "60 (3 pages)": 3}[count_label]
        with ctrl_col3:
            class_choice = st.selectbox(
                "Class filter",
                ["All"] + [CLASS_LABELS[c] for c in ["A", "B", "C", "D", "E", "F", "G", "K"]],
                index=0,
                help="Limit sweep to one keyword class. 'All' = unfiltered (default).",
                key="sweep_class_select",
            )
            class_arg = "" if class_choice == "All" else code_from_label(class_choice)

        busy = st.session_state.get("collect_running") or st.session_state.get("classify_running")
        if st.session_state.get("collect_running"):
            st.warning("⏳ Collection running. Wait for it to finish.")

        if st.button("▶ Run Collection", disabled=busy, key="run_collect_btn"):
            st.session_state.collect_running = True
            st.session_state.collect_logs = []
            _drop_dashboard_db_handle()  # dashboard holds no replica handle during the run
            try:
                st.session_state.collect_logs.append(
                    f"=== Starting keyword sweep (type={sweep_type}, max_pages={max_pages}, "
                    f"classes={class_arg or 'all'}) ==="
                )
                kw_cmd = [sys.executable, "-m", "agents.brand_visibility.x.orchestrator", "--once", "--mode", "keywords",
                          "--sweep-type", sweep_type, "--max-pages", str(max_pages)]
                if class_arg:
                    kw_cmd += ["--classes", class_arg]
                kw_exit, _ = run_orchestrator_command(
                    kw_cmd, "collect_logs", max_seconds=1500,
                )
                if kw_exit != 0:
                    st.session_state.collect_logs.append(
                        f"Keyword sweep exited with code {kw_exit}. Continuing to influencers."
                    )
                st.session_state.collect_logs.append("=== Starting influencer sweep ===")
                run_orchestrator_command(
                    [sys.executable, "-m", "agents.brand_visibility.x.orchestrator", "--once", "--mode", "influencers"],
                    "collect_logs", max_seconds=1500,
                )
                st.session_state.collect_logs.append("=== Collection complete ===")
            except Exception as e:
                st.session_state.collect_logs.append(f"FATAL ERROR: {e}")
            finally:
                st.session_state.collect_running = False
                st.rerun()

        if st.session_state.get("collect_logs"):
            with st.expander("Collection logs", expanded=True):
                st.code("\n".join(st.session_state.collect_logs[-100:]))


def show_classification_panel() -> None:
    with st.container(border=True):
        st.subheader("Run Classification")
        st.markdown(
            "Classify all PENDING tweets via Gemini Flash 2.5. "
            "Cost: ~$0.0006 per tweet. Time: ~7 min per 50-tweet batch. "
            "Loops until PENDING reaches 0 (max 15 batches per click)."
        )

        running = st.session_state.get("classify_running")
        busy = running or st.session_state.get("collect_running")

        # Only read the count when nothing is writing.
        pending_count = None
        if not busy:
            try:
                pending_count = _pending_count()
            except Exception as e:
                st.error(f"Could not read pending count: {e}")
                pending_count = None

        if pending_count is not None:
            st.info(f"📊 {pending_count} tweets pending classification")
            if pending_count == 0:
                st.success("Nothing to classify. All tweets are already processed.")

        if running:
            st.warning("⏳ Classification running. Wait for it to finish.")

        can_run = (not busy) and bool(pending_count)
        if st.button("▶ Run Classification", disabled=not can_run, key="run_classify_btn"):
            st.session_state.classify_running = True
            st.session_state.classify_logs = []
            _drop_dashboard_db_handle()
            try:
                max_batches = max(1, (pending_count // 50) + 1)
                max_batches = min(max_batches, 15)
                st.session_state.classify_logs.append(
                    f"=== Starting classification: {pending_count} pending, "
                    f"up to {max_batches} batches ==="
                )
                for batch_num in range(1, max_batches + 1):
                    st.session_state.classify_logs.append(
                        f"=== Batch {batch_num} of {max_batches} ==="
                    )
                    exit_code, _ = run_orchestrator_command(
                        [sys.executable, "-m", "agents.brand_visibility.x.orchestrator", "--once", "--mode", "classify"],
                        "classify_logs", max_seconds=900,
                    )
                    if exit_code != 0:
                        st.session_state.classify_logs.append(
                            f"Batch {batch_num} exited with code {exit_code}. Continuing."
                        )
                    # Safe to read now — the per-batch subprocess has fully exited.
                    try:
                        remaining = _pending_count()
                    except Exception as e:
                        st.session_state.classify_logs.append(
                            f"(could not read remaining count: {e})"
                        )
                        remaining = -1
                    st.session_state.classify_logs.append(
                        f"After batch {batch_num}: {remaining} pending remaining"
                    )
                    if remaining == 0:
                        st.session_state.classify_logs.append("=== All tweets classified ===")
                        break
                st.session_state.classify_logs.append("=== Classification complete ===")
            except Exception as e:
                st.session_state.classify_logs.append(f"FATAL ERROR: {e}")
            finally:
                st.session_state.classify_running = False
                st.rerun()

        if st.session_state.get("classify_logs"):
            with st.expander("Classification logs", expanded=True):
                st.code("\n".join(st.session_state.classify_logs[-100:]))


def page_run_agent() -> None:
    # Session state defaults (must exist before any handler reads them).
    for key, default in (
        ("collect_running", False),
        ("classify_running", False),
        ("collect_logs", []),
        ("classify_logs", []),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    st.title("Run Agent")
    st.markdown(
        "<div class='ka-subtitle'>Manually trigger collection and classification "
        "runs. Each spawns the orchestrator as a subprocess — the dashboard "
        "releases its database handle while a run is active.</div>",
        unsafe_allow_html=True,
    )

    show_current_state()
    st.divider()
    show_collection_panel()
    st.divider()
    show_classification_panel()


# --------------------------------------------------------------------------
# Page 5 — Keywords
# --------------------------------------------------------------------------

def page_keywords() -> None:
    summary = q.get_class_summary()
    total_kw = sum(c["keywords"] for c in summary)
    st.title("Keywords")
    st.markdown(
        f"<div class='ka-subtitle'>{total_kw} keywords across {len(summary)} classes. "
        f"Editing happens in the Next.js dashboard.</div>",
        unsafe_allow_html=True,
    )

    rows3 = [summary[i : i + 3] for i in range(0, len(summary), 3)]
    for group in rows3:
        cols = st.columns(3)
        for col, c in zip(cols, group):
            color = sty.class_color(c["class_key"])
            with col:
                with st.container(border=True):
                    # NB: do NOT inline sty.MONO here — its single quotes collide
                    # with the single-quoted style attribute and break rendering.
                    st.markdown(
                        f"<span style='color:{color};font-weight:700'>"
                        f"{_esc(c['class_key'])} — {_esc(c['name'])}</span>",
                        unsafe_allow_html=True,
                    )
                    a, b = st.columns(2)
                    a.metric("Keywords", c["keywords"])
                    b.metric(
                        "Avg rel.",
                        f"{c['avg_relevance']}" if c["avg_relevance"] is not None else "—",
                    )
                    st.caption(f"{c['scraped_7d']} scraped (7d) · {c['tweets_total']} total")

    st.divider()
    kws = q.get_keywords()
    all_classes = sorted({k["class_key"] for k in kws if k.get("class_key")})

    g1, g2, g3 = st.columns([2, 1, 2])
    with g1:
        fclasses = st.multiselect("Class", all_classes)
    with g2:
        only_matched = st.toggle("Only with hits", value=False)
    with g3:
        search = st.text_input("Search keyword", "")

    filtered = [
        k
        for k in kws
        if (not fclasses or k.get("class_key") in fclasses)
        and (not only_matched or (k.get("hit_count") or 0) > 0)
        and (not search or search.lower() in (k.get("keyword") or "").lower())
    ]
    st.caption(f"Showing {len(filtered)} of {len(kws)} keywords.")

    df = pd.DataFrame(
        [
            {
                "Class": k.get("class_key"),
                "Keyword": k.get("keyword"),
                "Enabled": "✓" if k.get("enabled") else "✗",
                "Hits": k.get("hit_count") or 0,
            }
            for k in filtered
        ]
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "Class": st.column_config.TextColumn(width="small"),
            "Enabled": st.column_config.TextColumn(width="small"),
            "Hits": st.column_config.NumberColumn(width="small"),
        },
    )


# --------------------------------------------------------------------------
# Page 6 — Costs & Health
# --------------------------------------------------------------------------

def page_costs_health() -> None:
    st.title("Costs & Health")
    st.markdown(
        "<div class='ka-subtitle'>Operational visibility: spend, API status, failures.</div>",
        unsafe_allow_html=True,
    )

    # --- Section 1: spend ---
    st.subheader("Spend (30d)")
    totals = q.get_spend_totals()
    daily = q.get_daily_spend()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total (30d)", f"${totals['total']:.4f}")
    c2.metric("LLM calls", totals["calls"])
    c3.metric("Avg / call", f"${totals['avg_per_call']:.5f}")

    if daily:
        ddf = pd.DataFrame(daily)
        pivot = ddf.pivot_table(
            index="day", columns="purpose", values="cost", aggfunc="sum"
        ).fillna(0)
        st.line_chart(pivot, height=240)
        if totals["by_purpose"]:
            bp = pd.DataFrame(totals["by_purpose"]).set_index("purpose")
            st.bar_chart(bp, height=200)
    else:
        st.info("No LLM cost rows in the last 30 days yet.")

    st.divider()

    # --- Section 2: API health ---
    st.subheader("API health")
    breakdown = q.get_status_breakdown()
    if breakdown:
        chips = " ".join(
            sty.pill(
                f"{r['status_code']}: {r['n']}",
                "#10B981" if str(r["status_code"]).startswith("2")
                else "#F59E0B" if str(r["status_code"]).startswith("4")
                else "#EF4444",
            )
            for r in breakdown
        )
        st.markdown(f"Last 7d status codes: {chips}", unsafe_allow_html=True)

    log = q.get_api_log(50)
    if log:
        df = pd.DataFrame(
            [
                {
                    "When": _fmt_ts(r["called_at"]),
                    "Service": r.get("service"),
                    "Endpoint": r.get("endpoint"),
                    "Status": r.get("status_code"),
                }
                for r in log
            ]
        )
        st.dataframe(df, hide_index=True, width="stretch", height=320)
    else:
        st.info("No API calls logged yet.")

    st.divider()

    # --- Section 3: errors ---
    st.subheader("Errors & failures (30d)")
    failures = q.get_failed_runs()
    if not failures:
        with st.container(border=True):
            st.markdown(
                "<span style='color:#10B981;font-weight:700'>✓ No failures in the last 30 days</span>",
                unsafe_allow_html=True,
            )
    else:
        df = pd.DataFrame(
            [
                {
                    "Run": r["id"],
                    "Started": _fmt_ts(r["started_at"]),
                    "Mode": r.get("mode") or "—",
                    "Error": _trunc(r.get("error_message"), 100),
                }
                for r in failures
            ]
        )
        st.dataframe(df, hide_index=True, width="stretch")


def page_classifier_prompt() -> None:
    """Editor + tester for the LLM classifier system prompt.

    Lets the user list, view, edit, save, create, and activate prompts stored
    in config/prompts/. Reads from disk on every render via prompt_manager so
    the dashboard reflects the actual filesystem state.
    """
    st.title("Classifier Prompt")
    st.markdown(
        "<div class='ka-subtitle'>Edit the system prompt the classifier "
        "sends to the LLM. The next classification batch picks up your "
        "changes automatically.</div>",
        unsafe_allow_html=True,
    )

    # Refresh state from disk each render
    try:
        available = pm.list_prompts()
        active_name = pm.get_active_name()
    except Exception as exc:
        st.error(f"Could not read prompts directory: {exc}")
        return

    if not available:
        st.warning(
            "No prompts found in config/prompts/. Create one below to begin."
        )
        # Allow creating from blank
        new_blank_name = st.text_input(
            "Name for new prompt",
            value="my-first-prompt",
            key="prompt_blank_new_name",
        )
        new_blank_content = st.text_area(
            "Prompt content", value="", height=300, key="prompt_blank_new_content"
        )
        if st.button("Create prompt", key="prompt_blank_create_btn"):
            try:
                pm.save_prompt(new_blank_name, new_blank_content)
                pm.set_active(new_blank_name)
                st.success(f"Created and activated: {new_blank_name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create prompt: {exc}")
        return

    # Active banner
    if active_name and active_name in available:
        st.success(f"Active prompt: **{active_name}**")
    elif active_name:
        st.warning(
            f"active.txt points to '{active_name}' but no such prompt file exists. "
            f"Choose a prompt below and click 'Set as Active'."
        )
    else:
        st.warning(
            "No active prompt set. Choose one below and click 'Set as Active'."
        )

    # Prompt selector
    default_idx = (
        available.index(active_name) if active_name in available else 0
    )
    selected = st.selectbox(
        "Select prompt",
        available,
        index=default_idx,
        key="prompt_selector",
    )

    # Load selected prompt's content
    try:
        original_content = pm.read_prompt(selected)
    except Exception as exc:
        st.error(f"Could not read prompt '{selected}': {exc}")
        return

    # Editor
    edited_content = st.text_area(
        "Prompt content (edits stay local until you save):",
        value=original_content,
        height=500,
        key=f"prompt_editor_{selected}",
    )

    # Stats row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Characters", f"{len(edited_content):,}")
    with stat_col2:
        st.metric("Tokens (est.)", f"{pm.estimate_tokens(edited_content):,}")
    with stat_col3:
        dirty = edited_content != original_content
        st.metric("Status", "Unsaved" if dirty else "Saved")
    with stat_col4:
        st.metric(
            "Active",
            "Yes" if selected == active_name else "No",
        )

    # Action buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("Save changes", key="prompt_save_btn", disabled=not dirty):
            try:
                pm.save_prompt(selected, edited_content)
                st.success(f"Saved {selected}")
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")
    with btn_col2:
        if selected != active_name:
            if st.button("Set as active", key="prompt_activate_btn"):
                try:
                    pm.set_active(selected)
                    st.success(f"{selected} is now the active prompt")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not set active: {exc}")
        else:
            st.button("Set as active", disabled=True, key="prompt_activate_btn")
    with btn_col3:
        if st.button("Discard edits", key="prompt_discard_btn", disabled=not dirty):
            st.rerun()
    with btn_col4:
        # Save-as-new is collapsed into an expander to avoid cluttering
        pass

    # Save as new (expander to keep UI clean)
    with st.expander("Save as new prompt"):
        new_name = st.text_input(
            "New prompt name (letters, digits, hyphens, underscores only)",
            value="",
            key="prompt_new_name",
        )
        if st.button("Save as new", key="prompt_save_as_new_btn"):
            try:
                if not new_name.strip():
                    st.error("Name cannot be empty")
                elif pm.prompt_exists(new_name.strip()):
                    st.error(f"A prompt named '{new_name}' already exists")
                else:
                    pm.save_prompt(new_name.strip(), edited_content)
                    st.success(
                        f"Created '{new_name}'. Select it from the dropdown to use."
                    )
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not save: {exc}")

    # Test the (potentially unsaved) prompt
    st.divider()
    st.subheader("Test prompt against a sample tweet")
    st.caption(
        "Runs ONE classification using the prompt currently in the editor "
        "(saved or not). Costs ~$0.001 per test."
    )
    sample_text = st.text_area(
        "Sample tweet text",
        value="Vapi latency is killing us at 3 seconds TTFT. Anyone moved off it to Retell?",
        height=80,
        key="prompt_test_sample",
    )
    sample_handle = st.text_input(
        "Sample author handle (optional)",
        value="testuser",
        key="prompt_test_handle",
    )
    if st.button("Test prompt", key="prompt_test_btn"):
        if not sample_text.strip():
            st.error("Sample tweet cannot be empty")
        elif not edited_content.strip():
            st.error("Prompt cannot be empty")
        else:
            with st.spinner("Calling classifier..."):
                try:
                    from agents.brand_visibility.x.classifier import classify_one
                    result = classify_one(
                        {
                            "tweet_id": "preview",
                            "author_handle": sample_handle or "testuser",
                            "author_followers": 100,
                            "text": sample_text,
                        },
                        system_prompt=edited_content,
                    )
                    if result is None:
                        st.error(
                            "Classifier returned no result. Check classifier "
                            "logs for parse failure or API error."
                        )
                    else:
                        st.success("Classification complete")
                        result_dict = (
                            result.model_dump()
                            if hasattr(result, "model_dump")
                            else dict(result)
                        )
                        st.json(result_dict)
                except Exception as exc:
                    st.error(f"Test failed: {exc}")


def page_scheduler() -> None:
    """Configure the standalone scheduler process and view its runtime status.

    The dashboard only writes config. To activate the schedule, the user
    runs `python scheduler.py` in a terminal — that separate process honors
    this config and fires orchestrator runs on the chosen schedule.
    """
    st.title("Scheduler")
    st.markdown(
        "<div class='ka-subtitle'>Configure automated runs. The schedule is "
        "honored by a separate scheduler process, not by the dashboard itself.</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------- runtime status ---
    status = sm.read_status()
    if status:
        # Scheduler is running
        state_label = status.get("state", "unknown")
        next_fire = status.get("next_fire_at")
        last_fire = status.get("last_fire_at")
        last_outcome = status.get("last_fire_outcome")

        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.metric("Scheduler", "Running", help=f"PID {status.get('pid')}")
        with status_col2:
            if last_fire:
                outcome_label = (
                    "Success" if last_outcome == "success"
                    else "Failure" if last_outcome == "failure"
                    else "—"
                )
                st.metric("Last run", _fmt_ts(last_fire), help=f"Outcome: {outcome_label}")
            else:
                st.metric("Last run", "Not yet")
        with status_col3:
            st.metric("Next run", _fmt_ts(next_fire) if next_fire else "—")
        st.caption(f"State: {state_label}")
    else:
        st.warning(
            "Scheduler is **not running**. After saving the config below, "
            "open a terminal and run:  \n"
            "`python scheduler.py`  \n"
            "to activate the schedule. The dashboard will detect the process "
            "and show its status here."
        )

    st.divider()

    # -------------------------------------------------- config form ---
    try:
        cfg = sm.load_schedule()
    except Exception as exc:
        st.error(f"Could not load schedule config: {exc}")
        return

    st.subheader("Schedule configuration")

    enabled = st.checkbox(
        "Enable schedule",
        value=cfg.get("enabled", False),
        help="When unchecked, the scheduler runs in idle state and does not fire runs.",
        key="sched_enabled",
    )

    freq_type = st.radio(
        "Frequency type",
        ["interval", "fixed_times"],
        index=0 if cfg.get("frequency_type", "interval") == "interval" else 1,
        horizontal=True,
        format_func=lambda x: "Every N minutes" if x == "interval" else "At specific times",
        key="sched_freq_type",
    )

    if freq_type == "interval":
        interval_minutes = st.number_input(
            "Interval (minutes)",
            min_value=5,
            max_value=1440,
            value=int(cfg.get("interval_minutes", 60)),
            step=5,
            help="Minimum 5 minutes. Lower values risk API quota exhaustion.",
            key="sched_interval",
        )
        fixed_times = cfg.get("fixed_times", ["09:00", "13:00", "17:00"])
    else:
        interval_minutes = int(cfg.get("interval_minutes", 60))
        default_times = ",".join(cfg.get("fixed_times", ["09:00", "13:00", "17:00"]))
        fixed_times_str = st.text_input(
            "Fixed times (HH:MM, comma-separated, UTC)",
            value=default_times,
            help="Example: 09:00,13:00,17:00 — scheduler fires at each time daily.",
            key="sched_fixed_times",
        )
        fixed_times = [t.strip() for t in fixed_times_str.split(",") if t.strip()]

    valid_modes = ["keywords", "classify", "influencers", "reply_trees"]
    default_modes = cfg.get("modes", ["keywords", "classify"])
    # Filter default_modes to only valid ones (forward-compat)
    default_modes = [m for m in default_modes if m in valid_modes]
    modes = st.multiselect(
        "Pipeline modes (each scheduled run will execute these in order)",
        valid_modes,
        default=default_modes,
        help="keywords = fetch new tweets; classify = LLM-classify pending tweets",
        key="sched_modes",
    )

    # Last modified timestamp display
    last_modified = cfg.get("last_modified")
    if last_modified:
        st.caption(f"Config last saved: {_fmt_ts(last_modified)}")

    # ----------------------------------------------------- save ---
    save_col, _ = st.columns([1, 3])
    with save_col:
        if st.button("Save schedule", key="sched_save_btn", type="primary"):
            new_config = {
                "enabled": enabled,
                "frequency_type": freq_type,
                "interval_minutes": int(interval_minutes),
                "fixed_times": fixed_times,
                "modes": modes,
            }
            try:
                sm.save_schedule(new_config)
                st.success(
                    "Schedule saved. "
                    + ("The running scheduler will pick it up at the next check."
                       if status else
                       "Run `python scheduler.py` to activate.")
                )
                st.rerun()
            except ValueError as exc:
                st.error(f"Invalid config: {exc}")
            except Exception as exc:
                st.error(f"Save failed: {exc}")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Signal Feed": page_signal_feed,
    "Workflow": page_workflow,
    "Run Agent": page_run_agent,
    "Classifier Prompt": page_classifier_prompt,
    "Scheduler": page_scheduler,
    "Keywords": page_keywords,
    "Costs & Health": page_costs_health,
    "KA018 LinkedIn": ka018.render,
}


def main() -> None:
    page = sidebar()
    try:
        PAGES[page]()
    except Exception as exc:  # demo safety — never hard-crash on stage
        st.error(f"Could not load **{page}**. Turso may be momentarily unreachable.")
        st.exception(exc)


if __name__ == "__main__":
    main()
