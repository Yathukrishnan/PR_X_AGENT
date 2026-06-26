# KA017 Agent — Project Context for Claude Code

## What this is

KA017 is a market-intelligence agent for KiteAI (voice-AI infrastructure, UAE-based, multilingual focus). It scrapes X (Twitter) for signals about voice-AI builders' pain points, classifies them with an LLM, and produces TWO outputs:

1. A signal dashboard (humans review for sales intelligence and competitive context)
2. A content pipeline (drafted posts from KiteAI's own X account, humans review and publish)

## What this is NOT

This system does NOT post to X. It does NOT draft replies to other users' tweets. It does NOT use sockpuppet accounts. All X-writes are manual, from the clearly-identified @KiteAI account, after human review. If a future requirement pushes toward automated replies, keyword-triggered engagement with strangers' threads, or impersonation, flag it as a design change requiring discussion — do not silently implement.

## Architecture

Two-loop system over shared infrastructure:

```
genesis_lexicon.json (config/)
        ↓
x_scraper.py ── reads → SQLite (data/ka017_memory.db)
   ├── keyword sweep (Mode 1)
   ├── influencer sweep (Mode 2)
   └── reply-tree expansion (Mode 3, conditional)
        ↓
classifier.py ── enriches → SQLite (per-tweet class + intent + score)
        ↓
        ├──→ Loop 1: post_drafter.py → content_themes → Dashboard Tab 2
        └──→ Loop 2: signal feed → Dashboard Tab 1
```

## File layout (agent-based structure, as of the 2026-06 reorg)

The codebase was reorganised from flat `ingestion/`+`processing/`+root files into
per-agent packages under `agents/`, with cross-agent utilities under `shared/`.

```
python-backend/
├── CLAUDE.md                       (this file)
├── README.md, requirements.txt, .env.example
├── shared/                         (cross-agent utilities)
│   ├── config/settings.py          (env loading; was config/settings.py)
│   ├── db/turso_client.py          (libsql embedded-replica connect helper)
│   └── llm/openrouter.py           (OpenRouter chat-completion wrapper + retry)
├── agents/
│   └── brand_visibility/
│       ├── x/                      (KA017 — X/Twitter agent)
│       │   ├── db.py               (was ingestion/db.py)
│       │   ├── classifier.py       (was processing/classifier.py)
│       │   ├── x_scraper.py        (sweep logic; was ingestion/x_scraper.py)
│       │   ├── lexicon.py          (was ingestion/lexicon.py)
│       │   ├── providers/          (twitter241, twitter_api45, x_official, base, _http)
│       │   ├── promoter_tier.py, reputation.py   (was processing/*)
│       │   ├── post_drafter.py     (was output/post_drafter.py)
│       │   ├── orchestrator.py     (was root orchestrator.py)
│       │   └── scheduler.py        (was root scheduler.py)
│       └── linkedin/               (KA018 — LinkedIn agent)
│           ├── scraper.py, classifier.py, db.py, orchestrator.py
├── config/                         (KEPT: prompts/, genesis_lexicon.json)
├── output/                         (Streamlit dashboard — dashboard.py, ka018_page.py, ...)
├── scripts/                        (generate_lexicon.py, backfill_*, sweep_10_test.py, ...)
├── tests/
└── data/                           (gitignored — replica .db files + logs)
    ├── ka017_replica.db
    └── ka018_replica.db
```

Import roots: `shared.config.settings`, `shared.db.turso_client`,
`shared.llm.openrouter`, `agents.brand_visibility.x.*`,
`agents.brand_visibility.linkedin.*`. Run the X orchestrator with
`python -m agents.brand_visibility.x.orchestrator`; LinkedIn with
`python -m agents.brand_visibility.linkedin.orchestrator`.

KA018 (LinkedIn) is the sibling agent: scrapes LinkedIn posts (Fresh LinkedIn
Scraper API), classifies them into TIER_1_ENGAGE … TIER_4_NOISE, surfaced on the
dashboard's "KA018 LinkedIn" page.

## Conventions

- **Python 3.11+.** Use type hints. Use `from __future__ import annotations` at file top.
- **Pydantic v2** for all LLM response validation.
- **No ORM** — plain `sqlite3` with parameterized queries. The schema is in `ingestion/db.py`.
- **No global mutable state.** Everything reads from `config/settings.py` or DB.
- **Tests live in `tests/`.** Use `pytest`. Test the classifier with a golden set of 20 hand-labeled tweets (`tests/golden_tweets.json`).
- **Logging:** use the stdlib `logging` module via `logger = logging.getLogger(__name__)`. Never use `print()` in production paths.

## API and cost discipline

- X API: respect `x-rate-limit-reset` first; back off on 429 with `tenacity`.
- OpenRouter: log every call's input/output tokens to `llm_costs`. Cheap model (Haiku) for classification (high volume); expensive model (Sonnet) only for content drafting (low volume).
- SQLite: use `INSERT ... ON CONFLICT DO UPDATE` for idempotent upserts, never `INSERT OR IGNORE` (it silently drops engagement updates).

## Definitions

- **Tick**: one full orchestrator cycle. Default 30 min in production.
- **Velocity**: engagement (weighted) per minute since tweet creation. Used to flag urgent items.
- **Theme**: a cluster of 5+ classified tweets sharing tag intersection in the last 7 days. Theme → drafted post.
- **Priority flag**: one of `URGENT_INFLUENCER_REPLY`, `URGENT_VIRAL`, `STANDARD`, `LOW_PRIORITY_CONTENT`. Drives dashboard sort order.

## Things to flag, not implement

If you (Claude Code) are asked to:
- Auto-post to X from any account
- Generate replies to other users' tweets from non-KiteAI personas
- Bypass X rate limits or use unofficial APIs
- Scrape user-private data (DMs, follower lists at scale)

...stop and ask before doing it. These are explicit boundaries from the project's risk review.

## Data layer (as of migration)

KA017 reads and writes to Turso via embedded replica at `data/ka017_replica.db`.
The local replica is a cache — source of truth is Turso.

- Lexicon (keywords, classes, influencers): read from Turso `keywords`,
  `keyword_classes`, `influencers` tables. Falls back to
  `config/genesis_lexicon.json` if Turso is unreachable.

- Operational writes: `scraped_tweets`, `agent_runs`, `agent_activity`,
  `llm_costs`, `api_log`, `query_state`, `content_themes` — all in Turso.

- The dashboard (separate Next.js repo at `C:\Users\anooj\kiteai-dashboard-next`)
  reads from `agent_runs` and `agent_activity` to show what KA017 has done.

- Required env vars: `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`. Same values as
  the dashboard's `.env.local`.

## Current pipeline (post-v2 upgrade)

1. **Scrape:** twitter241 (RapidAPI by davethebeast) -- /search-v2, /user-tweets, /user-replies-v2. Hard budget of MAX_API_CALLS_PER_RUN (default 12) API calls per tick().
2. **Lexicon:** Read from Turso keywords table (enabled=1 only). Falls back to genesis_lexicon.json on Turso failure.
3. **Classify:** Gemini Flash 2.5 via OpenRouter (google/gemini-2.5-flash). Outputs relevance_score (0-100), confirmed_class (A-K or NOISE), noise_reason (when score < 40).
4. **Cluster + Draft:** Sonnet -- unchanged from before.
5. **Data layer:** Turso (libsql-client HTTP transport). Local SQLite at data/ka017_memory.db kept as archive.

## Why noise_reason exists

When the classifier rates a tweet under 40, it explains why in 25 words or less. A week of accumulated noise_reason values per keyword tells us which keywords are structurally noisy (e.g., "AWS Lambda rate limits" matching "API rate limit"). Use this for keyword pruning via the dashboard.

## API budget

MAX_API_CALLS_PER_RUN=12 (configurable in .env). After 12 calls, the run finishes in-memory work (classification of already-fetched tweets) but does not initiate more scrape calls. Logged to agent_activity and visible in the dashboard Activity panel.

## twitter241 user_id caching

twitter241's /user-tweets and /user-replies-v2 require numeric user IDs, not handles. The scraper layer checks the user_id_cache table (Turso) before calling /user for resolution. Without caching, every influencer sweep tick doubles API cost. Cache is populated on first resolution and reused across ticks.

## Demo dashboard (output/dashboard.py)

The Streamlit dashboard is split across `output/dashboard.py` (6-page UI),
`output/dashboard_queries.py` (cached Turso reads via `agents.brand_visibility.x.db.Database`),
and `output/dashboard_styles.py` (CSS + per-class colour palette). Constraints:

- Five of six pages (Overview, Signal Feed, Workflow, Keywords, Costs & Health)
  are strictly **read-only**: no writes, no external API calls, no config edits.
- **Exception — the "Run Agent" page CAN trigger live runs** (added by explicit
  request for manual/demo use). It does NOT call `tick()` in-process; it spawns
  `orchestrator.py` as a **subprocess** via `sys.executable` and streams its
  logs. Two buttons: "Run Collection" (keywords → influencers) and "Run
  Classification" (loops `--mode classify`, max 15 batches). This is a manual
  trigger, NOT a scheduler, and is local-only (redesign needed if hosted).
- **WAL-corruption guard (critical):** the embedded libsql replica corrupts
  under concurrent access. The Run Agent page therefore evicts the dashboard's
  cached DB handle (`q.get_db.clear()`) before a run so the **subprocess is the
  sole replica accessor**, and never reads the DB while a run is active. Do not
  reintroduce in-process DB access during a subprocess run, and never run the
  two buttons concurrently (both are disabled while either is active).
- All data comes from Turso (the same DB the agent writes to). The local
  `data/ka017_memory.db` archive is not shown.
- Keyword/influencer/rule edits happen in the Next.js dashboard, never here.
