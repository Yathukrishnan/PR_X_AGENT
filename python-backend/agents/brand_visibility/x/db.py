"""
Agent-owned data access layer. Reads/writes to Turso via the `libsql` package
with embedded replica caching at data/ka017_replica.db.

Methods are identical to the previous SQLite version — callers don't change.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from shared.db import turso_client
from shared.config.settings import (
    REPLICA_PATH,
    TURSO_AUTH_TOKEN,
    TURSO_DATABASE_URL,
    TURSO_SYNC_INTERVAL,
)

logger = logging.getLogger(__name__)

# Agent-owned tables only. Dashboard tables (keywords, keyword_classes,
# influencers, classification_rules, settings, api_key_status) are managed by
# the dashboard and must NOT be created or altered here.
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS scraped_tweets (
  tweet_id             TEXT PRIMARY KEY,
  created_at           TIMESTAMP,
  author_id            TEXT,
  author_handle        TEXT,
  author_followers     INTEGER,
  author_bio           TEXT,
  text                 TEXT,
  like_count           INTEGER DEFAULT 0,
  reply_count          INTEGER DEFAULT 0,
  retweet_count        INTEGER DEFAULT 0,
  quote_count          INTEGER DEFAULT 0,
  impression_count     INTEGER,
  lang                 TEXT,
  source_type          TEXT,
  matched_class        TEXT,
  matched_query        TEXT,
  source_handle        TEXT,
  conversation_id      TEXT,
  ingested_at          TIMESTAMP,
  last_seen_at         TIMESTAMP,
  velocity             REAL,
  priority_flag        TEXT,
  classified_at        TIMESTAMP,
  confirmed_class      TEXT,
  intent_signal        TEXT,
  quality_score        INTEGER,
  is_builder           INTEGER,
  theme_tags           TEXT,
  competitor_mentioned TEXT,
  summary_one_line     TEXT,
  relevance_score      INTEGER,
  noise_reason         TEXT,
  status               TEXT DEFAULT 'PENDING'
)""",
    "CREATE INDEX IF NOT EXISTS idx_tweets_status   ON scraped_tweets(status)",
    "CREATE INDEX IF NOT EXISTS idx_tweets_class    ON scraped_tweets(matched_class)",
    "CREATE INDEX IF NOT EXISTS idx_tweets_priority ON scraped_tweets(priority_flag)",
    "CREATE INDEX IF NOT EXISTS idx_tweets_author   ON scraped_tweets(author_handle)",
    """CREATE TABLE IF NOT EXISTS query_state (
  query_hash        TEXT PRIMARY KEY,
  query_text        TEXT,
  last_since_id     TEXT,
  last_run_at       TIMESTAMP,
  tweet_count_total INTEGER DEFAULT 0
)""",
    """CREATE TABLE IF NOT EXISTS content_themes (
  theme_id         TEXT PRIMARY KEY,
  theme_class      TEXT,
  tag_intersection TEXT,
  tweet_ids        TEXT,
  tweet_count      INTEGER,
  summary          TEXT,
  draft_post       TEXT,
  draft_format     TEXT,
  draft_rationale  TEXT,
  created_at       TIMESTAMP,
  posted_url       TEXT,
  status           TEXT DEFAULT 'DRAFT'
)""",
    """CREATE TABLE IF NOT EXISTS llm_costs (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id           TEXT DEFAULT 'KA017',
  called_at          TIMESTAMP,
  purpose            TEXT,
  model              TEXT,
  input_tokens       INTEGER,
  output_tokens      INTEGER,
  estimated_cost_usd REAL,
  related_id         TEXT
)""",
    """CREATE TABLE IF NOT EXISTS api_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id        TEXT DEFAULT 'KA017',
  called_at       TIMESTAMP,
  endpoint        TEXT,
  query_text      TEXT,
  status_code     INTEGER,
  tweets_returned INTEGER,
  rate_remaining  INTEGER,
  rate_reset_at   TIMESTAMP,
  notes           TEXT
)""",
    """CREATE TABLE IF NOT EXISTS agent_runs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id        TEXT DEFAULT 'KA017',
  started_at      TIMESTAMP,
  ended_at        TIMESTAMP,
  triggered_by    TEXT,
  status          TEXT,
  mode            TEXT,
  calls_used      INTEGER DEFAULT 0,
  records_new     INTEGER DEFAULT 0,
  records_updated INTEGER DEFAULT 0,
  error_message   TEXT,
  summary_json    TEXT
)""",
    """CREATE TABLE IF NOT EXISTS agent_activity (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       INTEGER,
  agent_id     TEXT DEFAULT 'KA017',
  occurred_at  TIMESTAMP,
  level        TEXT,
  phase        TEXT,
  event        TEXT,
  message      TEXT,
  meta_json    TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_activity_run   ON agent_activity(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_activity_agent ON agent_activity(agent_id)",
    """CREATE TABLE IF NOT EXISTS user_id_cache (
  handle    TEXT PRIMARY KEY,
  user_id   TEXT NOT NULL,
  cached_at TIMESTAMP
)""",
]


def _row_to_dict(row, columns: list[str]) -> dict:
    """libsql rows are tuples; pair them with column names from cursor.description."""
    return {col: val for col, val in zip(columns, row)}


class Database:
    """Turso-backed data access via the libsql embedded-replica driver."""

    def __init__(self, db_path: str | None = None) -> None:
        # db_path kept for backward compatibility — ignored, we use Turso.
        if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
            raise RuntimeError(
                "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env. "
                "Copy them from the dashboard's .env.local."
            )
        self.replica_path = str(REPLICA_PATH)
        self._conn_obj = turso_client.connect(
            self.replica_path,
            TURSO_DATABASE_URL,
            TURSO_AUTH_TOKEN,
            sync_interval=TURSO_SYNC_INTERVAL,
        )
        self._conn_obj.sync()
        self._init_schema()
        self._migrate_columns()
        self._migrate_useful_promoters()
        self._migrate_author_reputation()
        self._migrate_classification_costs()

    def _init_schema(self) -> None:
        for stmt in SCHEMA_STATEMENTS:
            try:
                self._conn_obj.execute(stmt)
            except Exception as exc:
                logger.warning("Schema statement failed (may already exist): %s", exc)
        self._conn_obj.commit()

    def _migrate_columns(self) -> None:
        """ALTER TABLE ADD COLUMN — idempotent because we swallow 'already exists'."""
        migrations = [
            "ALTER TABLE scraped_tweets ADD COLUMN relevance_score INTEGER",
            "ALTER TABLE scraped_tweets ADD COLUMN noise_reason TEXT",
            "ALTER TABLE scraped_tweets ADD COLUMN author_reputation_label TEXT DEFAULT NULL",
        ]
        for stmt in migrations:
            try:
                self._conn_obj.execute(stmt)
                logger.info("Applied column migration: %s", stmt)
            except Exception:
                pass  # column already exists
        self._conn_obj.commit()

    def _migrate_useful_promoters(self) -> None:
        """Create the useful_promoters research-corpus table (additive, idempotent).

        Captures tweets classified as MARKETING intent. Owned by KA017. Each
        statement uses IF NOT EXISTS so re-running on every Database() init is a
        no-op once the table/indexes exist.
        """
        statements = [
            """CREATE TABLE IF NOT EXISTS useful_promoters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_handle TEXT NOT NULL,
                author_followers INTEGER,
                matched_class TEXT,
                tweet_id TEXT NOT NULL,
                promotion_kind TEXT NOT NULL,
                tier TEXT NOT NULL,
                added_at TEXT NOT NULL,
                UNIQUE(tweet_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_useful_promoters_handle ON useful_promoters(author_handle)",
            "CREATE INDEX IF NOT EXISTS idx_useful_promoters_tier ON useful_promoters(tier)",
            "CREATE INDEX IF NOT EXISTS idx_useful_promoters_kind ON useful_promoters(promotion_kind)",
            "CREATE INDEX IF NOT EXISTS idx_useful_promoters_added_at ON useful_promoters(added_at)",
        ]
        for stmt in statements:
            try:
                self._conn_obj.execute(stmt)
            except Exception as exc:
                logger.warning("useful_promoters migration statement failed (may already exist): %s", exc)
        self._conn_obj.commit()
        logger.info("useful_promoters table ready")

    def _migrate_author_reputation(self) -> None:
        """Create the author_reputation overlay table (additive, idempotent)."""
        statements = [
            """CREATE TABLE IF NOT EXISTS author_reputation (
                author_handle TEXT PRIMARY KEY,
                total_tweets INTEGER NOT NULL DEFAULT 0,
                marketing_count INTEGER NOT NULL DEFAULT 0,
                govt_promotion_count INTEGER NOT NULL DEFAULT 0,
                signal_count INTEGER NOT NULL DEFAULT 0,
                noise_count INTEGER NOT NULL DEFAULT 0,
                promotional_ratio REAL NOT NULL DEFAULT 0.0,
                reputation_label TEXT NOT NULL DEFAULT 'unknown',
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_author_reputation_label ON author_reputation(reputation_label)",
            "CREATE INDEX IF NOT EXISTS idx_author_reputation_ratio ON author_reputation(promotional_ratio)",
        ]
        for stmt in statements:
            try:
                self._conn_obj.execute(stmt)
            except Exception as exc:
                logger.warning("author_reputation migration statement failed (may already exist): %s", exc)
        self._conn_obj.commit()
        logger.info("author_reputation table ready")

    def _migrate_classification_costs(self) -> None:
        """Create the classification_costs table (additive, idempotent).

        Per-classification token usage + cost. Composite PK
        (tweet_id, classified_at) supports re-classification — a tweet
        classified twice at different times yields two cost rows.
        """
        statements = [
            """CREATE TABLE IF NOT EXISTS classification_costs (
                tweet_id TEXT NOT NULL,
                classified_at TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                input_cost_usd REAL NOT NULL,
                output_cost_usd REAL NOT NULL,
                total_cost_usd REAL NOT NULL,
                PRIMARY KEY (tweet_id, classified_at)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_costs_classified_at ON classification_costs(classified_at)",
        ]
        for stmt in statements:
            try:
                self._conn_obj.execute(stmt)
            except Exception as exc:
                logger.warning("classification_costs migration statement failed (may already exist): %s", exc)
        self._conn_obj.commit()
        logger.info("classification_costs table ready")

    # Allowed enum values for useful_promoters
    _PROMOTION_KINDS = {"vendor", "self", "agency", "course", "vc_portfolio"}
    _PROMOTION_TIERS = {"high", "medium", "low"}

    def add_useful_promoter(
        self,
        tweet_id: str,
        author_handle: str,
        author_followers: int,
        matched_class: str,
        promotion_kind: str,
        tier: str,
    ) -> bool:
        """Insert a MARKETING tweet into useful_promoters.

        Returns True if a row was inserted, False if the tweet_id was already
        present (idempotent — re-classification/backfill never duplicates).
        Raises ValueError on an invalid promotion_kind or tier.
        """
        if promotion_kind not in self._PROMOTION_KINDS:
            raise ValueError(
                f"invalid promotion_kind {promotion_kind!r}; "
                f"must be one of {sorted(self._PROMOTION_KINDS)}"
            )
        if tier not in self._PROMOTION_TIERS:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(self._PROMOTION_TIERS)}"
            )
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM useful_promoters WHERE tweet_id = ?", [tweet_id]
            ).fetchall()
            if existing:
                return False
            # INSERT OR IGNORE as belt-and-suspenders against the UNIQUE constraint.
            conn.execute(
                """INSERT OR IGNORE INTO useful_promoters
                   (author_handle, author_followers, matched_class, tweet_id,
                    promotion_kind, tier, added_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    author_handle,
                    author_followers,
                    matched_class,
                    tweet_id,
                    promotion_kind,
                    tier,
                    self._now(),
                ],
            )
            return True

    # ------------------------------------------------------------------
    # Author reputation overlay
    # ------------------------------------------------------------------

    def upsert_author_reputation(self, author_handle: str) -> dict:
        """Recompute and upsert reputation for a single author.

        Reads all of this author's CLASSIFIED tweets, computes counts + label,
        writes to author_reputation (preserving first_seen_at). Idempotent.
        Returns the new reputation row as a dict (empty dict if no tweets).
        """
        from agents.brand_visibility.x.reputation import (
            compute_promotional_ratio,
            derive_reputation_label,
        )

        if not author_handle:
            return {}

        rows = self.query(
            """SELECT intent_signal, relevance_score
               FROM scraped_tweets
               WHERE author_handle = ? AND status = 'CLASSIFIED'""",
            (author_handle,),
        )
        if not rows:
            return {}

        total = len(rows)
        marketing = sum(1 for r in rows if r.get("intent_signal") == "MARKETING")
        govt = sum(1 for r in rows if r.get("intent_signal") == "GOVT_PROMOTION")
        noise = sum(1 for r in rows if (r.get("relevance_score") or 0) < 40)
        signal_intents = {"OBSERVATION", "RECOMMENDATION", "BUILDER_PAIN", "BUILDER_QUESTION"}
        signal = sum(1 for r in rows if r.get("intent_signal") in signal_intents)

        ratio = compute_promotional_ratio(total, marketing, govt)
        label = derive_reputation_label(total, marketing, govt)
        now = self._now()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO author_reputation
                   (author_handle, total_tweets, marketing_count, govt_promotion_count,
                    signal_count, noise_count, promotional_ratio, reputation_label,
                    first_seen_at, last_updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(author_handle) DO UPDATE SET
                       total_tweets = excluded.total_tweets,
                       marketing_count = excluded.marketing_count,
                       govt_promotion_count = excluded.govt_promotion_count,
                       signal_count = excluded.signal_count,
                       noise_count = excluded.noise_count,
                       promotional_ratio = excluded.promotional_ratio,
                       reputation_label = excluded.reputation_label,
                       last_updated_at = excluded.last_updated_at""",
                [author_handle, total, marketing, govt, signal, noise, ratio, label, now, now],
            )

        return {
            "author_handle": author_handle,
            "total_tweets": total,
            "marketing_count": marketing,
            "govt_promotion_count": govt,
            "signal_count": signal,
            "noise_count": noise,
            "promotional_ratio": ratio,
            "reputation_label": label,
        }

    def get_author_reputation_label(self, author_handle: str) -> str:
        """Return the reputation_label for an author, or 'unknown' if not found."""
        if not author_handle:
            return "unknown"
        rows = self.query(
            "SELECT reputation_label FROM author_reputation WHERE author_handle = ?",
            (author_handle,),
        )
        if not rows:
            return "unknown"
        return rows[0]["reputation_label"]

    def update_tweet_reputation_label(self, tweet_id: str, label: str) -> None:
        """Set author_reputation_label on a single tweet row."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE scraped_tweets SET author_reputation_label = ? WHERE tweet_id = ?",
                [label, tweet_id],
            )

    def stamp_reputation_by_handle(self, author_handle: str, label: str) -> None:
        """Stamp the reputation label on all of an author's CLASSIFIED tweets in
        one statement. One commit per author instead of one per tweet — far less
        WAL pressure on the embedded replica during backfill."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE scraped_tweets SET author_reputation_label = ? "
                "WHERE author_handle = ? AND status = 'CLASSIFIED'",
                [label, author_handle],
            )

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        """Context manager kept for source compatibility with the old SQLite version."""
        try:
            yield self._conn_obj
            self._conn_obj.commit()
        except Exception:
            try:
                self._conn_obj.rollback()
            except Exception:
                pass
            raise

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def sync(self) -> None:
        """Force a remote-to-local sync."""
        try:
            self._conn_obj.sync()
        except Exception:
            logger.exception("Turso sync failed (continuing with local replica)")

    # ------------------------------------------------------------------
    # Tweet upsert — refreshes engagement + last_seen_at on re-encounter
    # ------------------------------------------------------------------

    def upsert_tweet(self, tweet: dict[str, Any]) -> None:
        now = self._now()
        ingested_at = tweet.get("ingested_at", now)
        is_builder = tweet.get("is_builder")
        if is_builder is not None:
            is_builder = int(bool(is_builder))
        # The libsql driver accepts only positional (?) params, not :named dicts.
        values = [
            tweet.get("tweet_id"),
            tweet.get("created_at"),
            tweet.get("author_id"),
            tweet.get("author_handle"),
            tweet.get("author_followers"),
            tweet.get("author_bio"),
            tweet.get("text"),
            tweet.get("like_count"),
            tweet.get("reply_count"),
            tweet.get("retweet_count"),
            tweet.get("quote_count"),
            tweet.get("impression_count"),
            tweet.get("lang"),
            tweet.get("source_type"),
            tweet.get("matched_class"),
            tweet.get("matched_query"),
            tweet.get("source_handle"),
            tweet.get("conversation_id"),
            ingested_at,
            now,
            tweet.get("velocity"),
            tweet.get("priority_flag"),
        ]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO scraped_tweets (
                  tweet_id, created_at, author_id, author_handle, author_followers,
                  author_bio, text, like_count, reply_count, retweet_count, quote_count,
                  impression_count, lang, source_type, matched_class, matched_query,
                  source_handle, conversation_id, ingested_at, last_seen_at,
                  velocity, priority_flag, status
                ) VALUES (
                  ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?, 'PENDING'
                )
                ON CONFLICT(tweet_id) DO UPDATE SET
                  like_count       = excluded.like_count,
                  reply_count      = excluded.reply_count,
                  retweet_count    = excluded.retweet_count,
                  quote_count      = excluded.quote_count,
                  impression_count = excluded.impression_count,
                  last_seen_at     = excluded.last_seen_at,
                  velocity         = excluded.velocity,
                  priority_flag    = CASE
                    WHEN scraped_tweets.priority_flag = 'URGENT_INFLUENCER_REPLY'
                    THEN 'URGENT_INFLUENCER_REPLY'
                    ELSE excluded.priority_flag
                  END
                """,
                values,
            )

    def insert_classification_cost(
        self,
        tweet_id: str,
        classified_at: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        input_cost_usd: float,
        output_cost_usd: float,
        total_cost_usd: float,
    ) -> None:
        """Record cost metadata for one classification call.

        Composite PK (tweet_id, classified_at) supports re-classification — a
        tweet classified twice at different times yields two cost rows. Use
        INSERT OR REPLACE so a rerun of the same exact (tweet_id, timestamp)
        is idempotent without raising.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO classification_costs (
                    tweet_id, classified_at, model,
                    input_tokens, output_tokens,
                    input_cost_usd, output_cost_usd, total_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tweet_id, classified_at, model,
                    input_tokens, output_tokens,
                    input_cost_usd, output_cost_usd, total_cost_usd,
                ),
            )

    def update_tweet_engagement(self, tweet_id: str, metrics: dict[str, int]) -> None:
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE scraped_tweets SET
                  like_count    = ?,
                  reply_count   = ?,
                  retweet_count = ?,
                  quote_count   = ?,
                  last_seen_at  = ?
                WHERE tweet_id = ?
                """,
                [
                    metrics.get("like_count"),
                    metrics.get("reply_count"),
                    metrics.get("retweet_count"),
                    metrics.get("quote_count"),
                    now,
                    tweet_id,
                ],
            )

    # ------------------------------------------------------------------
    # Classifier queue
    # ------------------------------------------------------------------

    def get_unclassified(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM scraped_tweets WHERE status = 'PENDING' "
                "ORDER BY priority_flag, created_at DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cur.description] if cur.description else []
            return [_row_to_dict(row, cols) for row in cur.fetchall()]

    def mark_classified(self, tweet_id: str, classification: dict[str, Any]) -> None:
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE scraped_tweets SET
                  confirmed_class      = ?,
                  intent_signal        = ?,
                  quality_score        = ?,
                  is_builder           = ?,
                  theme_tags           = ?,
                  competitor_mentioned = ?,
                  summary_one_line     = ?,
                  classified_at        = ?,
                  status               = 'CLASSIFIED'
                WHERE tweet_id = ?
                """,
                [
                    classification["confirmed_class"],
                    classification["intent_signal"],
                    classification["quality_score"],
                    int(bool(classification["is_builder"])),
                    json.dumps(classification.get("theme_tags", [])),
                    json.dumps(classification.get("competitor_mentioned", [])),
                    classification.get("summary_one_line", ""),
                    now,
                    tweet_id,
                ],
            )

    # ------------------------------------------------------------------
    # Clustering / drafting
    # ------------------------------------------------------------------

    def get_for_clustering(self, quality_threshold: int = 6, days: int = 7) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                """
                SELECT * FROM scraped_tweets
                WHERE status = 'CLASSIFIED'
                  AND quality_score >= ?
                  AND confirmed_class != 'NOISE'
                  AND classified_at >= datetime('now', ? || ' days')
                ORDER BY confirmed_class, quality_score DESC
                """,
                (quality_threshold, f"-{days}"),
            )
            cols = [d[0] for d in cur.description] if cur.description else []
            return [_row_to_dict(row, cols) for row in cur.fetchall()]

    def get_pending_drafts(self) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM content_themes WHERE status = 'DRAFT' ORDER BY created_at DESC"
            )
            cols = [d[0] for d in cur.description] if cur.description else []
            return [_row_to_dict(row, cols) for row in cur.fetchall()]

    def upsert_theme(self, theme: dict[str, Any]) -> None:
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO content_themes (
                  theme_id, theme_class, tag_intersection, tweet_ids,
                  tweet_count, summary, created_at, status
                ) VALUES (
                  ?, ?, ?, ?,
                  ?, ?, ?, 'DRAFT'
                )
                ON CONFLICT(theme_id) DO UPDATE SET
                  tweet_count = excluded.tweet_count,
                  tweet_ids   = excluded.tweet_ids
                """,
                [
                    theme.get("theme_id"),
                    theme.get("theme_class"),
                    theme.get("tag_intersection"),
                    theme.get("tweet_ids"),
                    theme.get("tweet_count"),
                    theme.get("summary"),
                    now,
                ],
            )

    def save_draft(self, theme_id: str, draft_post: str, draft_format: str, rationale: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE content_themes SET
                  draft_post      = ?,
                  draft_format    = ?,
                  draft_rationale = ?,
                  status          = 'DRAFT'
                WHERE theme_id = ?
                """,
                (draft_post, draft_format, rationale, theme_id),
            )

    def update_status(self, table: str, id_col: str, id_val: str, status: str, **extra: Any) -> None:
        allowed_tables = {"scraped_tweets", "content_themes"}
        if table not in allowed_tables:
            raise ValueError(f"Unknown table: {table}")
        set_clauses = ["status = ?"]
        params: list[Any] = [status]
        for col, val in extra.items():
            set_clauses.append(f"{col} = ?")
            params.append(val)
        params.append(id_val)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {id_col} = ?",
                tuple(params),
            )

    # ------------------------------------------------------------------
    # Query state (for since_id pagination)
    # ------------------------------------------------------------------

    def get_query_state(self, query_hash: str) -> dict | None:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM query_state WHERE query_hash = ?", (query_hash,)
            )
            cols = [d[0] for d in cur.description] if cur.description else []
            row = cur.fetchone()
            return _row_to_dict(row, cols) if row else None

    def set_query_state(
        self,
        query_hash: str,
        query_text: str,
        last_since_id: str,
        tweet_count_delta: int = 0,
    ) -> None:
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO query_state (query_hash, query_text, last_since_id, last_run_at, tweet_count_total)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                  last_since_id     = excluded.last_since_id,
                  last_run_at       = excluded.last_run_at,
                  tweet_count_total = tweet_count_total + ?
                """,
                (query_hash, query_text, last_since_id, now, tweet_count_delta, tweet_count_delta),
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_api_call(
        self,
        endpoint: str,
        query_text: str,
        status_code: int,
        tweets_returned: int,
        rate_remaining: int | None,
        rate_reset_at: str | None,
        notes: str = "",
        service: str = "twitter",
    ) -> None:
        # The Turso api_log table (shared with the dashboard) only has columns:
        # agent_id, called_at, service, endpoint, status_code, month_key.
        # The richer fields (query_text, tweets_returned, rate_*, notes) have no
        # column here — they remain in the caller signature for source-compat but
        # are not persisted. month_key drives the dashboard's monthly quota view.
        now = self._now()
        month_key = now[:7]  # 'YYYY-MM'
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO api_log (agent_id, called_at, service, endpoint,
                  status_code, month_key)
                VALUES ('KA017', ?, ?, ?, ?, ?)
                """,
                (now, service, endpoint, status_code, month_key),
            )

    def log_llm_cost(
        self,
        purpose: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        run_id: int | None = None,
    ) -> None:
        # Turso llm_costs columns: agent_id, called_at, purpose, model,
        # input_tokens, output_tokens, estimated_cost_usd, run_id.
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO llm_costs (agent_id, called_at, purpose, model,
                  input_tokens, output_tokens, estimated_cost_usd, run_id)
                VALUES ('KA017', ?, ?, ?, ?, ?, ?, ?)
                """,
                (now, purpose, model, input_tokens, output_tokens,
                 estimated_cost_usd, run_id),
            )

    # ------------------------------------------------------------------
    # Agent run lifecycle (visible to the dashboard's Activity panel)
    # ------------------------------------------------------------------

    def start_run(self, mode: str, triggered_by: str = "manual") -> int:
        # Turso agent_runs has no `mode` column — fold mode into triggered_by so
        # the dashboard still shows which pipeline phase ran.
        now = self._now()
        triggered = f"{triggered_by}:{mode}" if mode else triggered_by
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO agent_runs (agent_id, started_at, triggered_by, status)
                VALUES ('KA017', ?, ?, 'running')
                """,
                (now, triggered),
            )
            return cur.lastrowid

    def finish_run(
        self,
        run_id: int,
        status: str,
        calls_used: int = 0,
        records_new: int = 0,
        records_updated: int = 0,
        error_message: str | None = None,
        summary: dict | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE agent_runs SET
                  ended_at        = ?,
                  status          = ?,
                  calls_used      = ?,
                  records_new     = ?,
                  records_updated = ?,
                  error_message   = ?,
                  summary_json    = ?
                WHERE id = ?
                """,
                (
                    self._now(),
                    status,
                    calls_used,
                    records_new,
                    records_updated,
                    error_message,
                    json.dumps(summary) if summary else None,
                    run_id,
                ),
            )

    def log_activity(
        self,
        run_id: int | None,
        phase: str,
        event: str,
        message: str = "",
        level: str = "INFO",
        meta: dict | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO agent_activity
                  (run_id, agent_id, occurred_at, level, phase, event, message, meta_json)
                VALUES (?, 'KA017', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._now(),
                    level,
                    phase,
                    event,
                    message,
                    json.dumps(meta) if meta else None,
                ),
            )

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [_row_to_dict(row, cols) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # User ID cache (handle → Twitter numeric user_id)
    # ------------------------------------------------------------------

    def get_user_id(self, handle: str) -> str | None:
        normalized = handle.lstrip("@").lower()
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT user_id FROM user_id_cache WHERE handle = ?",
                (normalized,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set_user_id(self, handle: str, user_id: str) -> None:
        normalized = handle.lstrip("@").lower()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO user_id_cache (handle, user_id, cached_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(handle) DO UPDATE SET
                     user_id = excluded.user_id,
                     cached_at = excluded.cached_at""",
                (normalized, user_id, self._now()),
            )