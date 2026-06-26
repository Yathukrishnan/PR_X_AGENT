"""
KA018 data-access layer for the linkedin_* tables.

Mirrors KA017's ingestion/db.py: a thin class over Turso via the `libsql`
embedded-replica driver (cache at data/ka018_replica.db; source of truth is the
SAME Turso DB KA017 uses). Agent-owned tables (linkedin_posts, linkedin_runs,
linkedin_schedule, linkedin_prompts) are created here; linkedin_keywords is
managed externally and read-only from here.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from dotenv import load_dotenv

from shared.db import turso_client

# Load .env so the module is self-sufficient when imported directly (e.g. a bare
# `python -c "from linkedin.db import LinkedInDatabase"`), not only via shared.config.settings.
load_dotenv()

logger = logging.getLogger(__name__)

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
TURSO_SYNC_INTERVAL = int(os.getenv("TURSO_SYNC_INTERVAL", "60"))

# data/ lives at the python-backend root: agents/brand_visibility/linkedin/ -> parents[3]
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
REPLICA_PATH = _DATA_DIR / "ka018_replica.db"

# Volume is ordinal — used by get_keywords(min_volume=...).
_VOLUME_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Agent-owned schema. linkedin_keywords is intentionally absent — external.
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS linkedin_posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, post_urn TEXT UNIQUE,
  author_name TEXT, author_headline TEXT, author_urn TEXT,
  author_profile_url TEXT, author_followers INTEGER, text TEXT, posted_at TEXT,
  like_count INTEGER DEFAULT 0, comment_count INTEGER DEFAULT 0,
  repost_count INTEGER DEFAULT 0, post_url TEXT, matched_keyword TEXT,
  query_string TEXT, source_class TEXT, matched_category TEXT, raw_json TEXT,
  ingested_at TIMESTAMP, status TEXT DEFAULT 'PENDING', classified_at TIMESTAMP,
  confirmed_class TEXT, intent_signal TEXT, relevance_score INTEGER,
  summary_one_line TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_lp_status   ON linkedin_posts(status)",
    "CREATE INDEX IF NOT EXISTS idx_lp_category ON linkedin_posts(matched_category)",
    "CREATE INDEX IF NOT EXISTS idx_lp_keyword  ON linkedin_posts(matched_keyword)",
    """CREATE TABLE IF NOT EXISTS linkedin_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT DEFAULT 'KA018',
  started_at TIMESTAMP, completed_at TIMESTAMP, mode TEXT, status TEXT,
  keywords_queried INTEGER DEFAULT 0, api_calls_made INTEGER DEFAULT 0,
  posts_ingested INTEGER DEFAULT 0, posts_classified INTEGER DEFAULT 0,
  error_count INTEGER DEFAULT 0, notes TEXT
)""",
    """CREATE TABLE IF NOT EXISTS linkedin_schedule (
  id INTEGER PRIMARY KEY CHECK (id = 1), enabled INTEGER DEFAULT 0,
  interval_minutes INTEGER DEFAULT 1440, max_keywords INTEGER DEFAULT 5,
  max_pages INTEGER DEFAULT 1, categories TEXT, min_volume TEXT DEFAULT 'HIGH',
  date_posted TEXT DEFAULT 'past_week', sort_by TEXT DEFAULT 'date_posted',
  last_run_at TEXT, next_run_at TEXT, updated_at TEXT
)""",
    """CREATE TABLE IF NOT EXISTS linkedin_active_prompt (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  prompt_text TEXT NOT NULL,
  prompt_version TEXT NOT NULL DEFAULT 'v0',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)""",
    """CREATE TABLE IF NOT EXISTS linkedin_classification_costs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, post_id INTEGER NOT NULL,
  model TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, classified_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (post_id) REFERENCES linkedin_posts(id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_lcc_post_id ON linkedin_classification_costs(post_id)",
]

_POST_COLS = [
    "post_urn", "author_name", "author_headline", "author_urn",
    "author_profile_url", "author_followers", "text", "posted_at",
    "like_count", "comment_count", "repost_count", "post_url",
    "matched_keyword", "query_string", "source_class", "matched_category",
]

_SCHEDULE_COLUMNS = {
    "enabled", "interval_minutes", "max_keywords", "max_pages", "categories",
    "min_volume", "date_posted", "sort_by", "last_run_at", "next_run_at",
}


def _row_to_dict(row, columns: list[str]) -> dict:
    return {col: val for col, val in zip(columns, row)}


class LinkedInDatabase:
    """Turso-backed data access for KA018's linkedin_* tables."""

    def __init__(self, sync_interval: int | None = TURSO_SYNC_INTERVAL) -> None:
        """sync_interval: seconds between background replica syncs.
        Default 60 suits read-mostly callers (dashboard, ad-hoc queries). Pass
        None to disable background sync entirely — required for write-heavy runs
        (orchestrator sweeps), where a background sync firing mid-write triggers
        a libsql WalConflict. With None, call sync() manually at run start/end."""
        if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
            raise RuntimeError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set "
                               "in .env (same values as KA017 / the dashboard).")
        _DATA_DIR.mkdir(exist_ok=True)
        self._conn_obj = turso_client.connect(
            REPLICA_PATH, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN, sync_interval=sync_interval
        )
        self._conn_obj.sync()
        self._init_schema()
        self._migrate_columns()

    def _init_schema(self) -> None:
        for stmt in SCHEMA_STATEMENTS:
            try:
                self._conn_obj.execute(stmt)
            except Exception as exc:
                logger.warning("Schema statement failed (may already exist): %s", exc)
        self._conn_obj.commit()

    def _migrate_columns(self) -> None:
        """Idempotent ALTER TABLE ADD COLUMN for classifier outputs. SQLite has no
        'ADD COLUMN IF NOT EXISTS', so duplicate-column errors are swallowed.
        Note: classification_class did not exist on linkedin_posts (the base table
        has confirmed_class); the classifier uses classification_class, so we add
        it here alongside the three new *_score columns."""
        migrations = [
            "ALTER TABLE linkedin_posts ADD COLUMN commercial_fit_score REAL",
            "ALTER TABLE linkedin_posts ADD COLUMN relationship_value_score REAL",
            "ALTER TABLE linkedin_posts ADD COLUMN engagement_safety_score REAL",
            "ALTER TABLE linkedin_posts ADD COLUMN classification_class TEXT",
        ]
        for stmt in migrations:
            try:
                self._conn_obj.execute(stmt)
            except Exception:
                pass  # column already exists
        self._conn_obj.commit()

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
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

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [_row_to_dict(row, cols) for row in cur.fetchall()]

    def sync(self) -> None:
        try:
            self._conn_obj.sync()
        except Exception:
            logger.exception("Turso sync failed (continuing with local replica)")

    # --- Keywords (read-only; managed externally) ---
    def get_keywords(
        self,
        categories: list[str] | None = None,
        min_volume: str | None = None,
        min_source_count: int = 1,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[dict]:
        """Read linkedin_keywords with optional filters. min_volume is ordinal
        (LOW<MEDIUM<HIGH): 'MEDIUM' returns MEDIUM and HIGH. Ordered by
        source_count desc (cross-LLM agreement first), then volume desc."""
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("is_active = 1")
        if categories:
            clauses.append(f"category IN ({', '.join('?' for _ in categories)})")
            params.extend(categories)
        if min_volume:
            allowed = [v for v, r in _VOLUME_RANK.items()
                       if r >= _VOLUME_RANK.get(min_volume.upper(), 0)]
            if allowed:
                clauses.append(f"volume_estimate IN ({', '.join('?' for _ in allowed)})")
                params.extend(allowed)
        if min_source_count and min_source_count > 1:
            clauses.append("source_count >= ?")
            params.append(min_source_count)

        sql = "SELECT * FROM linkedin_keywords"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += (" ORDER BY source_count DESC, CASE volume_estimate "
                "WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END DESC")
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.query(sql, tuple(params))

    # --- Posts ---
    def insert_post(self, row: dict) -> int | None:
        """Insert one normalized post. Returns the new row id, or None on a
        UNIQUE(post_urn) conflict (already ingested) or a missing post_urn."""
        post_urn = row.get("post_urn")
        if not post_urn:
            logger.warning("insert_post: row has no post_urn — skipping")
            return None
        raw = row.get("raw")
        _zero = {"like_count", "comment_count", "repost_count"}  # default 0, not NULL
        values = [row.get(c, 0) if c in _zero else row.get(c) for c in _POST_COLS]
        values += [json.dumps(raw) if raw is not None else None, self._now()]
        placeholders = ", ".join("?" for _ in range(len(_POST_COLS) + 2))
        with self._conn() as conn:
            if conn.execute("SELECT 1 FROM linkedin_posts WHERE post_urn = ?",
                            [post_urn]).fetchall():
                return None
            cur = conn.execute(
                f"INSERT OR IGNORE INTO linkedin_posts "
                f"({', '.join(_POST_COLS)}, raw_json, ingested_at, status) "
                f"VALUES ({placeholders}, 'PENDING')",
                values,
            )
            return cur.lastrowid or None  # 0 -> ignored -> None

    def insert_posts_batch(self, rows: list[dict]) -> tuple[int, int]:
        """Insert many posts. Returns (inserted, skipped_duplicates)."""
        inserted = skipped = 0
        for row in rows:
            if self.insert_post(row) is not None:
                inserted += 1
            else:
                skipped += 1
        return inserted, skipped

    def list_recent_posts(self, limit: int = 50, category: str | None = None) -> list[dict]:
        if category:
            return self.query(
                "SELECT * FROM linkedin_posts WHERE matched_category = ? "
                "ORDER BY ingested_at DESC LIMIT ?", (category, limit))
        return self.query(
            "SELECT * FROM linkedin_posts ORDER BY ingested_at DESC LIMIT ?", (limit,))

    def count_posts(self) -> int:
        rows = self.query("SELECT COUNT(*) AS c FROM linkedin_posts")
        return rows[0]["c"] if rows else 0

    def count_posts_by_category(self) -> dict:
        rows = self.query("SELECT matched_category, COUNT(*) AS c FROM linkedin_posts "
                          "GROUP BY matched_category ORDER BY c DESC")
        return {r["matched_category"]: r["c"] for r in rows}

    # --- Run lifecycle ---
    def start_run(self, mode: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO linkedin_runs (agent_id, started_at, mode, status) "
                "VALUES ('KA018', ?, ?, 'running')", (self._now(), mode))
            return cur.lastrowid

    def finish_run(
        self,
        run_id: int,
        keywords_queried: int,
        api_calls_made: int,
        posts_ingested: int,
        posts_classified: int = 0,
        error_count: int = 0,
        notes: str | None = None,
    ) -> None:
        status = "failed" if error_count and posts_ingested == 0 else "completed"
        with self._conn() as conn:
            conn.execute(
                "UPDATE linkedin_runs SET completed_at=?, status=?, keywords_queried=?, "
                "api_calls_made=?, posts_ingested=?, posts_classified=?, "
                "error_count=?, notes=? WHERE id=?",
                (self._now(), status, keywords_queried, api_calls_made,
                 posts_ingested, posts_classified, error_count, notes, run_id))

    # --- Schedule (single row, id = 1) ---
    def get_schedule(self) -> dict:
        """Return the single schedule row, creating defaults on first access."""
        rows = self.query("SELECT * FROM linkedin_schedule WHERE id = 1")
        if rows:
            return rows[0]
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO linkedin_schedule (id, updated_at) "
                         "VALUES (1, ?)", (self._now(),))
        rows = self.query("SELECT * FROM linkedin_schedule WHERE id = 1")
        return rows[0] if rows else {}

    def update_schedule(self, **kwargs: Any) -> None:
        """Update schedule fields. Unknown keys are ignored with a warning."""
        self.get_schedule()  # ensure the row exists
        set_clauses: list[str] = []
        params: list[Any] = []
        for col, val in kwargs.items():
            if col not in _SCHEDULE_COLUMNS:
                logger.warning("update_schedule: ignoring unknown field %r", col)
                continue
            set_clauses.append(f"{col} = ?")
            params.append(val)
        if not set_clauses:
            return
        set_clauses.append("updated_at = ?")
        params.extend([self._now(), 1])
        with self._conn() as conn:
            conn.execute(f"UPDATE linkedin_schedule SET {', '.join(set_clauses)} "
                         f"WHERE id = ?", tuple(params))

    # --- Active prompt (single-row pointer; id = 1) ---
    def get_active_prompt(self) -> dict | None:
        rows = self.query("SELECT * FROM linkedin_active_prompt WHERE id = 1")
        return rows[0] if rows else None

    def set_active_prompt(self, prompt_text: str, prompt_version: str = "v1") -> None:
        """Upsert the single active prompt row (id = 1)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO linkedin_active_prompt (id, prompt_text, prompt_version, "
                "updated_at) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET prompt_text=excluded.prompt_text, "
                "prompt_version=excluded.prompt_version, updated_at=excluded.updated_at",
                (prompt_text, prompt_version, self._now()))

    # --- Classification costs (LLM cost tracking) ---
    def record_classification_cost(
        self, post_id: int, model: str, input_tokens: int | None,
        output_tokens: int | None, cost_usd: float | None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO linkedin_classification_costs (post_id, model, "
                "input_tokens, output_tokens, cost_usd, classified_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (post_id, model, input_tokens, output_tokens, cost_usd, self._now()))

    def total_cost_this_month(self) -> float:
        rows = self.query(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM "
            "linkedin_classification_costs WHERE "
            "strftime('%Y-%m', classified_at) = strftime('%Y-%m', 'now')")
        return float(rows[0]["total"]) if rows else 0.0

    def cost_summary_by_model(self) -> list[tuple]:
        """Return [(model, total_cost_usd, post_count), ...] across all time."""
        rows = self.query(
            "SELECT model, COALESCE(SUM(cost_usd), 0) AS total, "
            "COUNT(DISTINCT post_id) AS posts FROM linkedin_classification_costs "
            "GROUP BY model ORDER BY total DESC")
        return [(r["model"], float(r["total"]), r["posts"]) for r in rows]
