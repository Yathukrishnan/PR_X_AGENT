"""
One-shot script to copy historical data from the local SQLite archive
(data/ka017_memory.db) into Turso. Idempotent — uses INSERT OR IGNORE so
re-running is safe.

Notes:
- Uses the `libsql` embedded-replica driver (same as ingestion/db.py), NOT
  libsql-client.
- Only migrates tables whose schema matches between the local archive and
  Turso: scraped_tweets, query_state, content_themes. The local llm_costs and
  api_log tables use the agent's OLD column layout, which differs from the
  dashboard-owned Turso schema — historical rows there are intentionally NOT
  migrated to avoid column mismatches.
- For each table, only columns present in BOTH the local archive and the Turso
  table are copied (column-intersection), so schema drift can't break the run.

Run once:
  py -3 scripts/migrate_local_to_turso.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import libsql

from shared.config.settings import (
    DB_PATH,
    REPLICA_PATH,
    TURSO_AUTH_TOKEN,
    TURSO_DATABASE_URL,
    TURSO_SYNC_INTERVAL,
)

# Only schema-compatible tables. llm_costs / api_log deliberately excluded
# (different Turso schema — see module docstring).
TABLES_TO_MIGRATE = [
    "scraped_tweets",
    "query_state",
    "content_themes",
]

COMMIT_EVERY = 200


def _turso_columns(dst: "libsql.Connection", table: str) -> list[str]:
    try:
        cur = dst.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]
    except Exception:
        return []


def main() -> None:
    if not DB_PATH.exists():
        print(f"Local DB {DB_PATH} not found — nothing to migrate.")
        return

    if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
        print("ERROR: TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not set in .env")
        sys.exit(1)

    print(f"Source: {DB_PATH}")
    print(f"Target: {TURSO_DATABASE_URL}")
    print()

    src = sqlite3.connect(str(DB_PATH))
    src.row_factory = sqlite3.Row

    dst = libsql.connect(
        str(REPLICA_PATH),
        sync_url=TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN,
        sync_interval=TURSO_SYNC_INTERVAL,
    )
    dst.sync()

    for table in TABLES_TO_MIGRATE:
        try:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
        except sqlite3.OperationalError:
            print(f"  {table}: not in local archive — skipping")
            continue

        if not rows:
            print(f"  {table}: 0 rows — skipping")
            continue

        local_cols = list(rows[0].keys())
        turso_cols = _turso_columns(dst, table)
        if not turso_cols:
            print(f"  {table}: not present in Turso — skipping")
            continue

        # Only copy columns that exist on both sides.
        cols = [c for c in local_cols if c in turso_cols]
        dropped = [c for c in local_cols if c not in turso_cols]
        if dropped:
            print(f"  {table}: ignoring local-only columns {dropped}")

        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"

        migrated = 0
        skipped = 0
        for i, row in enumerate(rows):
            try:
                dst.execute(sql, [row[c] for c in cols])
                migrated += 1
            except Exception as row_exc:
                skipped += 1
                if skipped <= 3:
                    print(f"    skip {table} row: {row_exc}")
            if (i + 1) % COMMIT_EVERY == 0:
                dst.commit()
        dst.commit()

        print(f"  {table}: migrated {migrated}, skipped {skipped} (of {len(rows)} total)")

    dst.sync()
    print()
    print("Verification (Turso row counts):")
    for table in TABLES_TO_MIGRATE:
        try:
            cur = dst.execute(f"SELECT COUNT(*) FROM {table}")
            n = cur.fetchone()[0]
            print(f"  {table}: {n}")
        except Exception as exc:
            print(f"  {table}: ERROR {exc}")

    print()
    print("Migration complete.")
    src.close()


if __name__ == "__main__":
    main()
