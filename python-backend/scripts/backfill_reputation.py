"""Backfill author_reputation from existing CLASSIFIED tweets.

Run ONCE after the migration is applied. Idempotent — safe to re-run.

  py -3 scripts/backfill_reputation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

from agents.brand_visibility.x.db import Database


def main() -> None:
    db = Database()
    print("Loading distinct authors with CLASSIFIED tweets...")
    rows = db.query(
        """SELECT DISTINCT author_handle
           FROM scraped_tweets
           WHERE status = 'CLASSIFIED'
             AND author_handle IS NOT NULL AND author_handle != ''"""
    )
    print(f"Found {len(rows)} distinct authors to process.")

    updated = 0
    for i, row in enumerate(rows):
        handle = row["author_handle"]
        result = db.upsert_author_reputation(handle)
        if not result:
            continue
        updated += 1
        # One bulk UPDATE stamps all of this author's tweets (1 commit, not N).
        db.stamp_reputation_by_handle(handle, result["reputation_label"])
        # Pace + flush periodically — the embedded replica destabilizes under
        # rapid unspaced commits (audit P0-2). Idempotent on re-run.
        time.sleep(0.05)
        if (i + 1) % 15 == 0:
            db.sync()
            print(f"  ...{i + 1}/{len(rows)} authors processed", flush=True)

    db.sync()
    print(f"Backfill complete: {updated} authors processed.")


if __name__ == "__main__":
    main()
