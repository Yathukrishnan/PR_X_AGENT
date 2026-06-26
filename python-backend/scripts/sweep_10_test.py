"""
One-off diverse-coverage sweep: 10 explicit keywords across categories.
Hard cap 10 API calls (1 page each). Scrapes + stores only — no classification.

Run:
  py -3 scripts/sweep_10_test.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()  # must precede scraper import (it reads the API key at import time)

from agents.brand_visibility.linkedin import scraper
from agents.brand_visibility.linkedin.db import LinkedInDatabase
from agents.brand_visibility.linkedin.orchestrator import (
    _CATEGORY_TO_SOURCE_CLASS,
    _month_to_date_calls,
)

KEYWORDS = [
    "elevenlabs", "vapi", "cartesia",
    "speech-to-speech", "voice cloning",
    "tts latency", "barge-in detection",
    "outbound lead qualification", "multilingual customer support agent",
    "voice latency too high",
]
MAX_CALLS = 10
MONTHLY_BUDGET = 50


def main() -> None:
    # Write-heavy: disable background sync to avoid mid-write WalConflict; sync
    # manually at start (pull) and end (push).
    db = LinkedInDatabase(sync_interval=None)
    db.sync()

    used_before = _month_to_date_calls(db)
    remaining = MONTHLY_BUDGET - used_before
    print(f"Budget: used {used_before}/{MONTHLY_BUDGET} this month, {remaining} remaining")
    if remaining < len(KEYWORDS):
        print(f"WARNING: only {remaining} calls left; will stop when budget hits 0.")

    run_id = db.start_run(mode="diverse_10_test")
    api_calls = posts_ingested = skipped = errors = kw_queried = 0

    for kw in KEYWORDS:
        if api_calls >= MAX_CALLS or (remaining - api_calls) <= 0:
            print(f"Budget/cap reached before {kw!r} — stopping cleanly.")
            break

        row = db.query("SELECT category FROM linkedin_keywords WHERE keyword = ?", (kw,))
        category = row[0]["category"] if row else ""
        if not row:
            print(f"  NOTE: {kw!r} not found in linkedin_keywords; category=''")
        source_class = _CATEGORY_TO_SOURCE_CLASS.get(category, "grouped")

        try:
            raw_posts = scraper.search_posts(
                keyword=kw, date_posted="past_week",
                sort_by="date_posted", max_pages=1,
            )
            api_calls += 1
            kw_queried += 1
        except Exception as exc:
            errors += 1
            print(f"  ERROR querying {kw!r}: {exc}")
            continue

        rows = [
            scraper.normalize_post(p, kw, kw, source_class, category)
            for p in raw_posts
        ]
        ins, skp = db.insert_posts_batch(rows)
        posts_ingested += ins
        skipped += skp
        print(f"  {kw!r:42} cat={category:14} posts={len(rows):3} new={ins:3} dup={skp}")

        time.sleep(3)

    db.finish_run(
        run_id,
        keywords_queried=kw_queried,
        api_calls_made=api_calls,
        posts_ingested=posts_ingested,
        error_count=errors,
        notes=f"diverse_10_test; skipped_duplicates={skipped}",
    )
    db.sync()  # push the run's writes to Turso (no background sync with sync_interval=None)

    print("\n=== Sweep summary ===")
    print(f"  keywords_queried : {kw_queried}")
    print(f"  api_calls_made   : {api_calls}")
    print(f"  posts_ingested   : {posts_ingested}")
    print(f"  duplicates       : {skipped}")
    print(f"  errors           : {errors}")

    used_after = _month_to_date_calls(db)
    print(f"\nBudget status: used {used_after}/{MONTHLY_BUDGET} this month, "
          f"{MONTHLY_BUDGET - used_after} remaining")


if __name__ == "__main__":
    main()
