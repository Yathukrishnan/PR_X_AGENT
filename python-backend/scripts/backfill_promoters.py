"""Backfill useful_promoters from existing MARKETING-tagged scraped_tweets.

Run AFTER the v2 classifier prompt has been applied and the corpus has been
re-classified. Idempotent — safe to run multiple times.

  py -3 scripts/backfill_promoters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.brand_visibility.x.db import Database
from agents.brand_visibility.x.promoter_tier import compute_tier, infer_promotion_kind


def main() -> None:
    db = Database()
    rows = db.query(
        """SELECT tweet_id, author_handle, author_followers, matched_class,
                  text, like_count, retweet_count, reply_count
           FROM scraped_tweets
           WHERE status = 'CLASSIFIED' AND intent_signal = 'MARKETING'"""
    )

    inserted = 0
    skipped = 0
    for row in rows:
        tier = compute_tier(
            row.get("author_followers") or 0,
            row.get("like_count") or 0,
            row.get("retweet_count") or 0,
            row.get("reply_count") or 0,
        )
        kind = infer_promotion_kind(
            row.get("text") or "",
            row.get("author_handle") or "",
            row.get("matched_class") or "",
        )
        was_inserted = db.add_useful_promoter(
            tweet_id=row["tweet_id"],
            author_handle=row.get("author_handle") or "",
            author_followers=row.get("author_followers") or 0,
            matched_class=row.get("matched_class") or "",
            promotion_kind=kind,
            tier=tier,
        )
        if was_inserted:
            inserted += 1
        else:
            skipped += 1

    print(
        f"Backfill complete: inserted={inserted}, skipped={skipped}, "
        f"total_marketing_tweets={len(rows)}"
    )


if __name__ == "__main__":
    main()
