"""Quick targeted classifier: process only the most recent unclassified Class C tweets.

For demo prep — bypasses the older pending tweets to get fresh Class C data
classified quickly without burning Gemini quota on older backlog.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the python-backend root is importable (scripts/ -> parent).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.brand_visibility.x.db import Database
from agents.brand_visibility.x.classifier import CLASSIFIER_MODEL, classify_one, compute_cost

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("classify_class_c")

LIMIT = 300  # safety margin over 266

def main() -> int:
    db = Database()
    rows = db.query(
        "SELECT tweet_id, author_handle, author_followers, text, matched_class, matched_query "
        "FROM scraped_tweets "
        "WHERE matched_class = ? AND classified_at IS NULL "
        "ORDER BY ingested_at DESC "
        "LIMIT ?",
        ('C', LIMIT)
    )
    total = len(rows)
    logger.info("Found %d unclassified Class C tweets. Starting classification.", total)

    classified = 0
    skipped = 0
    noise = 0
    for i, row in enumerate(rows, 1):
        tweet = dict(row)
        tweet_id = tweet['tweet_id']
        try:
            result = classify_one(tweet)
            if result is None:
                skipped += 1
                logger.warning("[%d/%d] tweet %s skipped (parse failure or API error)", i, total, tweet_id)
                continue

            # Persist results — same shape as classify_pending
            with db._conn() as conn:
                if result.relevance_score < 40 or result.confirmed_class == "NOISE":
                    conn.execute(
                        "UPDATE scraped_tweets SET classified_at=datetime('now'), "
                        "confirmed_class=?, intent_signal=?, relevance_score=?, "
                        "quality_score=?, is_builder=?, theme_tags=?, "
                        "competitor_mentioned=?, summary_one_line=?, noise_reason=?, "
                        "status='CLASSIFIED' WHERE tweet_id=?",
                        (
                            result.confirmed_class, result.intent_signal,
                            result.relevance_score, result.quality_score,
                            result.is_builder, ",".join(result.theme_tags),
                            ",".join(result.competitor_mentioned),
                            result.summary_one_line, result.noise_reason, tweet_id,
                        )
                    )
                    noise += 1
                else:
                    conn.execute(
                        "UPDATE scraped_tweets SET classified_at=datetime('now'), "
                        "confirmed_class=?, intent_signal=?, relevance_score=?, "
                        "quality_score=?, is_builder=?, theme_tags=?, "
                        "competitor_mentioned=?, summary_one_line=?, noise_reason=?, "
                        "status='CLASSIFIED' WHERE tweet_id=?",
                        (
                            result.confirmed_class, result.intent_signal,
                            result.relevance_score, result.quality_score,
                            result.is_builder, ",".join(result.theme_tags),
                            ",".join(result.competitor_mentioned),
                            result.summary_one_line, result.noise_reason, tweet_id,
                        )
                    )
                    classified += 1

            # Persist classification cost (no-op if model pricing unknown).
            in_cost, out_cost, total_cost = compute_cost(
                CLASSIFIER_MODEL,
                result.input_tokens,
                result.output_tokens,
            )
            try:
                db.insert_classification_cost(
                    tweet_id=tweet_id,
                    classified_at=datetime.now(timezone.utc).isoformat(),
                    model=CLASSIFIER_MODEL,
                    input_tokens=result.input_tokens or 0,
                    output_tokens=result.output_tokens or 0,
                    input_cost_usd=in_cost,
                    output_cost_usd=out_cost,
                    total_cost_usd=total_cost,
                )
            except Exception as cost_exc:
                logger.warning(
                    "Cost persistence failed for tweet %s: %s",
                    tweet_id, cost_exc,
                )

            if i % 25 == 0:
                logger.info("[%d/%d] progress: %d classified, %d noise, %d skipped", i, total, classified, noise, skipped)
        except Exception as exc:
            logger.exception("[%d/%d] tweet %s failed: %s", i, total, tweet_id, exc)
            skipped += 1

    logger.info(
        "DONE. Total: %d | Classified (signal): %d | NOISE: %d | Skipped: %d",
        total, classified, noise, skipped,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())