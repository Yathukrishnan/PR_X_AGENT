"""Author reputation label derivation. Pure functions, no DB access."""
from __future__ import annotations


def derive_reputation_label(
    total_tweets: int,
    marketing_count: int,
    govt_promotion_count: int,
) -> str:
    """Compute a reputation label from raw counts.

    Returns one of:
      - 'unknown'         (insufficient data, < 3 tweets)
      - 'known_promoter'  (high promotional ratio, enough samples)
      - 'mixed'           (some promotion, some signal)
      - 'clean_signal'    (consistent signal, no/few promotional posts)
      - 'institutional'   (mostly GOVT_PROMOTION specifically)
    """
    # Insufficient data — withhold judgment
    if total_tweets < 3:
        return "unknown"

    promotional = marketing_count + govt_promotion_count
    promo_ratio = promotional / total_tweets

    # Mostly govt_promotion specifically — its own category (BHASHINI etc.)
    govt_ratio = govt_promotion_count / total_tweets
    if govt_ratio >= 0.6:
        return "institutional"

    # High promotional ratio → known promoter
    if promo_ratio >= 0.7:
        return "known_promoter"

    # Moderate ratio → mixed (some promotion, some real content)
    if promo_ratio >= 0.3:
        return "mixed"

    # Low ratio → clean signal
    return "clean_signal"


def compute_promotional_ratio(
    total_tweets: int,
    marketing_count: int,
    govt_promotion_count: int,
) -> float:
    """Compute promotional_ratio for storage. Returns 0.0 if total_tweets is 0."""
    if total_tweets == 0:
        return 0.0
    return (marketing_count + govt_promotion_count) / total_tweets
