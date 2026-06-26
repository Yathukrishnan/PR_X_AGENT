"""
Pure helpers for classifying MARKETING tweets into the useful_promoters corpus.

No database access — these are stand-alone functions so they can be unit-tested
in isolation. Both are defensive against None/0 inputs.
"""
from __future__ import annotations


def compute_tier(
    author_followers: int | None,
    like_count: int | None,
    retweet_count: int | None,
    reply_count: int | None,
) -> str:
    """Compute promoter tier from follower count and tweet engagement.

    Returns one of: 'high', 'medium', 'low'. None inputs are treated as 0.
    """
    followers = author_followers or 0
    engagement = (like_count or 0) + (retweet_count or 0) + (reply_count or 0)

    if followers >= 50_000 or engagement >= 500:
        return "high"
    if (5_000 <= followers < 50_000) or (100 <= engagement < 500):
        return "medium"
    return "low"


# VC accounts — match the audit's findings
_VC_HANDLES = {
    "ycombinator", "lightspeedindia", "lightspeed", "a16z", "sequoia",
    "khoslaventures", "peakxvpartners", "founderscollective",
}

_COURSE_MARKERS = [
    "roadmap", "learning path", "from zero to", "phase 1:", "follow this",
    "20 advanced questions", "$400k", "best youtube channels", "mini project:",
]

_AGENCY_MARKERS = [
    "we help businesses", "i build custom", "i can build you", "dm me to",
    "book a call", "appointment setter", "fiverr.com", "automation agency",
    "we build", "my team and i can build",
]

_SELF_MARKERS = [
    "day 1 of", "day 2 of", "day 3 of", "build in public", "building in public",
    "#buildinpublic", "i made $", "subscribe free", "my newsletter", "my course",
    "weekly update on",
]


def infer_promotion_kind(
    tweet_text: str | None,
    author_handle: str | None,
    matched_class: str | None,
) -> str:
    """Infer the promotion kind for a MARKETING-tagged tweet.

    Returns one of: 'vendor', 'self', 'agency', 'course', 'vc_portfolio'.
    Simple string heuristics applied in priority order — first match wins.
    Defaults to 'vendor' (the most common kind in the audit).
    """
    text_lower = (tweet_text or "").lower()
    handle_lower = (author_handle or "").lower().lstrip("@")

    # VC accounts
    if handle_lower in _VC_HANDLES:
        return "vc_portfolio"

    # Course / learning-path content
    if any(marker in text_lower for marker in _COURSE_MARKERS):
        return "course"

    # Agency / build-for-hire
    if any(marker in text_lower for marker in _AGENCY_MARKERS):
        return "agency"

    # Self-promotion / build-in-public (also: author linking to their own handle)
    if any(marker in text_lower for marker in _SELF_MARKERS):
        return "self"
    if handle_lower and ("follow @" + handle_lower) in text_lower:
        return "self"

    # Default: vendor (third-party hyping someone else's product)
    return "vendor"
