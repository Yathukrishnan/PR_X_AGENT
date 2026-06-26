"""
Theme clustering + post drafting for @KiteAI's content pipeline.
Usage:
  python -m agents.brand_visibility.x.post_drafter --cluster
  python -m agents.brand_visibility.x.post_drafter --draft
  python -m agents.brand_visibility.x.post_drafter --cluster --draft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

from shared.config.settings import DRAFTER_MODEL, LLM_SLEEP_SECONDS, OPENROUTER_API_KEY, OPENROUTER_BASE
from agents.brand_visibility.x.db import Database

logger = logging.getLogger(__name__)

DRAFTER_SYSTEM_PROMPT = """You are drafting a post for @KiteAI's own X account. KiteAI builds voice-AI infrastructure with a focus on edge deployment and multilingual support (Arabic dialects, Indian languages, SEA languages).

You are writing as KiteAI, openly. This is marketing content, but it must earn attention by being technically substantive — not by being clever or salesy. Imagine the reader is a senior engineer who will close the tab the moment they smell a pitch.

Given the theme cluster below, draft a single X post (under 280 chars) OR a thread (3-7 posts, each under 280 chars) that:

1. States the technical problem precisely, in the words a builder would use
2. Offers a real observation, decomposition, or data point about it
3. Mentions KiteAI's relevant capability only if it follows naturally from the technical content — never as a CTA, never with a link in the same post
4. Ends with a question or hook that invites reply, not a sales close

Forbidden:
- Marketing voice ("excited to announce", "we're thrilled")
- Hashtags
- Emoji except where technically clarifying
- Links in the main post
- Claiming benchmarks you haven't run
- Naming competitors negatively
- Mentioning specific individuals from the source tweets

Return JSON: { "format": "single" | "thread", "posts": ["..."], "rationale": "..." }"""


class DraftOutput(BaseModel):
    format: str
    posts: list[str]
    rationale: str


_DRAFTER_COST_PER_1K: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.5": (0.003, 0.015),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _DRAFTER_COST_PER_1K.get(model, (0.003, 0.015))
    return (input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate


def _anonymize(text: str) -> str:
    """Strip @handles and URLs from tweet text before sending to LLM."""
    text = re.sub(r"@\w+", "[user]", text)
    text = re.sub(r"https?://\S+", "[link]", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _theme_id(cls: str, tags: tuple[str, ...]) -> str:
    key = f"{cls}:{'|'.join(sorted(tags))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cluster_themes(db: Database, quality_threshold: int = 6, days: int = 7) -> list[dict]:
    rows = db.get_for_clustering(quality_threshold=quality_threshold, days=days)

    # Group by class → tag tuples
    class_tag_tweets: dict[str, dict[tuple[str, ...], list[str]]] = {}

    for row in rows:
        cls = row["confirmed_class"]
        tags_raw = row["theme_tags"] or "[]"
        try:
            tags: list[str] = json.loads(tags_raw)
        except json.JSONDecodeError:
            tags = []

        if not tags:
            continue

        if cls not in class_tag_tweets:
            class_tag_tweets[cls] = {}

        # All non-empty single tags as keys (also intersections handled below)
        for tag in tags:
            key = (tag,)
            class_tag_tweets[cls].setdefault(key, []).append(row["tweet_id"])

        # Pairs
        if len(tags) >= 2:
            for i, t1 in enumerate(tags):
                for t2 in tags[i + 1:]:
                    key = tuple(sorted([t1, t2]))
                    class_tag_tweets[cls].setdefault(key, []).append(row["tweet_id"])

    themes_saved: list[dict] = []
    MIN_CLUSTER_SIZE = 5

    for cls, tag_map in class_tag_tweets.items():
        for tag_tuple, tweet_ids in tag_map.items():
            unique_ids = list(dict.fromkeys(tweet_ids))  # preserve order, dedupe
            if len(unique_ids) < MIN_CLUSTER_SIZE:
                continue

            theme_id = _theme_id(cls, tag_tuple)
            summary = f"Class {cls} signal cluster around: {', '.join(tag_tuple)}"

            theme = {
                "theme_id": theme_id,
                "theme_class": cls,
                "tag_intersection": json.dumps(list(tag_tuple)),
                "tweet_ids": json.dumps(unique_ids[:50]),
                "tweet_count": len(unique_ids),
                "summary": summary,
            }
            db.upsert_theme(theme)
            themes_saved.append(theme)

    logger.info("cluster_themes: saved %d themes", len(themes_saved))
    return themes_saved


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

def _call_openrouter(messages: list[dict], model: str = DRAFTER_MODEL) -> dict[str, Any]:
    resp = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kiteai.dev",
            "X-Title": "KA017",
        },
        json={"model": model, "messages": messages, "temperature": 0.7},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def draft_post(theme: Any, db: Database, dry_run: bool = False) -> bool:
    theme_id = theme["theme_id"]
    cls = theme["theme_class"]
    tag_intersection = json.loads(theme["tag_intersection"] or "[]")
    tweet_ids = json.loads(theme["tweet_ids"] or "[]")

    # Fetch representative tweet texts (up to 5)
    rep_tweets: list[str] = []
    for tid in tweet_ids[:10]:
        rows = db.query("SELECT text FROM scraped_tweets WHERE tweet_id = ?", (tid,))
        if rows:
            rep_tweets.append(_anonymize(rows[0]["text"]))
        if len(rep_tweets) >= 5:
            break

    user_msg = (
        f"Theme class: {cls}\n"
        f"Tag intersection: {', '.join(tag_intersection)}\n"
        f"Theme summary: {theme['summary']}\n"
        f"Tweet count in cluster: {theme['tweet_count']}\n\n"
        f"Representative tweets (anonymized):\n"
        + "\n".join(f"- {t}" for t in rep_tweets)
        + "\n\nRecent KiteAI posts for voice consistency: None yet"
    )

    if dry_run:
        logger.info("[DRY-RUN] Would draft for theme %s: %s", theme_id, tag_intersection)
        return True

    messages = [
        {"role": "system", "content": DRAFTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        result = _call_openrouter(messages)
    except requests.HTTPError as exc:
        logger.error("OpenRouter draft call failed: %s", exc)
        return False

    raw = result["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        output = DraftOutput.model_validate_json(raw)
    except (ValidationError, Exception) as exc:
        logger.warning("Draft parse failure for theme %s: %s", theme_id, exc)
        return False

    db.save_draft(
        theme_id=theme_id,
        draft_post=json.dumps(output.posts),
        draft_format=output.format,
        rationale=output.rationale,
    )

    usage = result.get("usage", {})
    db.log_llm_cost(
        purpose="draft",
        model=DRAFTER_MODEL,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        estimated_cost_usd=_estimate_cost(
            DRAFTER_MODEL,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        ),
        related_id=theme_id,
    )

    logger.info("Drafted post for theme %s (%s)", theme_id, tag_intersection)
    return True


def draft_posts_for_new_themes(db: Database, dry_run: bool = False) -> dict[str, int]:
    rows = db.query(
        "SELECT * FROM content_themes WHERE status = 'DRAFT' AND draft_post IS NULL ORDER BY created_at DESC"
    )
    stats = {"drafted": 0, "failed": 0}

    for row in rows:
        success = draft_post(dict(row), db, dry_run=dry_run)
        if success:
            stats["drafted"] += 1
        else:
            stats["failed"] += 1
        if not dry_run:
            time.sleep(LLM_SLEEP_SECONDS)

    logger.info("draft_posts_for_new_themes done: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KA017 post drafter")
    parser.add_argument("--cluster", action="store_true", help="Run theme clustering")
    parser.add_argument("--draft", action="store_true", help="Draft posts for undrafted themes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not OPENROUTER_API_KEY and args.draft and not args.dry_run:
        logger.error("OPENROUTER_API_KEY not set")
        raise SystemExit(1)

    db = Database()

    if args.cluster:
        themes = cluster_themes(db)
        print(f"Clustered {len(themes)} themes")

    if args.draft:
        stats = draft_posts_for_new_themes(db, dry_run=args.dry_run)
        print(f"Drafting complete: {stats}")

    if not args.cluster and not args.draft:
        parser.print_help()


if __name__ == "__main__":
    main()
