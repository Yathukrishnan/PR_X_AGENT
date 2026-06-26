"""
Classify PENDING tweets via OpenRouter (Google Gemini Flash 2.5).
Usage: python -m agents.brand_visibility.x.classifier
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from shared.config.settings import (
    CLASSIFIER_MODEL,
    LLM_SLEEP_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE,
)
from shared.llm import openrouter
from agents.brand_visibility.x.db import Database

logger = logging.getLogger(__name__)

# Module-level prompt loading. The prompt lives in config/prompts/<active>.txt
# and is identified by config/prompts/active.txt (which contains the filename
# without extension). Read once per batch via _load_system_prompt(); the
# classifier never holds a stale prompt across batches.

from pathlib import Path

# config/ lives at the python-backend root: agents/brand_visibility/x/ -> parents[3]
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"
_ACTIVE_POINTER = _PROMPTS_DIR / "active.txt"
_FALLBACK_PROMPT = (
    "You are an analyst classifying X (Twitter) posts for voice AI market relevance. "
    "Output JSON with fields: relevance_score (0-100), confirmed_class (A-K or NOISE), "
    "intent_signal (BUILDER_PAIN/BUILDER_QUESTION/RECOMMENDATION/OBSERVATION/MARKETING/GOVT_PROMOTION), "
    "is_builder (0 or 1), quality_score (0-10), theme_tags (list), competitor_mentioned (list), "
    "summary_one_line (str), noise_reason (str if score<40 else empty)."
)


def _load_system_prompt() -> str:
    """Read the active classifier prompt from disk.

    Looks up config/prompts/active.txt to find the active prompt name,
    then reads config/prompts/<name>.txt. Returns a minimal fallback if
    either file is missing. Read fresh on every call so the next batch
    picks up any edits saved through the dashboard.
    """
    try:
        if not _ACTIVE_POINTER.exists():
            logger.warning("active.txt missing at %s; using fallback prompt", _ACTIVE_POINTER)
            return _FALLBACK_PROMPT
        active_name = _ACTIVE_POINTER.read_text(encoding="utf-8").strip()
        if not active_name:
            logger.warning("active.txt empty; using fallback prompt")
            return _FALLBACK_PROMPT
        prompt_path = _PROMPTS_DIR / f"{active_name}.txt"
        if not prompt_path.exists():
            logger.warning("Prompt file missing: %s; using fallback", prompt_path)
            return _FALLBACK_PROMPT
        return prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to load system prompt; using fallback: %s", exc)
        return _FALLBACK_PROMPT


class ClassifierOutput(BaseModel):
    relevance_score: int = Field(ge=0, le=100)
    confirmed_class: str
    intent_signal: str
    is_builder: int = Field(ge=0, le=1)
    quality_score: int = Field(ge=0, le=10)
    theme_tags: list[str] = Field(default_factory=list)
    competitor_mentioned: list[str] = Field(default_factory=list)
    summary_one_line: str
    noise_reason: str = ""
    # Token usage from OpenRouter response (populated by classify_one, used for cost logging).
    # Default 0 ensures model_validate accepts responses that don't include these fields.
    input_tokens: int = 0
    output_tokens: int = 0


# Cost constants for Gemini Flash 2.5 (USD per 1M tokens)
GEMINI_FLASH_INPUT_PER_M: float = 0.30
GEMINI_FLASH_OUTPUT_PER_M: float = 2.50


def _estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * GEMINI_FLASH_INPUT_PER_M + (
        output_tokens / 1_000_000
    ) * GEMINI_FLASH_OUTPUT_PER_M


# Pricing for cost computation, $/million tokens.
# Source: OpenRouter pricing page (verify at https://openrouter.ai/models when adding new entries).
# Updated: 2026-06-15
MODEL_PRICING: dict[str, dict[str, float]] = {
    "google/gemini-2.5-flash": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
}


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float]:
    """Compute (input_cost_usd, output_cost_usd, total_cost_usd) for a call.

    Returns (0.0, 0.0, 0.0) for unknown models — caller should log a warning
    so we know pricing config needs updating, but processing continues.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return (0.0, 0.0, 0.0)
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    return (input_cost, output_cost, input_cost + output_cost)


def classify_one(tweet: dict, system_prompt: str | None = None) -> ClassifierOutput | None:
    if system_prompt is None:
        system_prompt = _load_system_prompt()
    handle = tweet.get("author_handle") or tweet.get("handle") or "unknown"
    followers = tweet.get("author_followers") or tweet.get("followers") or 0
    text = tweet.get("text") or ""
    tweet_id = tweet.get("tweet_id", "unknown")

    user_msg = f"Tweet by @{handle} (followers: {followers}):\n\n{text}"

    payload: dict[str, Any] = {
        "model": CLASSIFIER_MODEL,
        "temperature": 0.1,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    }

    try:
        body = openrouter.chat_completion(
            OPENROUTER_BASE, OPENROUTER_API_KEY, payload, title="KA017"
        )
    except Exception as exc:
        logger.error("OpenRouter request error for tweet %s: %s", tweet_id, exc)
        return None

    try:
        raw_content = body["choices"][0]["message"]["content"]
        raw_obj = json.loads(raw_content)
        result = ClassifierOutput.model_validate(raw_obj)
        # Attach real token counts from OpenRouter usage data
        usage = body.get("usage", {}) or {}
        result.input_tokens = int(usage.get("prompt_tokens") or 0)
        result.output_tokens = int(usage.get("completion_tokens") or 0)
        return result
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Classifier parse failure for tweet %s: %s", tweet_id, exc)
        return None


def classify_pending(
    db: Database,
    limit: int = 50,
    dry_run: bool = False,
    run_id: int | None = None,
) -> dict:
    stats: dict[str, Any] = {
        "classified": 0,
        "skipped": 0,
        "noise": 0,
        "promoters_added": 0,
        "reputation_updated": 0,
        "total_cost_usd": 0.0,
    }

    pending = db.get_unclassified(limit=limit)

    for row in pending:
        # Normalise sqlite3.Row or plain sequence to dict
        if hasattr(row, "keys"):
            tweet: dict = dict(row)
        else:
            tweet = dict(row)

        tweet_id = tweet.get("tweet_id", "unknown")

        if dry_run:
            logger.info("[DRY-RUN] Would classify tweet %s", tweet_id)
            stats["skipped"] += 1
            continue

        result = classify_one(tweet)

        if result is None:
            stats["skipped"] += 1
            continue

        db.mark_classified(tweet_id, result.model_dump())

        # Persist relevance_score and noise_reason directly on the row
        with db._conn() as conn:
            conn.execute(
                "UPDATE scraped_tweets SET relevance_score=?, noise_reason=? WHERE tweet_id=?",
                [result.relevance_score, result.noise_reason, tweet_id],
            )

        # Route MARKETING tweets to the useful_promoters research corpus.
        # GOVT_PROMOTION is deliberately NOT routed. Any failure here must not
        # break classification, so it is fully wrapped.
        if result.intent_signal == "MARKETING":
            try:
                from agents.brand_visibility.x.promoter_tier import compute_tier, infer_promotion_kind

                tier = compute_tier(
                    author_followers=tweet.get("author_followers") or 0,
                    like_count=tweet.get("like_count") or 0,
                    retweet_count=tweet.get("retweet_count") or 0,
                    reply_count=tweet.get("reply_count") or 0,
                )
                kind = infer_promotion_kind(
                    tweet_text=tweet.get("text") or "",
                    author_handle=tweet.get("author_handle") or "",
                    matched_class=tweet.get("matched_class") or "",
                )
                inserted = db.add_useful_promoter(
                    tweet_id=tweet_id,
                    author_handle=tweet.get("author_handle") or "",
                    author_followers=tweet.get("author_followers") or 0,
                    matched_class=tweet.get("matched_class") or "",
                    promotion_kind=kind,
                    tier=tier,
                )
                if inserted:
                    stats["promoters_added"] += 1
            except Exception:
                logger.exception("useful_promoters routing failed for tweet %s", tweet_id)

        # Author reputation overlay — runs for EVERY classified tweet, not just
        # MARKETING. Recompute the author's running reputation now that this
        # tweet is counted, then stamp the label onto this row. Fully wrapped so
        # a failure here never blocks classification.
        try:
            author_handle = tweet.get("author_handle") or ""
            if author_handle:
                db.upsert_author_reputation(author_handle)
                label = db.get_author_reputation_label(author_handle)
                db.update_tweet_reputation_label(tweet_id, label)
                stats["reputation_updated"] += 1
        except Exception as exc:
            logger.warning("reputation update failed for %s: %s", tweet_id, exc)

        if result.relevance_score < 40:
            stats["noise"] += 1

        stats["classified"] += 1

        # Use actual token counts from OpenRouter response (fallback to estimates if missing)
        in_tok = result.input_tokens if result.input_tokens > 0 else 400
        out_tok = result.output_tokens if result.output_tokens > 0 else 200
        est_cost = _estimate_cost_usd(in_tok, out_tok)
        stats["total_cost_usd"] += est_cost

        db.log_llm_cost(
            purpose="classify",
            model=CLASSIFIER_MODEL,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=est_cost,
            run_id=run_id,
        )

        if run_id is not None:
            db.log_activity(
                run_id,
                phase="classify",
                event="tweet_classified",
                message=f"score={result.relevance_score} class={result.confirmed_class}",
                meta={"tweet_id": tweet_id},
            )

        time.sleep(LLM_SLEEP_SECONDS)

    logger.info("classification done: %s", stats)
    return stats
