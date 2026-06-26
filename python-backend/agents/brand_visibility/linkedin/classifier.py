"""
KA018 LinkedIn classifier — scores scraped posts for voice-AI lead potential.

Mirrors KA017's processing/classifier.py: module-level functions (no class),
OpenRouter via requests with tenacity retries on transient errors only, Pydantic
validation of the LLM JSON. The system prompt is read once at import from
config/prompts/linkedin_active.txt.

Usage: python -m linkedin.classifier   (or call classify_pending() directly)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, field_validator

from shared.config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE
from shared.llm import openrouter
from agents.brand_visibility.linkedin.db import LinkedInDatabase

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENROUTER_MODEL_CLASSIFIER", "google/gemini-2.5-flash")
LLM_SLEEP_SECONDS = 2

# Fallback pricing only — the authoritative cost comes from OpenRouter's
# usage.cost. These are OpenRouter's listed Gemini 2.5 Flash rates, USD per 1M
# tokens (NOT Google's direct-API rates).
_PRICING: dict[str, dict[str, float]] = {
    "google/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}
_FALLBACK_PRICING = {"in": 0.30, "out": 2.50}

_ALLOWED_FLAGS = {
    "recruiter", "homonym", "ai_generated", "off_topic",
    "competitor_promo", "non_english", "personal_update", "thought_leadership",
}

# --- System prompt: read once at module load, cached ---
# config/ lives at the python-backend root: agents/brand_visibility/linkedin/ -> parents[3]
_PROMPT_PATH = Path(__file__).resolve().parents[3] / "config" / "prompts" / "linkedin_active.txt"


def _load_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s (%s); using minimal fallback prompt", _PROMPT_PATH, exc)
        return (
            "Classify the LinkedIn post for voice-AI lead potential. Return JSON with "
            "relevance_voice_ai, commercial_fit, relationship_value, engagement_safety "
            "(ints 0-10), noise_flags (list), final_tier (TIER_1_ENGAGE/TIER_2_WATCH/"
            "TIER_3_SIGNAL/TIER_4_NOISE), one_line_reason (str)."
        )


_SYSTEM_PROMPT = _load_prompt()


# --------------------------------------------------------------------------
# Output model
# --------------------------------------------------------------------------

class LinkedInClassifierOutput(BaseModel):
    relevance_voice_ai: int
    commercial_fit: int
    relationship_value: int
    engagement_safety: int
    noise_flags: list[str] = []
    final_tier: Literal["TIER_1_ENGAGE", "TIER_2_WATCH", "TIER_3_SIGNAL", "TIER_4_NOISE"]
    one_line_reason: str
    # Internal — set by classify_one for cost logging.
    input_tokens: int = 0
    output_tokens: int = 0
    or_cost: float | None = None   # authoritative USD cost from OpenRouter usage.cost

    @field_validator(
        "relevance_voice_ai", "commercial_fit", "relationship_value",
        "engagement_safety", mode="before",
    )
    @classmethod
    def _clamp_scores(cls, v: Any) -> int:
        """Coerce to int and clamp to 0-10 (don't reject out-of-range scores)."""
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            v = 0
        return max(0, min(10, v))

    @field_validator("noise_flags")
    @classmethod
    def _drop_unknown_flags(cls, v: list[str]) -> list[str]:
        """Lenient: keep allowed flags, drop unknown ones with a warning."""
        kept = []
        for flag in v:
            if flag in _ALLOWED_FLAGS:
                kept.append(flag)
            else:
                logger.warning(f"Dropped unknown noise_flag: {flag!r}")
        return kept


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = _PRICING.get(model)
    if pricing is None:
        logger.warning("No pricing for model %r; using gemini-flash fallback rates", model)
        pricing = _FALLBACK_PRICING
    return (input_tokens / 1_000_000) * pricing["in"] + (output_tokens / 1_000_000) * pricing["out"]


# --------------------------------------------------------------------------
# OpenRouter call (transient-only retry)
# --------------------------------------------------------------------------

def _post_openrouter(user_msg: str) -> dict:
    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},  # ask OpenRouter to return actual USD cost
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    return openrouter.chat_completion(
        OPENROUTER_BASE, OPENROUTER_API_KEY, payload, title="KA018"
    )


def classify_one(post: dict) -> LinkedInClassifierOutput:
    """Classify a single post. Raises on LLM failure or unparseable/invalid JSON."""
    headline = post.get("author_headline") or ""
    text = post.get("text") or ""
    user_msg = f"Author headline: {headline}\n\nPost text:\n{text[:2000]}"

    body = _post_openrouter(user_msg)
    content = body["choices"][0]["message"]["content"]
    raw_obj = json.loads(content)  # raises JSONDecodeError on bad JSON
    result = LinkedInClassifierOutput.model_validate(raw_obj)  # raises ValidationError

    usage = body.get("usage") or {}
    result.input_tokens = int(usage.get("prompt_tokens") or 0)
    result.output_tokens = int(usage.get("completion_tokens") or 0)
    cost = usage.get("cost")
    result.or_cost = float(cost) if cost is not None else None
    return result


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def classify_pending(limit: int | None = None) -> dict:
    """Classify posts that have text and no classification_class yet (oldest first)."""
    # Write-heavy: disable background sync (avoids WalConflict); sync manually.
    db = LinkedInDatabase(sync_interval=None)
    db.sync()

    sql = ("SELECT * FROM linkedin_posts "
           "WHERE classification_class IS NULL AND LENGTH(text) > 0 ORDER BY id ASC")
    rows = db.query(sql + " LIMIT ?", (limit,)) if limit else db.query(sql)

    stats: dict[str, Any] = {"classified": 0, "skipped": 0, "errors": 0, "total_cost_usd": 0.0}
    first = True

    for post in rows:
        post_id = post.get("id")
        text = post.get("text") or ""
        if not text.strip():
            stats["skipped"] += 1
            continue

        try:
            result = classify_one(post)
        except Exception as exc:
            logger.warning("classify failed for post %s: %s", post_id, exc)
            stats["errors"] += 1
            continue

        noise = ",".join(result.noise_flags)
        with db._conn() as conn:
            conn.execute(
                "UPDATE linkedin_posts SET relevance_score=?, commercial_fit_score=?, "
                "relationship_value_score=?, engagement_safety_score=?, "
                "classification_class=?, intent_signal=?, summary_one_line=?, "
                "classified_at=datetime('now') WHERE id=?",
                [result.relevance_voice_ai, result.commercial_fit,
                 result.relationship_value, result.engagement_safety,
                 result.final_tier, noise, result.one_line_reason, post_id],
            )

        # Prefer OpenRouter's authoritative cost; fall back to local math.
        if result.or_cost is not None:
            cost = result.or_cost
            logger.debug("Cost source: openrouter_usage_cost=$%.8f", cost)
        else:
            cost = compute_cost(result.input_tokens, result.output_tokens, MODEL)
            logger.debug("Cost source: fallback_calculation=$%.8f", cost)
        db.record_classification_cost(
            post_id=post_id, model=MODEL,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            cost_usd=cost,
        )
        stats["classified"] += 1
        stats["total_cost_usd"] += cost

        if first:
            logger.info("=== First classification result (for prompt validation) ===")
            logger.info("Post text preview: %s", text[:200])
            logger.info(
                "Result: tier=%s scores=[rel=%d, com=%d, val=%d, eng=%d] noise=%s reason=%s",
                result.final_tier, result.relevance_voice_ai, result.commercial_fit,
                result.relationship_value, result.engagement_safety,
                noise or "(none)", result.one_line_reason,
            )
            logger.info("=== Continuing with remaining posts ===")
            first = False

        time.sleep(LLM_SLEEP_SECONDS)

    db.sync()  # push classifications to Turso
    logger.info("classification done: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(classify_pending())
