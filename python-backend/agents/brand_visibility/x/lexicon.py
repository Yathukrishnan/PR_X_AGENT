"""
Lexicon loader. Primary source: Turso keywords table. Fallback: genesis_lexicon.json.

The scraper expects a JSON shape with:
  - Keyword_Classes: dict of class_key -> {name, priority, queries: [query_str, ...]}
  - Tracked_Handles: dict of tier_n -> [handle, ...]

This module rebuilds that shape from Turso rows respecting enabled=1.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from shared.config.settings import LEXICON_PATH

logger = logging.getLogger(__name__)

# Operator suffix added to every chunk. Matches existing genesis_lexicon format.
OPERATOR_SUFFIX_DEFAULT = " -is:retweet -is:reply -is:nullcast lang:en"
OPERATOR_SUFFIX_NO_LANG = " -is:retweet -is:reply -is:nullcast"

# Class E (multilingual) does not pin lang:en
CLASSES_WITHOUT_LANG = {"E"}

# Cap each OR'd chunk to stay under X's 500-char query limit
MAX_KEYWORDS_PER_CHUNK = 18


def _chunk_keywords(keywords: list[str], class_key: str) -> list[str]:
    """Group keywords into OR'd query strings with operator suffix."""
    suffix = OPERATOR_SUFFIX_NO_LANG if class_key in CLASSES_WITHOUT_LANG else OPERATOR_SUFFIX_DEFAULT
    chunks: list[str] = []
    for i in range(0, len(keywords), MAX_KEYWORDS_PER_CHUNK):
        batch = keywords[i:i + MAX_KEYWORDS_PER_CHUNK]
        quoted = [f'"{kw}"' for kw in batch]
        query = "(" + " OR ".join(quoted) + ")" + suffix
        chunks.append(query)
    return chunks


def _row_get(row: Any, key: str, index: int) -> Any:
    """Get a value from a libsql row by name (preferred) or positional index."""
    try:
        return row[key]
    except (TypeError, KeyError, IndexError):
        return row[index]


def load_from_turso(db: Any) -> dict[str, Any]:
    """Build the lexicon dict from Turso. Raises if Turso is unreachable."""
    db.sync()

    class_rows = db.query(
        "SELECT class_key, name, priority FROM keyword_classes "
        "WHERE enabled = 1 ORDER BY display_order"
    )

    classes_out: dict[str, dict] = {}
    for row in class_rows:
        class_key = _row_get(row, "class_key", 0)
        name = _row_get(row, "name", 1)
        priority = _row_get(row, "priority", 2)

        kw_rows = db.query(
            "SELECT keyword FROM keywords "
            "WHERE class_key = ? AND enabled = 1 "
            "ORDER BY keyword",
            (class_key,),
        )
        keywords = [_row_get(r, "keyword", 0) for r in kw_rows]

        if not keywords:
            logger.info("class %s has no enabled keywords — skipping", class_key)
            continue

        classes_out[class_key] = {
            "name": name,
            "priority": priority or "STANDARD",
            "queries": _chunk_keywords(keywords, class_key),
        }

    inf_rows = db.query(
        "SELECT handle, follower_tier FROM influencers WHERE enabled = 1"
    )

    handles_out: dict[str, list[str]] = {"tier_1": [], "tier_2": [], "tier_3": []}
    for row in inf_rows:
        handle = _row_get(row, "handle", 0)
        tier = _row_get(row, "follower_tier", 1)
        bucket = tier if tier in handles_out else "tier_2"
        handles_out[bucket].append(handle)

    return {
        "version": "turso-live",
        "source": "turso",
        "Keyword_Classes": classes_out,
        "Tracked_Handles": handles_out,
    }


def _resolve_lexicon_path() -> Path:
    """Lexicon file path: KA017_LEXICON_FILE env override, else the default."""
    override = os.environ.get("KA017_LEXICON_FILE", "").strip()
    if override:
        return Path(override)
    return LEXICON_PATH


def load_from_file() -> dict[str, Any]:
    """Fallback: read the static JSON file (honors KA017_LEXICON_FILE override)."""
    return json.loads(_resolve_lexicon_path().read_text(encoding="utf-8"))


def load_lexicon() -> dict[str, Any]:
    """Load the raw lexicon JSON from disk, honoring KA017_LEXICON_FILE.

    No Turso, no normalization — returns the file's JSON as-is (including its
    'version' field). Used for file-based overrides and tests.
    """
    return load_from_file()


def load(db: Any) -> dict[str, Any]:
    """
    Primary: Turso. Fallback: genesis_lexicon.json.
    Override: if env var USE_V2_LEXICON is set (truthy), load directly from
    LEXICON_PATH and bypass Turso. Used for v2 lexicon pilot.
    Logs which source was used. Never raises.
    """
    import os
    if os.environ.get("KA017_LEXICON_FILE", "").strip():
        try:
            lex = load_from_file()
            lex["source"] = "file-override"
            total_queries = sum(len(c["queries"]) for c in lex["Keyword_Classes"].values())
            logger.info(
                "lexicon loaded from KA017_LEXICON_FILE override: %d classes, %d query chunks, path=%s",
                len(lex["Keyword_Classes"]), total_queries, _resolve_lexicon_path(),
            )
            return lex
        except Exception as exc:
            logger.error(
                "KA017_LEXICON_FILE set but file load failed (%s) — falling through", exc
            )
    if os.environ.get("USE_V2_LEXICON"):
        try:
            lex = load_from_file()
            lex["source"] = "v2-file-override"
            total_queries = sum(len(c["queries"]) for c in lex["Keyword_Classes"].values())
            total_handles = sum(len(v) for v in lex.get("Tracked_Handles", {}).values())
            logger.info(
                "lexicon loaded from V2 file (USE_V2_LEXICON set): %d classes, %d query chunks, %d handles, path=%s",
                len(lex["Keyword_Classes"]), total_queries, total_handles, LEXICON_PATH,
            )
            return lex
        except Exception as exc:
            logger.error(
                "USE_V2_LEXICON set but file load failed (%s) — falling through to Turso", exc
            )
            # Fall through to Turso path on failure (safer than crashing)

    try:
        lex = load_from_turso(db)
        total_queries = sum(len(c["queries"]) for c in lex["Keyword_Classes"].values())
        total_handles = sum(len(v) for v in lex["Tracked_Handles"].values())
        logger.info(
            "lexicon loaded from Turso: %d classes, %d query chunks, %d handles",
            len(lex["Keyword_Classes"]), total_queries, total_handles,
        )
        return lex
    except Exception as exc:
        logger.warning(
            "Turso lexicon load failed (%s) — falling back to %s",
            exc, LEXICON_PATH,
        )
        lex = load_from_file()
        lex["source"] = "file-fallback"
        return lex
