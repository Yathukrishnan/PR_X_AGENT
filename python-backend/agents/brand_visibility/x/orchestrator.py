"""
KA017 orchestrator — Interactive Control Panel & Scheduled Loop

Usage:
  python orchestrator.py                 # Opens interactive menu
  python orchestrator.py --once          # run one full tick and exit
  python orchestrator.py --loop          # run forever (30-min ticks)
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from shared.config.settings import DATA_DIR, LOG_PATH

# ---------------------------------------------------------------------------
# Logging — structured, rotating file + stdout
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(stdout_handler)

    DATA_DIR.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(LOG_PATH), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)

TICK_STATE_FILE = DATA_DIR / "tick_state.json"
TICK_INTERVAL_SECONDS = 30 * 60  # 30 min
CONTENT_DRAFT_EVERY_N_TICKS = 48  # ~once per day


# ---------------------------------------------------------------------------
# Tick state persistence
# ---------------------------------------------------------------------------

def load_tick_state() -> int:
    if TICK_STATE_FILE.exists():
        try:
            return json.loads(TICK_STATE_FILE.read_text())["tick_number"]
        except Exception:
            pass
    return 0


def save_tick_state(tick_number: int) -> None:
    TICK_STATE_FILE.write_text(
        json.dumps({"tick_number": tick_number, "saved_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tick logic
# ---------------------------------------------------------------------------

def tick(tick_number: int, mode: str = "all", dry_run: bool = False) -> dict:
    from agents.brand_visibility.x.db import Database
    from agents.brand_visibility.x.x_scraper import expand_reply_trees, sweep_influencers, sweep_keywords
    from agents.brand_visibility.x.post_drafter import cluster_themes, draft_posts_for_new_themes
    from agents.brand_visibility.x.classifier import classify_pending

    start = datetime.now(timezone.utc)
    logger.info("=== Tick %d start (%s) dry_run=%s mode=%s ===", tick_number, start.isoformat(), dry_run, mode)

    from shared.config.settings import MAX_API_CALLS_PER_RUN
    from agents.brand_visibility.x.x_scraper import reset_call_budget
    db = Database()
    reset_call_budget(MAX_API_CALLS_PER_RUN)
    run_id = db.start_run(mode=mode, triggered_by="orchestrator")
    db.log_activity(run_id, phase="orchestrator", event="tick_start",
                    message=f"tick={tick_number} dry_run={dry_run} budget={MAX_API_CALLS_PER_RUN}")

    summary: dict[str, object] = {"tick": tick_number, "start": start.isoformat()}
    total_calls = 0
    total_new = 0

    try:
        if mode in ("all", "keywords"):
            try:
                stats = sweep_keywords(db, dry_run=dry_run, run_id=run_id)
                summary["keywords"] = stats
                total_calls += stats.get("api_calls", 0)
                total_new += stats.get("new", 0)
            except Exception:
                logger.exception("sweep_keywords failed")

        if mode in ("all", "influencers"):
            try:
                stats = sweep_influencers(db, tick_number=tick_number, dry_run=dry_run, run_id=run_id)
                summary["influencers"] = stats
                total_calls += stats.get("api_calls", 0)
                total_new += stats.get("new", 0)
            except Exception:
                logger.exception("sweep_influencers failed")

        if mode in ("all", "classify"):
            try:
                stats = classify_pending(db, limit=50, dry_run=dry_run)
                summary["classify"] = stats
            except Exception:
                logger.exception("classify_pending failed")

        if mode in ("all", "reply_trees"):
            try:
                stats = expand_reply_trees(db, dry_run=dry_run, run_id=run_id)
                summary["reply_trees"] = stats
                total_calls += stats.get("api_calls", 0)
                total_new += stats.get("reply_tweets", 0)
            except Exception:
                logger.exception("expand_reply_trees failed")

        if mode in ("all", "cluster", "draft") and (
            tick_number % CONTENT_DRAFT_EVERY_N_TICKS == 0 or mode in ("cluster", "draft")
        ):
            if mode in ("all", "cluster"):
                try:
                    themes = cluster_themes(db)
                    summary["themes_clustered"] = len(themes)
                except Exception:
                    logger.exception("cluster_themes failed")

            if mode in ("all", "draft"):
                try:
                    stats = draft_posts_for_new_themes(db, dry_run=dry_run)
                    summary["drafts"] = stats
                except Exception:
                    logger.exception("draft_posts_for_new_themes failed")

        end = datetime.now(timezone.utc)
        duration = (end - start).total_seconds()
        summary["end"] = end.isoformat()
        summary["duration_seconds"] = round(duration, 1)

        db.finish_run(run_id, status="completed",
                      calls_used=total_calls,
                      records_new=total_new,
                      summary=summary)

    except Exception as exc:
        end = datetime.now(timezone.utc)
        summary["end"] = end.isoformat()
        summary["duration_seconds"] = round((end - start).total_seconds(), 1)
        db.finish_run(run_id, status="failed",
                      calls_used=total_calls,
                      records_new=total_new,
                      error_message=str(exc),
                      summary=summary)
        raise
    finally:
        db.log_activity(run_id, phase="orchestrator", event="tick_end")

    logger.info("=== Tick %d done in %.1f s | %s ===",
                tick_number, summary["duration_seconds"], summary)
    return summary


# ---------------------------------------------------------------------------
# CLI & Interactive Menu
# ---------------------------------------------------------------------------

def interactive_menu(tick_number: int, dry_run: bool) -> None:
    """Displays an interactive control panel for manual orchestration."""
    while True:
        print("\n" + "="*55)
        print(" 🚀 KA017 Orchestrator - Interactive Control Panel")
        print("="*55)
        print(" [1] Sweep Keywords        (Scraper Only)")
        print(" [2] Sweep Influencers     (Scraper Only)")
        print(" [3] Expand Reply Trees    (Scraper Only)")
        print(" [4] Classify Pending      (Analysis)")
        print(" [5] Cluster Themes        (Analysis)")
        print(" [6] Draft Posts           (Auto-Reply/Content Engine)")
        print(" [7] Run Full Pipeline     (All Modes)")
        print(" [8] Start Continuous Loop (Automated)")
        print(" [9] Exit")
        print("="*55)
        
        choice = input("\nSelect a phase to trigger [1-9]: ").strip()

        mode_map = {
            "1": "keywords",
            "2": "influencers",
            "3": "reply_trees",
            "4": "classify",
            "5": "cluster",
            "6": "draft",
            "7": "all"
        }

        if choice == "9":
            print("Exiting KA017 Orchestrator. Goodbye.")
            sys.exit(0)
            
        elif choice == "8":
            print("\nStarting continuous loop... (Press Ctrl+C to stop)")
            run_continuous_loop(tick_number, dry_run)
            break
            
        elif choice in mode_map:
            selected_mode = mode_map[choice]
            print(f"\n⚡ Triggering Phase: {selected_mode.upper()}...")
            try:
                tick(tick_number, mode=selected_mode, dry_run=dry_run)
                tick_number += 1
                save_tick_state(tick_number)
            except Exception as e:
                logger.exception(f"Manual trigger for {selected_mode} failed.")
        else:
            print("Invalid selection. Please choose a number between 1 and 9.")

def run_continuous_loop(tick_number: int, dry_run: bool) -> None:
    logger.info("Starting continuous loop (tick interval: %d min)", TICK_INTERVAL_SECONDS // 60)
    while True:
        try:
            tick(tick_number, mode="all", dry_run=dry_run)
        except KeyboardInterrupt:
            logger.info("Continuous loop halted by user.")
            break
        except Exception:
            logger.exception("Tick %d failed — sleeping before retry", tick_number)

        tick_number += 1
        save_tick_state(tick_number)
        logger.info("Sleeping %d s until next tick…", TICK_INTERVAL_SECONDS)
        try:
            time.sleep(TICK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Continuous loop halted by user during sleep.")
            break

def main() -> None:
    parser = argparse.ArgumentParser(description="KA017 orchestrator")
    parser.add_argument("--once", action="store_true", help="Run one full tick and exit without menu")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop without menu")
    parser.add_argument(
        "--mode",
        choices=["keywords", "influencers", "reply_trees", "classify", "cluster", "draft", "all"],
        default="all",
        help="Specify pipeline step (used with --once)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, make no API calls")
    parser.add_argument(
        "--sweep-type", choices=["Latest", "Top"], default="Latest",
        help="Sweep type for keyword search (twitter241 'type' param). Latest = chronological, Top = algorithmic.",
    )
    parser.add_argument(
        "--max-pages", type=int, default=1,
        help="Pages per query (1 page = ~20 tweets). Higher = deeper capture at proportional API cost.",
    )
    parser.add_argument(
        "--classes", type=str, default="",
        help="Comma-separated class codes to sweep (e.g. 'A,C'). Empty = all classes (default).",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        help="Hour-granular time filter for sweeps. If set, replaces {{since_time}} "
             "placeholders in lexicon queries with current_unix_time - N*3600. "
             "If unset, placeholders are stripped from queries.",
    )
    parser.add_argument(
        "--lexicon-file",
        type=str,
        default=None,
        help="Override the default lexicon path. Useful for testing with a "
             "temporary lexicon (e.g. config/genesis_lexicon_topsweep.json).",
    )
    args = parser.parse_args()

    # Pass sweep config to scraper via env vars (read by x_scraper at provider-call time)
    os.environ["KA017_SWEEP_TYPE"] = args.sweep_type
    os.environ["KA017_MAX_PAGES"] = str(args.max_pages)
    os.environ["KA017_CLASS_FILTER"] = args.classes
    if args.since_hours is not None:
        os.environ["KA017_SINCE_HOURS"] = str(args.since_hours)
    if args.lexicon_file:
        os.environ["KA017_LEXICON_FILE"] = args.lexicon_file

    _setup_logging()

    tick_number = load_tick_state()

    # Bypass menu if flags are passed directly
    if args.once:
        try:
            tick(tick_number, mode=args.mode, dry_run=args.dry_run)
        except Exception:
            logger.exception("Single tick failed")
            raise SystemExit(1)
        save_tick_state(tick_number + 1)
        return

    if args.loop:
        run_continuous_loop(tick_number, dry_run=args.dry_run)
        return

    # Default to Interactive Menu
    try:
        interactive_menu(tick_number, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\nExiting KA017 Orchestrator. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()