"""KA017 standalone scheduler.

Reads config/schedule.json and runs the orchestrator on schedule. Writes
runtime status to data/scheduler_status.json so the dashboard can detect
whether a scheduler is active and when the last/next run happened.

Run from terminal:
    python scheduler.py

Stop with Ctrl+C — cleanly removes the status file.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bootstrap so this script can be run from anywhere. The python-backend root is
# 3 levels up (agents/brand_visibility/x/scheduler.py -> parents[3]); add it to
# sys.path so output/, config/, and agents/ resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output.scheduler_manager import (
    load_schedule,
    write_status,
    clear_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scheduler")

_PYTHON = sys.executable


# ----------------------------------------------------------- scheduling ---

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _next_fire_interval(last_fire: datetime | None, interval_minutes: int) -> datetime:
    """Next fire = last_fire + interval, or now if no last fire yet."""
    if last_fire is None:
        return _now_utc() + timedelta(minutes=interval_minutes)
    return last_fire + timedelta(minutes=interval_minutes)


def _next_fire_fixed(fixed_times: list[str], now: datetime | None = None) -> datetime:
    """Next clock time matching any of the HH:MM entries (in UTC for simplicity)."""
    now = now or _now_utc()
    today = now.date()
    candidates: list[datetime] = []
    for t in fixed_times:
        h, m = (int(x) for x in t.split(":"))
        candidate = datetime(today.year, today.month, today.day, h, m, tzinfo=timezone.utc)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        candidates.append(candidate)
    return min(candidates)


# ------------------------------------------------------------ fire run ---

def _fire_run(modes: list[str]) -> str:
    """Spawn orchestrator subprocess for each requested mode, sequentially.

    Returns 'success' if all subprocesses exited 0, else 'failure'.
    """
    overall = "success"
    for mode in modes:
        logger.info("Firing orchestrator: mode=%s", mode)
        # Run the orchestrator as a module (its package imports need the repo
        # root on sys.path; -m with cwd=_REPO_ROOT provides that).
        cmd = [_PYTHON, "-m", "agents.brand_visibility.x.orchestrator", "--once", "--mode", mode]
        try:
            proc = subprocess.run(
                cmd, cwd=str(_REPO_ROOT), capture_output=False, timeout=1800,
            )
            if proc.returncode != 0:
                logger.warning("Orchestrator mode=%s exited %d", mode, proc.returncode)
                overall = "failure"
        except subprocess.TimeoutExpired:
            logger.error("Orchestrator mode=%s timed out after 30 min", mode)
            overall = "failure"
        except Exception as exc:
            logger.exception("Orchestrator mode=%s failed: %s", mode, exc)
            overall = "failure"
    return overall


# ------------------------------------------------------------- main loop ---

_shutdown = False


def _signal_handler(signum, frame):  # noqa: ARG001
    global _shutdown
    logger.info("Received signal %d, shutting down...", signum)
    _shutdown = True


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except (AttributeError, ValueError):
            pass

    pid = os.getpid()
    started_at = _now_utc()
    last_fire: datetime | None = None
    last_outcome: str | None = None

    logger.info("Scheduler starting (pid=%d)", pid)

    try:
        while not _shutdown:
            cfg = load_schedule()
            if not cfg.get("enabled", False):
                # Disabled — write status so dashboard sees us, but don't fire
                write_status({
                    "pid": pid,
                    "started_at": started_at.isoformat(),
                    "last_fire_at": last_fire.isoformat() if last_fire else None,
                    "last_fire_outcome": last_outcome,
                    "next_fire_at": None,
                    "state": "idle (schedule disabled)",
                })
                # Sleep 30s and re-check
                for _ in range(30):
                    if _shutdown:
                        break
                    time.sleep(1)
                continue

            ftype = cfg.get("frequency_type", "interval")
            if ftype == "interval":
                interval = max(5, int(cfg.get("interval_minutes", 60)))
                next_fire = _next_fire_interval(last_fire, interval)
            elif ftype == "fixed_times":
                times = cfg.get("fixed_times", [])
                next_fire = _next_fire_fixed(times)
            else:
                logger.error("Unknown frequency_type: %s", ftype)
                next_fire = _now_utc() + timedelta(minutes=60)

            write_status({
                "pid": pid,
                "started_at": started_at.isoformat(),
                "last_fire_at": last_fire.isoformat() if last_fire else None,
                "last_fire_outcome": last_outcome,
                "next_fire_at": next_fire.isoformat(),
                "state": "scheduled",
            })

            # Wait until next fire (or shutdown)
            while not _shutdown:
                if _now_utc() >= next_fire:
                    break
                time.sleep(min(30, (next_fire - _now_utc()).total_seconds()))

            if _shutdown:
                break

            # Re-load config in case it changed during the wait
            cfg = load_schedule()
            if not cfg.get("enabled", False):
                continue  # User disabled mid-wait

            modes = cfg.get("modes", ["keywords", "classify"])
            logger.info("Firing scheduled run: modes=%s", modes)
            write_status({
                "pid": pid,
                "started_at": started_at.isoformat(),
                "last_fire_at": last_fire.isoformat() if last_fire else None,
                "last_fire_outcome": last_outcome,
                "next_fire_at": next_fire.isoformat(),
                "state": f"running ({', '.join(modes)})",
            })
            last_outcome = _fire_run(modes)
            last_fire = _now_utc()
    finally:
        logger.info("Scheduler exiting, clearing status file")
        clear_status()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
