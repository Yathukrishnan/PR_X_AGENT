from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API endpoints
X_API_BASE = "https://api.x.com/2"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Scraper provider
SCRAPER_PROVIDER: str = os.getenv("SCRAPER_PROVIDER", "twitter241")
_VALID_PROVIDERS = {"x_official", "twitter_api45", "twitter241"}
if SCRAPER_PROVIDER not in _VALID_PROVIDERS:
    raise ValueError(
        f"SCRAPER_PROVIDER must be one of {_VALID_PROVIDERS}, got: {SCRAPER_PROVIDER!r}"
    )

# Credentials
X_BEARER_TOKEN: str = os.getenv("X_BEARER_TOKEN", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST", "twitter-api45.p.rapidapi.com")

# Models
CLASSIFIER_MODEL: str = os.getenv("OPENROUTER_MODEL_CLASSIFIER", "google/gemini-2.5-flash")
DRAFTER_MODEL: str = os.getenv("OPENROUTER_MODEL_DRAFTER", "anthropic/claude-sonnet-4.5")

# Tier scheduling (every N ticks)
TIER_1_EVERY_N_TICKS = 1
TIER_2_EVERY_N_TICKS = 4
TIER_3_EVERY_N_TICKS = 24

# Reply tree caps
MAX_REPLY_TREE_PAGES = 3
MAX_REPLY_EXPANSIONS_PER_TICK = 5

# Sleep between API calls
SCRAPE_SLEEP_SECONDS = 5
LLM_SLEEP_SECONDS = 3

# Paths — settings.py now lives at shared/config/, so the python-backend root is
# two levels up (shared/config/settings.py -> parents[2]). config/ and data/
# stay at that root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "ka017_memory.db"
LEXICON_PATH = REPO_ROOT / "config" / "genesis_lexicon.json"
LOG_PATH = DATA_DIR / "ka017.log"

DATA_DIR.mkdir(exist_ok=True)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")

# Turso (libSQL) shared database — used for cross-agent config + dashboard
# Credentials live ONLY in .env — never hardcode them here.
TURSO_DATABASE_URL: str = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN: str = os.getenv("TURSO_AUTH_TOKEN", "")

# Embedded replica file (local cache, auto-syncs to Turso)
REPLICA_PATH = DATA_DIR / "ka017_replica.db"

# Sync interval for embedded replica (seconds)
TURSO_SYNC_INTERVAL = 60
MAX_API_CALLS_PER_RUN: int = int(os.getenv("MAX_API_CALLS_PER_RUN", "12"))
