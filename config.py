"""
config.py
---------
Centralized configuration. Everything sensitive is pulled from environment
variables so the exact same code runs locally (.env file) and on a cloud
worker (Render/PythonAnywhere secrets manager) without any code changes.
"""

import os

# ---- Optional: load .env for local development only -----------------------
# On Render/PythonAnywhere you will set real environment variables in their
# dashboard, so python-dotenv simply does nothing there (no .env file present).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _require(name: str, default: str | None = None) -> str:
    """Fetch an env var, raising a clear error early if a critical one is missing."""
    value = os.getenv(name, default)
    if value is None:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Set it in your cloud platform's secrets/env settings."
        )
    return value


# ---- Gemini (Google Generative AI) -----------------------------------------
GEMINI_API_KEY = _require("GEMINI_API_KEY")

# Model fallback chain, tried in order. Google periodically deprecates and
# shuts down old model IDs (e.g. all gemini-1.5-* models, and gemini-2.0-flash
# / gemini-2.0-flash-lite, are already shut down and return 404 as of this
# writing) — ai_agent.py automatically falls through to the next entry here
# if one model 404s or otherwise fails, so a single deprecation can't take
# the whole bot down again.
#
# Override with GEMINI_MODELS as a comma-separated list if you want to
# change this without a code change, e.g.:
#   GEMINI_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite
#
# GEMINI_MODEL (singular) is still honored for backward compatibility with
# existing deployments — if set, it's used as the sole/first entry.
_env_models_list = os.getenv("GEMINI_MODELS")
_env_model_single = os.getenv("GEMINI_MODEL")

if _env_models_list:
    GEMINI_MODELS = [m.strip() for m in _env_models_list.split(",") if m.strip()]
elif _env_model_single:
    GEMINI_MODELS = [_env_model_single.strip()]
else:
    GEMINI_MODELS = [
        "gemini-2.5-flash",       # primary: stable, well-established, cheap
        "gemini-2.5-flash-lite",  # fallback 1: even cheaper/faster, same generation
        "gemini-3.5-flash",       # fallback 2: newer generation
        "gemini-3.5-flash-lite",  # fallback 3: newer generation, lite tier
    ]

# Kept as an alias for any code that still references the singular name.
GEMINI_MODEL = GEMINI_MODELS[0]

# ---- Telegram ---------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _require("TELEGRAM_CHAT_ID")

# ---- Mostaql scraping ---------------------------------------------------------
MOSTAQL_PROJECTS_URL = os.getenv(
    "MOSTAQL_PROJECTS_URL", "https://mostaql.com/projects"
)
# Comma-separated category filters, e.g. "python,data-science,machine-learning"
MOSTAQL_CATEGORIES = os.getenv("MOSTAQL_CATEGORIES", "")

# ---- Matching / scoring -------------------------------------------------------
MY_SKILLS = [
    "Python",
    "Data Science",
    "Machine Learning",
    "Embedded Systems",
    "C",
    "C++",
    "PowerPoint Presentation Design",
    "C#",
    "Flutter",
    "Object-Oriented Programming (OOP)",
    "MATLAB",
    "Document Formatting",
]
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "60"))

# ---- Loop timing (seconds) ----------------------------------------------------
POLL_INTERVAL_MIN = int(os.getenv("POLL_INTERVAL_MIN", "300"))   # 5 min
POLL_INTERVAL_MAX = int(os.getenv("POLL_INTERVAL_MAX", "600"))   # 10 min

# ---- Persistence (avoid re-processing the same project) -----------------------
SEEN_PROJECTS_FILE = os.getenv("SEEN_PROJECTS_FILE", "seen_projects.json")

# ---- Networking / anti-ban -----------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))       # requests (scraper/telegram)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ---- Gemini call timeout (seconds) ---------------------------------------------
# The google-generativeai SDK does NOT time out by default — this is the #1
# cause of a worker silently hanging forever with no error and no logs.
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))

# ---- Watchdog: max seconds a single monitor->evaluate->notify cycle may take ---
# If a cycle exceeds this, the main loop abandons it and moves on instead of
# hanging forever. Should comfortably exceed (new_projects * per-project time).
CYCLE_TIMEOUT = int(os.getenv("CYCLE_TIMEOUT", "600"))  # 10 min

# ---- Dummy HTTP server (Render Web Service requires a bound port) --------------
PORT = int(os.getenv("PORT", "10000"))
