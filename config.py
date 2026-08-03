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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

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
]
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "75"))

# ---- Loop timing (seconds) ----------------------------------------------------
POLL_INTERVAL_MIN = int(os.getenv("POLL_INTERVAL_MIN", "300"))   # 5 min
POLL_INTERVAL_MAX = int(os.getenv("POLL_INTERVAL_MAX", "600"))   # 10 min

# ---- Persistence (avoid re-processing the same project) -----------------------
SEEN_PROJECTS_FILE = os.getenv("SEEN_PROJECTS_FILE", "seen_projects.json")

# ---- Networking / anti-ban -----------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
