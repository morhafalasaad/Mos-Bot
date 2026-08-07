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


# ---- Gemini (Google Gen AI) -------------------------------------------------
# Multiple API keys for rotation on quota exhaustion (429 RESOURCE_EXHAUSTED).
# ai_agent.py automatically rotates to the next key in this list when the
# active one hits its quota, so a single free-tier key limit doesn't stall
# the bot. Set as a comma-separated list:
#   GEMINI_API_KEYS=key_one,key_two,key_three
# GEMINI_API_KEY (singular) is still honored for backward compatibility —
# if only that's set, it's used as the sole key (no rotation available).
_env_keys_list = os.getenv("GEMINI_API_KEYS")
_env_key_single = os.getenv("GEMINI_API_KEY")

if _env_keys_list:
    GEMINI_API_KEYS = [k.strip() for k in _env_keys_list.split(",") if k.strip()]
elif _env_key_single:
    GEMINI_API_KEYS = [_env_key_single.strip()]
else:
    raise EnvironmentError(
        "No Gemini API key configured. Set GEMINI_API_KEYS (comma-separated, "
        "for rotation) or at least GEMINI_API_KEY (single key) in your cloud "
        "platform's secrets/env settings."
    )

# Kept as an alias for any code that still references the singular name.
GEMINI_API_KEY = GEMINI_API_KEYS[0]

# Single stable model, used directly with no fallback chain (by request).
# Override via the GEMINI_MODEL env var if Google renames/retires this
# model later — no code change needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# ---- Telegram ---------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _require("TELEGRAM_CHAT_ID")

# ---- Mostaql scraping ---------------------------------------------------------
MOSTAQL_PROJECTS_URL = os.getenv(
    "MOSTAQL_PROJECTS_URL", "https://mostaql.com/projects"
)
# Comma-separated category filters, e.g. "python,data-science,machine-learning"
MOSTAQL_CATEGORIES = os.getenv("MOSTAQL_CATEGORIES", "")

# Fetch each NEW project's own detail page to read its official required-
# skill tags ("المهارات المطلوبة"), used for local pre-filtering before any
# Gemini call is made (see ai_agent.local_skill_prefilter). This is an EXTRA
# Mostaql HTTP request per newly-seen project (not per Gemini call) — the
# trade is one cheap Mostaql request to potentially skip one Gemini API
# call entirely. Disable if this causes extra bot-detection friction; the
# pre-filter fails OPEN (never blocks) when tags are unavailable, so
# disabling this only means "back to evaluating every new project."
FETCH_PROJECT_TAGS = os.getenv("FETCH_PROJECT_TAGS", "true").strip().lower() in ("1", "true", "yes")

# Client warning system: a project is NEVER skipped/filtered based on the
# client's profile — a rating below this is just appended as a "⚠️" note in
# the Telegram message so you can decide with full information. See
# scraper.build_client_warning / parse_client_info.
LOW_CLIENT_RATING_THRESHOLD = float(os.getenv("LOW_CLIENT_RATING_THRESHOLD", "3.5"))

# ---- Matching / scoring -------------------------------------------------------
# Mostaql project tags are frequently in Arabic (e.g. "علم البيانات" rather
# than "Data Science"), so each skill below has an Arabic entry alongside
# its English one — the local pre-filter matches whichever one appears in
# the project's actual tags. Add more Arabic synonyms here if you notice
# real projects being filtered out that shouldn't be (check the
# "Fetched N tag(s) for project ..." log line in scraper.py to see what
# Mostaql is actually tagging things with).
MY_SKILLS = [
    "Python", "بايثون",
    "Data Science", "علم البيانات",
    "Machine Learning", "تعلم الآلة", "تعلم الالة",
    "Embedded Systems", "الأنظمة المدمجة", "أنظمة مدمجة",
    "C", "C++", "C#",
    "PowerPoint Presentation Design", "تصميم عروض بوربوينت", "بوربوينت",
    "Flutter", "فلاتر",
    "Object-Oriented Programming (OOP)", "البرمجة الكائنية", "البرمجة الشيئية",
    "MATLAB", "ماتلاب",
    "Document Formatting", "تنسيق المستندات", "تنسيق مستندات",
]
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "60"))

# ---- Loop timing (seconds) ----------------------------------------------------
# Fast polling (5-10 min) so new projects are caught close to real-time.
# NOTE: this was previously widened to 45-60 min specifically to fit inside
# Gemini's free-tier quota (20 requests/day for gemini-3.5-flash) — at this
# faster interval you WILL hit 429 RESOURCE_EXHAUSTED again once the daily
# quota is used up (likely within a couple of hours, same as before). That's
# not a bug: ai_agent.py's 429 handling (config.GEMINI_API_KEYS rotation,
# and the local tag pre-filter that skips irrelevant projects at zero API
# cost) will keep the bot from crashing when that happens, but scoring will
# simply stop working — reasoning="AI scoring unavailable (error)" — for the
# rest of the day until the quota resets. If real-time checking matters more
# than exhaustive scoring, consider adding more entries to GEMINI_API_KEYS
# (each free key gets its own 20/day budget) or a paid Gemini tier.
# Override with POLL_INTERVAL_MIN/MAX (seconds) as always.
POLL_INTERVAL_MIN = int(os.getenv("POLL_INTERVAL_MIN", "300"))   # 5 min
POLL_INTERVAL_MAX = int(os.getenv("POLL_INTERVAL_MAX", "420"))   # 7 min

# ---- Persistence (avoid re-processing the same project) -----------------------
SEEN_PROJECTS_FILE = os.getenv("SEEN_PROJECTS_FILE", "seen_projects.json")

# ---- Networking / anti-ban -----------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))       # requests (scraper/telegram)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ---- Gemini call timeout (seconds) ---------------------------------------------
# The google-generativeai SDK does NOT time out by default — this is the #1
# cause of a worker silently hanging forever with no error and no logs.
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))

# ---- Transient Gemini error retry (504 Gateway Timeout / DEADLINE_EXCEEDED / 503) ---
# Distinct from API-key rotation (which only helps with 429 quota errors):
# these are gateway/server-side hiccups where retrying the SAME key after a
# short wait is the right move. See ai_agent._generate() for the full
# retry-then-rotate logic.
GEMINI_MAX_TRANSIENT_RETRIES = int(os.getenv("GEMINI_MAX_TRANSIENT_RETRIES", "2"))
GEMINI_RETRY_BACKOFF_BASE = float(os.getenv("GEMINI_RETRY_BACKOFF_BASE", "2"))  # seconds

# ---- Watchdog: max seconds a single monitor->evaluate->notify cycle may take ---
# If a cycle exceeds this, the main loop abandons it and moves on instead of
# hanging forever. Should comfortably exceed (new_projects * per-project time).
CYCLE_TIMEOUT = int(os.getenv("CYCLE_TIMEOUT", "600"))  # 10 min

# ---- Dummy HTTP server (Render Web Service requires a bound port) --------------
PORT = int(os.getenv("PORT", "10000"))
