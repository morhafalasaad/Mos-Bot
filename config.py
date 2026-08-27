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


# ---- Logging ----------------------------------------------------------------
# Controls verbosity for the whole app (see main.py's logging.basicConfig).
# INFO (default) shows only meaningful state changes: new projects found,
# matches, notifications sent, errors. DEBUG additionally shows internal
# per-request/per-attempt chatter (raw HTTP statuses, rate-limit bookkeeping,
# etc.) — turn it on only when actively troubleshooting.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()


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

def _clean_key(k: str) -> str:
    """Strips whitespace and stray surrounding quote characters — a common
    mistake when pasting comma-separated values into a cloud platform's env
    var UI (e.g. GEMINI_API_KEYS="key_one","key_two" instead of
    GEMINI_API_KEYS=key_one,key_two), which would otherwise silently
    produce keys with literal quote characters baked in and every request
    failing authentication rather than hitting quota."""
    return k.strip().strip('"').strip("'").strip()


if _env_keys_list:
    GEMINI_API_KEYS = [_clean_key(k) for k in _env_keys_list.split(",") if _clean_key(k)]
elif _env_key_single:
    GEMINI_API_KEYS = [_clean_key(_env_key_single)]
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

# ---- Gemini-only proxy (split tunneling) -----------------------------------
# Optional. If your local IP is in a region the Gemini API rejects
# ("400 FAILED_PRECONDITION: User location is not supported"), set this to
# a proxy URL and ONLY the Gemini client's traffic is routed through it —
# e.g. http://user:pass@host:port or socks5://user:pass@host:port (SOCKS
# needs `pip install "httpx[socks]"`). Everything else in this process
# (Mostaql scraping via scraper.py/cloudscraper, Telegram, GitHub) keeps
# using your normal local connection directly, unaffected by this setting —
# routing THOSE through the same proxy is what gets cloudscraper's requests
# to Mostaql flagged and 403'd as datacenter/VPN traffic by Cloudflare.
# Leave unset if Gemini already works from your location.
GEMINI_PROXY_URL = os.getenv("GEMINI_PROXY_URL", "").strip() or None

# ---- Token-usage optimization ---------------------------------------------------
# Applied ONLY to the scoring call (ai_agent.score_project) — its output is
# a small fixed-shape JSON object, so a low temperature and a tight output
# cap are safe. draft_proposal() deliberately does NOT use these: it needs
# natural, varied prose (per the "human, non-robotic tone" requirement), and
# a temperature this low would make every proposal read identically.
GEMINI_SCORING_TEMPERATURE = float(os.getenv("GEMINI_SCORING_TEMPERATURE", "0.1"))
GEMINI_SCORING_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_SCORING_MAX_OUTPUT_TOKENS", "300"))
# Project descriptions longer than this are truncated before being placed
# into ANY prompt (both scoring and proposal drafting) — see
# ai_agent.smart_truncate_description. Title and tags are never touched.
GEMINI_DESCRIPTION_MAX_CHARS = int(os.getenv("GEMINI_DESCRIPTION_MAX_CHARS", "1400"))

# ---- Silent token-usage analytics ------------------------------------------------
# Local JSON file (see ai_agent.TokenUsageTracker) — unlike the pending-
# project queue elsewhere in this codebase, this is NOT required to be
# GitHub-hosted: it's non-critical analytics, and losing history on a
# Render restart (local disk is ephemeral there) has no functional impact,
# unlike losing a project that still needs to be evaluated.
TOKEN_USAGE_STATS_FILE = os.getenv("TOKEN_USAGE_STATS_FILE", "token_usage_stats.json")

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

# ---- Transient-error backoff cap (tenacity) -------------------------------------
# GEMINI_QUOTA_BACKOFF_MAX caps tenacity's exponential backoff for
# TRANSIENT errors only (504/DEADLINE_EXCEEDED/503/500) — see
# ai_agent._call_gemini_once's @retry decorator. As of this revision, 429
# quota errors are NEVER retried locally at all (no same-key wait, no
# backoff) — a 429 immediately moves to the next (key, model) pair in the
# fallback chain (see ai_agent._generate), which is the fastest way to
# actually get an answer during a burst backlog rather than waiting out a
# per-minute window that may not have reset yet.
GEMINI_QUOTA_BACKOFF_MAX = float(os.getenv("GEMINI_QUOTA_BACKOFF_MAX", "20"))    # seconds cap

# ---- Proactive local rate limiter (avoid triggering 429s in the first place) ---
# Hard cap of requests per key per rolling 60s window, enforced client-side
# BEFORE a request is sent — free tier is 15 RPM, so 14 leaves a safety
# margin for clock drift between our tracker and Google's. See
# ai_agent.KeyRateLimiter. When every configured key is at this cap,
# _generate() raises AllKeysRateLimited immediately (zero API calls made)
# instead of firing a request likely to be rejected anyway.
GEMINI_MAX_RPM_PER_KEY = int(os.getenv("GEMINI_MAX_RPM_PER_KEY", "14"))

# ---- Multi-model fallback cascade -----------------------------------------------
# Tried in order, highest-RPM first, falling back to the next entry on
# rate limits, timeouts, or API errors that survive the current (key,
# model) pair. See ai_agent._generate(). Override via a comma-separated
# GEMINI_MODEL_CASCADE env var if your available models/tiers differ.
GEMINI_MODEL_CASCADE = [
    m.strip() for m in os.getenv(
        "GEMINI_MODEL_CASCADE",
        "gemini-3.5-flash-lite,gemini-2.5-flash-lite,gemini-3.5-flash,gemini-2.5-flash",
    ).split(",") if m.strip()
]

# Per-model RPM caps for the LOCAL proactive rate limiter, as specified.
# Not independently re-verified against Google's live quota pages as of
# this writing — confirm against your actual quota tier
# (https://aistudio.google.com/app/apikey or your Cloud Console quota
# page) and override via the env vars below; quotas change over time and
# by billing tier. Any model in GEMINI_MODEL_CASCADE not listed here falls
# back to GEMINI_MAX_RPM_PER_KEY.
MODEL_RPM_LIMITS = {
    "gemini-3.5-flash-lite": int(os.getenv("GEMINI_RPM_FLASH_LITE_35", "15")),
    "gemini-2.5-flash-lite": int(os.getenv("GEMINI_RPM_FLASH_LITE_25", "10")),
    "gemini-3.5-flash": int(os.getenv("GEMINI_RPM_FLASH_35", "5")),
    "gemini-2.5-flash": int(os.getenv("GEMINI_RPM_FLASH_25", "5")),
}

# Light throttling between consecutive (key, model) attempts within a
# single _generate() call, so a burst of backlog items being drained in
# quick succession doesn't itself trip RPM limits — deliberately small
# (a fraction of a Gemini call's own latency), not a rate-limit recovery
# wait. Not applied before the very first attempt.
GEMINI_INTER_REQUEST_DELAY = float(os.getenv("GEMINI_INTER_REQUEST_DELAY", "1.0"))

# ---- Watchdog: max seconds a SINGLE project evaluation task may take -----------
# Env var name kept as CYCLE_TIMEOUT for backward compatibility with
# existing deployments, but its meaning changed with the producer/consumer
# architecture: it used to bound one whole scrape+evaluate "cycle"; now
# main.py's consumer thread enforces it per INDIVIDUAL project evaluation
# (see process_project_with_watchdog). If a single evaluation exceeds this,
# it's abandoned and routed to the GitHub fallback so the consumer isn't
# stalled — the default is generous but per-task, not per-batch.
CYCLE_TIMEOUT = int(os.getenv("CYCLE_TIMEOUT", "600"))  # 10 min

# ---- Producer/Consumer queue tuning ---------------------------------------------
# Max items the producer can have queued for the consumer before put()
# starts blocking (mild backpressure) — plenty of headroom for Mostaql's
# realistic project volume.
TASK_QUEUE_MAXSIZE = int(os.getenv("TASK_QUEUE_MAXSIZE", "200"))
# How often (seconds) the consumer re-checks the GitHub-hosted retry
# queue/issues for projects to re-evaluate, independent of how often new
# projects are being pulled off task_queue.
GITHUB_RETRY_CHECK_INTERVAL = int(os.getenv("GITHUB_RETRY_CHECK_INTERVAL", "300"))  # 5 min

# ---- Dummy HTTP server (Render Web Service requires a bound port) --------------
PORT = int(os.getenv("PORT", "10000"))

# ---- GitHub fallback (used when the Gemini AI call itself fails, e.g. every ---
# key in GEMINI_API_KEYS hit 429 RESOURCE_EXHAUSTED) --------------------------
# By design, no local storage is used for this — raw project data is sent
# straight to GitHub (as an Issue or an uploaded Markdown file) so nothing
# is silently lost and nothing is written to local/ephemeral disk. See
# github_fallback.py. Optional feature: only active if both GITHUB_TOKEN and
# GITHUB_REPO are set; otherwise main.py just logs a warning and moves on.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")            # a fine-grained PAT with Issues/Contents write access
GITHUB_REPO = os.getenv("GITHUB_REPO", "")          # "owner/repo"
GITHUB_FALLBACK_MODE = os.getenv("GITHUB_FALLBACK_MODE", "issue").strip().lower()  # "issue" or "file"
GITHUB_FALLBACK_BRANCH = os.getenv("GITHUB_FALLBACK_BRANCH", "main")
GITHUB_FALLBACK_DIR = os.getenv("GITHUB_FALLBACK_DIR", "unevaluated_projects")
GITHUB_API_TIMEOUT = int(os.getenv("GITHUB_API_TIMEOUT", "20"))
GITHUB_FALLBACK_ENABLED = bool(GITHUB_TOKEN and GITHUB_REPO)

# Structured retry queue: separate from the human-readable Issue/file
# record above. This JSON file in the repo is what main.py actually reads
# back from and re-evaluates each cycle once Gemini quota/rate limits
# allow — see github_fallback.load_pending_queue() / queue_project().
GITHUB_QUEUE_FILE = os.getenv("GITHUB_QUEUE_FILE", "pending_projects.json")
# Safety cap so a project that fails for a NON-quota reason (e.g. a
# malformed description that breaks something every time) doesn't sit in
# the queue being retried forever. After this many failed re-evaluation
# attempts, it's dropped from the auto-retry queue (still preserved in the
# human-readable Issue/file record from when it was first queued).
GITHUB_QUEUE_MAX_RETRIES = int(os.getenv("GITHUB_QUEUE_MAX_RETRIES", "20"))

# ---- Token-usage stats sync (analytics — separate from the fallback queue above) ---
# Note: deliberately a SEPARATE env var from GITHUB_FALLBACK_BRANCH — the
# token-stats file can be synced to a different branch than the pending-
# project queue/issues if you want (e.g. an "analytics" branch), though
# both default to "main". GITHUB_TOKEN/GITHUB_REPO above are reused as-is.
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
# How often (seconds) main.py's consumer syncs the local token_usage_stats.json
# up to GitHub — shares the same "once per backlog check" cadence as the
# GitHub retry-queue check by default, but independently tunable.
TOKEN_STATS_SYNC_INTERVAL = int(os.getenv("TOKEN_STATS_SYNC_INTERVAL", "300"))  # 5 min
