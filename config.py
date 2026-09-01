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


# ---- Batch scoring (reduces Gemini calls per cycle) ------------------------
# Multiple newly-scraped projects are scored together in ONE Gemini call
# instead of one call per project — the single biggest lever for staying
# inside a free-tier daily request quota (RPD) when several new projects
# appear in the same poll cycle. Proposal drafting is NOT batched (see
# ai_agent.evaluate_projects_batch's docstring) — it stays one call per
# accepted match, which is normally the minority of any given batch.
GEMINI_SCORE_BATCH_SIZE = int(os.getenv("GEMINI_SCORE_BATCH_SIZE", "5"))
# How long (seconds) the consumer waits to accumulate a batch before
# scoring whatever it's collected so far — prevents a slow trickle of
# projects from waiting indefinitely for a full batch to form.
GEMINI_BATCH_MAX_WAIT_SECONDS = int(os.getenv("GEMINI_BATCH_MAX_WAIT_SECONDS", "15"))

# ---- Token-usage optimization ---------------------------------------------------
# Applied ONLY to the scoring call (ai_agent.score_project) — its output is
# a small fixed-shape JSON object, so a low temperature and a tight output
# cap are safe. draft_proposal() deliberately does NOT use these: it needs
# natural, varied prose (per the "human, non-robotic tone" requirement), and
# a temperature this low would make every proposal read identically.
GEMINI_SCORING_TEMPERATURE = float(os.getenv("GEMINI_SCORING_TEMPERATURE", "0.1"))
# Bumped from 300 -> 380 to accommodate the matched_skills/missing_skills
# list fields (see ai_agent.ProjectScoreSchema) on top of the original
# fixed-shape fields.
GEMINI_SCORING_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_SCORING_MAX_OUTPUT_TOKENS", "380"))
# Project descriptions longer than this are truncated before being placed
# into the PROPOSAL DRAFTING prompt — see ai_agent.smart_truncate_description.
# Title and tags are never touched. (Scoring uses its own, shorter limit —
# see GEMINI_SCORING_DESCRIPTION_MAX_CHARS below — since a coarse 0-100
# match read doesn't need the full detail that draft_proposal() genuinely
# needs later to "prove understanding" of the client's specific ask.)
GEMINI_DESCRIPTION_MAX_CHARS = int(os.getenv("GEMINI_DESCRIPTION_MAX_CHARS", "1400"))
# Truncation length used ONLY for the scoring call — applied to EVERY
# project (not just matches), so trimming this harder than the proposal
# limit above saves prompt tokens on every single scoring call, batched or
# not.
GEMINI_SCORING_DESCRIPTION_MAX_CHARS = int(os.getenv("GEMINI_SCORING_DESCRIPTION_MAX_CHARS", "600"))

# ---- Score caching ----------------------------------------------------------------
# Caches score_project()'s result keyed by a hash of (title, the FULL
# untruncated description, current MY_SKILLS) so a project re-evaluated
# with byte-for-byte identical content — most commonly the GitHub-fallback
# retry queue re-checking an entry whose earlier AI call failed, or a
# repost with unchanged text — skips a fresh Gemini call entirely.
# Changing MY_SKILLS changes the cache key, so it can never silently serve
# a score computed against an old skill list. Only the SCORING step is
# cached: proposal drafting always runs fresh whenever a (possibly cached)
# score clears MATCH_THRESHOLD, since its prose is deliberately non-
# deterministic and only runs for the minority of projects that match, so
# caching it would save little while risking a stale/repetitive proposal.
SCORE_CACHE_ENABLED = os.getenv("SCORE_CACHE_ENABLED", "true").strip().lower() == "true"
SCORE_CACHE_FILE = os.getenv("SCORE_CACHE_FILE", "score_cache.json")
# Oldest entries are evicted once the cache exceeds this many rows, so a
# long-running process can't grow the file unbounded.
SCORE_CACHE_MAX_ENTRIES = int(os.getenv("SCORE_CACHE_MAX_ENTRIES", "500"))

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
# — see MOSTAQL_CATEGORY_URL_TEMPLATE below for how these get turned into
# actual request URLs. Leave blank (default) to keep scraping the single
# unfiltered listing page at MOSTAQL_PROJECTS_URL, exactly as before this
# setting existed.
MOSTAQL_CATEGORIES = os.getenv("MOSTAQL_CATEGORIES", "")
# Template used to build one request URL per entry in MOSTAQL_CATEGORIES —
# {category} is replaced with each slug. The query-param form below
# (?category=<slug>) is a REASONABLE GUESS at Mostaql's filtering scheme,
# NOT verified against the live site (this codebase has no network access
# to Mostaql at development time) — same caveat as scraper.py's SELECTORS
# dict: open mostaql.com/projects, apply a category filter through the
# site's own UI, and copy the resulting URL's actual pattern here if it
# differs (e.g. a path segment like "/projects/category/<slug>" instead of
# a query param). If MOSTAQL_CATEGORIES is set but this template turns out
# to be wrong, scraper.py's existing block/anomaly detection and per-page
# logging will surface it (0 projects parsed, or a suspiciously identical
# page across different "categories") rather than failing silently.
MOSTAQL_CATEGORY_URL_TEMPLATE = os.getenv(
    "MOSTAQL_CATEGORY_URL_TEMPLATE", "https://mostaql.com/projects?category={category}"
)

# Fetch each NEW project's own detail page to read its official required-
# skill tags ("المهارات المطلوبة"), used for local pre-filtering before any
# Gemini call is made (see ai_agent.local_skill_prefilter). This is an EXTRA
# Mostaql HTTP request per newly-seen project (not per Gemini call) — the
# trade is one cheap Mostaql request to potentially skip one Gemini API
# call entirely. Disable if this causes extra bot-detection friction; the
# pre-filter fails OPEN (never blocks) when tags are unavailable, so
# disabling this only means "back to evaluating every new project."
FETCH_PROJECT_TAGS = os.getenv("FETCH_PROJECT_TAGS", "true").strip().lower() in ("1", "true", "yes")

# When a project has no official tags to check (FETCH_PROJECT_TAGS=false,
# or Mostaql simply didn't provide any), ai_agent.local_skill_prefilter()
# falls back to the same keyword-overlap check applied to the project's
# own title+description text instead of unconditionally sending every
# untagged project to Gemini. Disable if this filters out real matches
# whose descriptions don't happen to use your exact MY_SKILLS wording.
TITLE_PREFILTER_ENABLED = os.getenv("TITLE_PREFILTER_ENABLED", "true").strip().lower() == "true"

# Client warning system: a project is NEVER skipped/filtered based on the
# client's profile — a rating below this is just appended as a "⚠️" note in
# the Telegram message so you can decide with full information. See
# scraper.build_client_warning / parse_client_info.
LOW_CLIENT_RATING_THRESHOLD = float(os.getenv("LOW_CLIENT_RATING_THRESHOLD", "3.5"))

# Client-aware proposal tone: a rating at/above this (with at least a few
# reviews) lets ai_agent.draft_proposal() write with a bit more directness
# and confidence — advisory only, never mentioned in the proposal text
# itself (see draft_proposal's docstring and its prompt's explicit rule
# against referencing the client's rating/history at all).
STRONG_CLIENT_RATING_THRESHOLD = float(os.getenv("STRONG_CLIENT_RATING_THRESHOLD", "4.5"))

# ---- Outcome tracking (Telegram Won/Lost buttons) ---------------------------
# Every matched-project Telegram notification includes "✅ فاز بالمشروع" /
# "❌ لم يفز" buttons (see notifier.build_inline_keyboard). Tapping one is
# picked up by main.py's telegram_feedback_loop (long-polls Telegram's
# getUpdates) and recorded via outcome_tracker.record_outcome — see that
# module's docstring for what this data is (and isn't yet) used for.
OUTCOME_LOG_FILE = os.getenv("OUTCOME_LOG_FILE", "outcomes.json")
# Telegram's own long-poll duration (seconds) for getUpdates — the request
# itself blocks server-side for up to this long waiting for a new button
# tap, or returns immediately if one's already pending. Not a sleep-then-
# poll interval.
TELEGRAM_FEEDBACK_POLL_TIMEOUT = int(os.getenv("TELEGRAM_FEEDBACK_POLL_TIMEOUT", "25"))
# How long to back off after a 409 Conflict (another process already
# long-polling this same bot token) before retrying — deliberately longer
# than the 5s used for other transient errors, since retrying fast can't
# make a conflict resolve any sooner. See main.py's telegram_feedback_loop.
TELEGRAM_CONFLICT_BACKOFF_SECONDS = int(os.getenv("TELEGRAM_CONFLICT_BACKOFF_SECONDS", "30"))

# ---- Repost/duplicate detection ---------------------------------------------
# Flags (advisory only — never suppresses a notification) when a matched
# project looks like a near-duplicate of one already notified about — see
# repost_detector.py's docstring for the full rationale and the false-
# positive trade-off (similarly-worded but genuinely unrelated projects
# from different clients can happen with generic/templated descriptions,
# which is exactly why this warns rather than blocks).
REPOST_DETECTION_ENABLED = os.getenv("REPOST_DETECTION_ENABLED", "true").strip().lower() == "true"
# Similarity ratio (0-1, via stdlib difflib.SequenceMatcher) at/above which
# a project is flagged as a likely repost. Kept high by default (0.85) to
# minimize false positives on the text-comparison approach's biggest
# limitation — set higher for more caution, lower to catch loosely-
# reworded reposts at the cost of more false positives.
REPOST_SIMILARITY_THRESHOLD = float(os.getenv("REPOST_SIMILARITY_THRESHOLD", "0.85"))
# How many recently-notified projects to keep comparing new ones against —
# oldest entries are evicted once over this cap, same bounded-growth
# pattern as ScoreCache.
REPOST_HISTORY_MAX_ENTRIES = int(os.getenv("REPOST_HISTORY_MAX_ENTRIES", "300"))
# Entries older than this many days are pruned before comparison, so a
# project posted months ago can't produce a stale "repost" match.
REPOST_HISTORY_MAX_AGE_DAYS = int(os.getenv("REPOST_HISTORY_MAX_AGE_DAYS", "30"))
REPOST_HISTORY_FILE = os.getenv("REPOST_HISTORY_FILE", "repost_history.json")

# ---- Matching / scoring -------------------------------------------------------
# DEFAULT skill list, used only if the MY_SKILLS env var isn't set (see
# below). Mostaql project tags are frequently in Arabic (e.g. "علم
# البيانات" rather than "Data Science"), so each skill below has an
# Arabic entry alongside its English one — the local pre-filter matches
# whichever one appears in the project's actual tags/title/description.
_DEFAULT_MY_SKILLS = [
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

# Comma-separated skill list — overrides _DEFAULT_MY_SKILLS above entirely
# when set (same "full override, not additive" convention as
# GEMINI_API_KEYS/MOSTAQL_CATEGORIES elsewhere in this file), e.g.:
#   MY_SKILLS=Python,بايثون,React,ريأكت,WordPress,ووردبريس
# This is what makes retuning what the bot looks for a matter of editing
# ONE environment variable — in Render's dashboard this takes effect on
# the next restart, no code change or git push required — rather than
# editing the Python list above and redeploying. Add more Arabic synonyms
# if you notice real projects being filtered out that shouldn't be (check
# the "Fetched N tag(s) for project ..." log line in scraper.py to see
# what Mostaql is actually tagging things with).
_env_skills = os.getenv("MY_SKILLS", "").strip()
if _env_skills:
    MY_SKILLS = [s.strip() for s in _env_skills.split(",") if s.strip()]
else:
    MY_SKILLS = _DEFAULT_MY_SKILLS
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "60"))

# ---- Adaptive threshold under quota pressure --------------------------------
# As today's ACTUAL Gemini request count (see ai_agent.DailyRequestTracker —
# real requests, not projects; a batch scoring call is 1 request regardless
# of how many projects it covers) climbs toward the estimated daily quota,
# the EFFECTIVE threshold used to decide "does this clear the bar" ramps up
# from MATCH_THRESHOLD toward ADAPTIVE_THRESHOLD_HARD_CAP — see
# ai_agent.get_effective_match_threshold(). The idea: spend the LAST portion
# of the day's request budget on the strongest remaining candidates instead
# of running out partway through an average one on a first-come-first-served
# basis. MATCH_THRESHOLD itself is left completely unchanged by this — it's
# always the FLOOR the effective threshold ramps up FROM, never below it.
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").strip().lower() == "true"
# Rough total daily request budget across ALL configured keys combined.
# Free tier is commonly ~20 requests/day per key for gemini-3.5-flash (see
# the POLL_INTERVAL note below) — defaults to that figure times however
# many keys are configured; override directly if your actual tier differs.
GEMINI_ESTIMATED_DAILY_QUOTA = int(os.getenv("GEMINI_ESTIMATED_DAILY_QUOTA", str(20 * max(len(GEMINI_API_KEYS), 1))))
# Ramping starts once today's usage crosses this fraction of the estimated
# quota (0.7 = starts tightening at 70% used) and reaches the hard cap at
# 100%+ used.
ADAPTIVE_THRESHOLD_TRIGGER_RATIO = float(os.getenv("ADAPTIVE_THRESHOLD_TRIGGER_RATIO", "0.7"))
# The effective threshold never climbs above this, however close to (or
# past) the quota wall the day gets — keeps at least a chance of matching
# a truly exceptional project even at 100%+ of estimated quota used,
# rather than a threshold that could climb high enough to reject everything.
ADAPTIVE_THRESHOLD_HARD_CAP = float(os.getenv("ADAPTIVE_THRESHOLD_HARD_CAP", "90"))
DAILY_REQUEST_COUNT_FILE = os.getenv("DAILY_REQUEST_COUNT_FILE", "daily_request_count.json")

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
#
# gemini-2.5-flash-lite was REMOVED from this default (previously the
# 2nd entry) after Google retired it entirely — it started returning a
# permanent `404 NOT_FOUND: ... no longer available to new users` for
# every single call, not a transient/rate-limit error. Unlike a rate
# limit (worth retrying later, or on a different key/model), a retired
# model can NEVER succeed again — leaving it in the cascade meant every
# fallback chain that fell through past the first model wasted an
# attempt (and the latency of a full request round-trip) on something
# guaranteed to fail, before ever reaching a model that could actually
# work. If gemini-2.5-flash (still below) starts doing the same, remove
# it here too — Google tends to retire an entire model generation
# together, though that isn't confirmed for this one as of this writing.
GEMINI_MODEL_CASCADE = [
    m.strip() for m in os.getenv(
        "GEMINI_MODEL_CASCADE",
        "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-2.5-flash",
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
