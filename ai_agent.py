"""
ai_agent.py
-----------
Uses Google's Gen AI SDK (`from google import genai`) to:
  1. Score how well a project matches the freelancer's skill set (0-100),
     and estimate a suggested bid price and delivery time.
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's details.

QUOTA-AWARE MODEL CASCADE + KEY ROTATION
-------------------------------------------------------------------
The actual Gemini call plumbing (model hierarchy, RPM/RPD-aware 429
handling, round-robin key rotation, exponential backoff) lives in
`gemini_client.py` as a small, fully async, framework-free module — see
its docstring for the complete design. In short:

  * MODEL-MAJOR cascade: every key is tried on the current (highest-
    quota) model before moving to the next model, so a high-RPM/RPD
    model is drained across ALL keys before a scarcer one is ever
    touched. Order: config.GEMINI_MODEL_CASCADE (default: the two
    high-quota "-lite" models, then six low-quota fallbacks).
  * Blacklisted (0 RPM / 0 RPD, deprecated/disabled) models can never be
    configured or routed to — enforced in gemini_client.build_cascade()/
    is_blacklisted(), independent of what a stale env var might contain.
  * An HTTP 429 is classified as RPM or RPD from Google's own
    `QuotaFailure.violations[].quotaId` (NOT from the retryDelay hint,
    which does not distinguish the two): an RPM 429 rotates to the next
    key on the SAME model; an RPD 429 parks that (model, key) until the
    next Pacific-time midnight (when Google resets RPD) and moves on.
  * When every (model, key) pair is unavailable, the whole cascade is
    retried with exponential backoff (bounded number of sweeps, jitter,
    a hard overall deadline) before raising.

SYNC-OVER-ASYNC BRIDGE
-------------------------------------------------------------------
gemini_client.GeminiCascadeClient is `async`. The rest of this codebase
(main.py's producer/consumer threads, reply_assistant.py, etc.) is
synchronous. `_generate()` below is the same synchronous entry point
every caller already uses; internally it schedules the async call onto
ONE dedicated background event-loop thread (`_ASYNC_LOOP`) and blocks the
calling thread on the result — preserving both "only one Gemini call in
flight process-wide" (the old global mutex's guarantee) and every
existing caller's synchronous call signature, so score_project(),
draft_proposal(), draft_screening_answers(), and
reply_assistant.get_reply_options() needed NO changes beyond importing
this module.

LOCAL TAG PRE-FILTERING (zero API cost for irrelevant projects)
-------------------------------------------------------------------
Before any Gemini call is made, `local_skill_prefilter()` compares the
project's official Mostaql skill tags (scraped from its detail page —
see scraper.fetch_project_tags) against config.MY_SKILLS — which can mix
English and Arabic entries freely; matching is plain case-insensitive
substring matching, language-agnostic (Python's str.lower() is a safe
no-op on Arabic script, so mixed-language lists just work). If there's no
overlap at all, `evaluate_project()` returns immediately with
match_score=0.0 and makes ZERO Gemini API calls for that project.

IMPORTANT — fail-open by design: if tags weren't fetched (empty list —
either FETCH_PROJECT_TAGS is off, or the detail-page fetch/parse failed),
the pre-filter does NOT block the project; it falls through to the normal
Gemini evaluation. We would rather spend an API call on an uncertain
project than silently drop a good one because of a scraping gap.

Any error _generate() raises (QuotaExhaustedError, a non-retryable API
error, or the sync bridge's own errors) is caught by score_project()/
draft_proposal() via their existing broad `except Exception` and turned
into `None`, which evaluate_project() turns into a safe fallback
Evaluation (match_score=0.0, suggested_price=None, delivery_days=None)
rather than letting the exception propagate — so one bad project can
never take down main.py's loop.

SDK NOTE: model selection is now the full quota-aware cascade described
above (config.GEMINI_MODEL_CASCADE), not a single fixed model.
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from google.genai import types

import config
import db
import gemini_client
from gemini_client import (
    AllKeysExhaustedError,  # noqa: F401  (back-compat alias, see below)
    AllKeysRateLimited,     # noqa: F401  (back-compat alias, see below)
)

# Optional final-fallback for genuinely malformed JSON (e.g. a stray/garbled
# token where a value should be — corruption beyond what regex cleanup can
# reliably fix). json_repair is purpose-built for exactly this: repairing
# broken JSON from LLM outputs. If it's not installed, _extract_json still
# works via the sanitizer + balanced-brace scan below; json_repair is only
# the last line of defense for the messiest cases.
try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

logger = logging.getLogger("ai_agent")


# ---------------------------------------------------------------------------
# Async cascade client + the sync bridge every existing caller uses
# ---------------------------------------------------------------------------
# One GeminiCascadeClient per process, built from config.py's key list and
# cascade. See gemini_client.py's module docstring for the full design
# (model-major cascade, RPM-vs-RPD-aware 429 handling, round-robin key
# rotation, bounded exponential backoff).
_cascade_client = gemini_client.GeminiCascadeClient(
    api_keys=config.GEMINI_API_KEYS,
    cascade=gemini_client.build_cascade(
        overrides=config.GEMINI_MODEL_CASCADE,
        rpm_overrides=config.MODEL_RPM_LIMITS,
        rpd_overrides=getattr(config, "MODEL_RPD_LIMITS", None),
    ),
    timeout_seconds=config.GEMINI_TIMEOUT,
    proxy_url=config.GEMINI_PROXY_URL,
    tunables=gemini_client._Tunables(
        transient_retries=config.GEMINI_MAX_TRANSIENT_RETRIES,
        transient_backoff_base=config.GEMINI_RETRY_BACKOFF_BASE,
        transient_backoff_cap=config.GEMINI_QUOTA_BACKOFF_MAX,
        inter_request_delay=config.GEMINI_INTER_REQUEST_DELAY,
        backoff_sweeps=getattr(config, "GEMINI_BACKOFF_SWEEPS", 3),
        backoff_base=getattr(config, "GEMINI_OUTER_BACKOFF_BASE", 2.0),
        backoff_cap=getattr(config, "GEMINI_OUTER_BACKOFF_CAP", 60.0),
        total_deadline=getattr(config, "GEMINI_TOTAL_DEADLINE_SECONDS", 180.0),
    ),
)

if config.GEMINI_PROXY_URL:
    logger.info(
        "Gemini: routing ONLY Gemini API traffic through proxy %s "
        "(scraper/Telegram/GitHub traffic is unaffected and stays direct).",
        config.GEMINI_PROXY_URL,
    )


class _AsyncLoopThread:
    """
    Owns ONE background thread running ONE asyncio event loop for the
    lifetime of the process. `run()` schedules a coroutine onto that loop
    from any synchronous caller and blocks until it completes — this is
    what lets every existing synchronous call site (score_project(),
    draft_proposal(), reply_assistant.get_reply_options(), ...) call
    `_generate()` exactly as before, with no `async`/`await` of their own,
    while the real Gemini call happens through gemini_client's async
    cascade engine.

    Using a single loop (rather than a fresh `asyncio.run()` per call)
    also means only one Gemini call is ever in flight at a time,
    process-wide — the same guarantee the old global `_generate_lock`
    provided — since main.py's producer/consumer architecture already
    only calls this from one thread at a time, and every call funnels
    through this one loop regardless of which OS thread issued it.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop_forever, name="gemini-asyncio", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run_loop_forever(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro):
        """Blocks the CALLING (synchronous) thread until `coro` completes
        on the dedicated event loop, returning its result or re-raising
        its exception unchanged."""
        if self._loop is None:
            raise RuntimeError("Gemini async event loop failed to start")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_ASYNC_LOOP = _AsyncLoopThread()


def _generate(
    prompt: str,
    json_mode: bool = False,
    response_schema=None,
    temperature: float = None,
    max_output_tokens: int = None,
):
    """
    Synchronous entry point used by every caller in this codebase
    (score_project, score_projects_batch, draft_proposal,
    draft_screening_answers, reply_assistant.get_reply_options). Builds
    the GenerateContentConfig, runs the async cascade
    (gemini_client.GeminiCascadeClient.generate) on the dedicated event
    loop via `_ASYNC_LOOP.run()`, updates the daily-request counter and
    "which key served this" bookkeeping on success, and returns the raw
    SDK response object — exactly the same shape/contract the previous
    implementation returned, so every downstream `.parsed` / `.text` /
    `.usage_metadata` access below is unaffected.

    Raises gemini_client.QuotaExhaustedError when the entire model x key
    grid is exhausted (kept importable here as AllKeysExhaustedError for
    backward compatibility with existing call sites/tests that reference
    ai_agent.AllKeysExhaustedError), or the original non-retryable
    exception for anything else (bad request, auth failure, safety
    block, ...).
    """
    global _current_key_index, _current_model

    gen_config_kwargs = {
        # Explicit, visible guarantee: this codebase never passes `tools=`
        # to Gemini, so Automatic Function Calling cannot trigger today
        # regardless — disabling it here makes that guarantee visible
        # rather than implicit, so a future change can't silently
        # introduce hidden remote calls without this line having to
        # change too.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    # response_schema requires response_mime_type="application/json" to be
    # set too (the SDK validates this) — implied automatically here so
    # callers only need to pass response_schema, not both.
    if json_mode or response_schema is not None:
        gen_config_kwargs["response_mime_type"] = "application/json"
    if response_schema is not None:
        gen_config_kwargs["response_schema"] = response_schema
    if temperature is not None:
        gen_config_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        gen_config_kwargs["max_output_tokens"] = max_output_tokens
    gen_config = types.GenerateContentConfig(**gen_config_kwargs)

    result = _ASYNC_LOOP.run(_cascade_client.generate(prompt, gen_config))

    # Only a genuinely successful call counts toward RPD — see
    # DailyRequestTracker's docstring for why failed attempts are
    # deliberately excluded.
    _daily_request_tracker.increment()
    _current_key_index = result.key_index
    _current_model = result.model
    return result.response


def generate_with_outer_backoff(prompt: str, **kwargs):
    """
    Thin compatibility wrapper: the cascade client ALREADY performs bounded
    exponential backoff internally across full model x key sweeps (see
    gemini_client.GeminiCascadeClient.generate) before ever raising
    QuotaExhaustedError, so there is no additional outer retry to add here
    — this simply calls `_generate()`. Kept as a distinct name because
    existing code/tests may reference `ai_agent.generate_with_outer_backoff`
    directly.
    """
    return _generate(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Back-compat state mirrors
# ---------------------------------------------------------------------------
# A few call sites (record_token_usage, health_server, tests) read these
# module-level globals directly to report "which key/model served the last
# call" — kept as plain attributes updated by _generate() above, rather
# than reaching into the cascade client's internals, so that surface is
# unaffected by this rewrite.
_current_key_index = 0
_current_model: Optional[str] = None


@dataclass
class Evaluation:
    """Container for one project's full AI evaluation outcome — filled in
    by evaluate_project()/evaluate_projects_batch() and consumed by
    main.py to decide whether to notify, draft, or queue for retry."""
    match_score: float
    reasoning: str
    suggested_price: Optional[str] = None
    delivery_days: Optional[int] = None
    proposal_ar: Optional[str] = None
    # Fast-scan breakdown for human review in Telegram — which of the
    # freelancer's OWN skills this project actually calls for, vs. which
    # skills/technologies the project needs that aren't in the skill list
    # at all (real gaps). Populated by score_project()/score_projects_batch
    # via ProjectScoreSchema/_BatchScoreItem; stays empty (not an error)
    # for locally-filtered or AI-failed evaluations, since neither ran a
    # real scoring call.
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    # True specifically when the Gemini call itself failed (e.g. every key
    # in GEMINI_API_KEYS hit 429/RESOURCE_EXHAUSTED, or another API error) —
    # as opposed to a successful call that simply scored the project low.
    # main.py uses this to route to the GitHub fallback instead of the
    # normal "below threshold" path, since match_score=0.0 alone can't
    # distinguish "genuinely irrelevant project" from "we never actually
    # found out."
    ai_failed: bool = False
    # --- Analytics metadata, consolidated for TokenUsageTracker (see
    # record_token_usage()) — main.py reads these off the Evaluation object
    # after deciding whether to notify Telegram, since that decision is the
    # one piece of the record this module can't know on its own.
    original_desc_length: int = 0
    truncated_desc_length: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    response_time_sec: float = 0.0
    key_alias: Optional[str] = None
    proposal_generated: bool = False
    # The threshold ACTUALLY used to decide whether to draft a proposal for
    # this project — normally equal to config.MATCH_THRESHOLD, but may be
    # higher if get_effective_match_threshold() ramped it up under quota
    # pressure (see config.ADAPTIVE_THRESHOLD_ENABLED). Callers should
    # compare match_score against THIS, not config.MATCH_THRESHOLD directly
    # — otherwise a project whose proposal was skipped due to a raised
    # effective threshold would be logged/reported against the wrong bar.
    effective_threshold: float = 0.0
    # --- Budget/timeline adherence (see ProjectScoreSchema.suggested_price/
    # delivery_days docstrings for the actual rule) — surfaced separately so
    # notifier.py can show a short note ONLY when the bot deviated from what
    # the client explicitly asked for, instead of silently substituting a
    # different number with no explanation visible to the human reviewer.
    budget_timeline_adjusted: bool = False
    budget_timeline_note: str = ""
    # --- Screening-question answers (see draft_screening_answers()) — a
    # list of {"question": str, "answer": str} dicts, one per detected
    # question, in the SAME order as project.screening_questions. Empty
    # list if the project had no screening questions, OR if it did but
    # drafting answers for them failed (never raises; degrades to []
    # rather than blocking the rest of the proposal).
    screening_answers: List[dict] = field(default_factory=list)


def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Scans for the first top-level {...} object using string-aware
    brace-depth tracking, instead of relying on regex alone. While
    scanning, any RAW (unescaped) control character found INSIDE a string
    literal — most commonly a literal newline, because Gemini sometimes
    line-wraps a free-text field like "reasoning" without escaping it — is
    rewritten to its valid escaped form (\\n, \\r, \\t). This is invisible
    in a terminal/log either way, which is exactly why this class of bug
    is so easy to miss just by reading the logged text: a raw newline
    inside a JSON string and the whitespace between key-value pairs look
    identical when printed, but only one of them is valid JSON.

    Why brace-depth tracking instead of regex: a plain greedy regex like
    r'\\{.*\\}' matches from the FIRST '{' to the VERY LAST '}' anywhere in
    the text, which breaks on trailing prose, a stray extra closing brace,
    or a second brace-like fragment further in the response. Tracking
    depth character-by-character (while staying string-aware) finds the
    exact end of the real JSON object and ignores everything after it.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    out = []
    control_escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                out.append(ch)
                continue
            if ch == "\\":
                escape = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch in control_escapes:
                out.append(control_escapes[ch])  # fix: escape the raw control char
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)

    return None  # unbalanced (truncated response) — no complete object found


# Smart/curly quotes and other invisible Unicode characters that Gemini
# occasionally emits and that break json.loads while being visually
# indistinguishable (or literally invisible) from valid JSON in a log.
_SMART_QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"',   # “ ”  -> "
    "\u2018": "'", "\u2019": "'",   # ‘ ’  -> '
    "\u00a0": " ",                  # non-breaking space -> regular space
    "\u200b": "", "\u200c": "", "\u200d": "",  # zero-width chars -> removed
}


def _sanitize_text(text: str) -> str:
    """Strips a BOM and normalizes smart quotes / invisible Unicode
    whitespace that break json.loads but render as normal-looking
    characters (or nothing at all) wherever this text gets logged."""
    text = text.lstrip("\ufeff")
    for bad, good in _SMART_QUOTE_MAP.items():
        text = text.replace(bad, good)
    return text


def _strip_trailing_commas(text: str) -> str:
    """Removes a trailing comma right before a closing '}' or ']' — a very
    common small mistake in LLM-generated JSON that json.loads rejects
    outright (e.g. '{"a": 1,}')."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extracts and parses a JSON object from Gemini's response,
    tolerating the anomalies Gemini actually produces in practice:
    markdown fences, leading/trailing prose, trailing garbage or a stray
    extra closing brace, smart/curly quotes, a leading BOM, invisible
    Unicode whitespace, trailing commas, raw unescaped control characters
    (typically a literal newline) inside a string value, and — as a final
    layer — genuinely garbled/corrupted key-value pairs (e.g. a stray
    extra ':'/'"' where a value should be) via json_repair.

    Strategy, most-common-case first, most-permissive last:
      1. Sanitize invisible/smart characters, strip ```json fences,
         attempt a direct parse.
      2. Balanced-brace scan (string-aware, also fixes raw control chars
         found inside strings) for the first complete {...} object.
      3. Same balanced-brace result with trailing commas stripped, in case
         that was the (additional) problem.
      4. Greedy regex (\\{.*\\}), also with trailing commas stripped, in
         case the balanced scan found nothing at all (e.g. a genuinely
         truncated response).
      5. json_repair (if installed) as an absolute last resort for
         corruption too irregular for regex-based cleanup to fix
         deterministically — it may occasionally guess a value differently
         than intended for truly ambiguous input, but it reliably avoids
         raising, which is the actual goal: a formatting slip should never
         crash the pipeline.
    """
    sanitized = _sanitize_text(text.strip())
    cleaned = re.sub(r"^```(?:json)?|```$", "", sanitized, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    balanced = _extract_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            try:
                return json.loads(_strip_trailing_commas(balanced))
            except json.JSONDecodeError:
                pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_strip_trailing_commas(candidate))
            except json.JSONDecodeError:
                pass

    # Absolute last resort: json_repair is a heuristic JSON repair library
    # built for exactly this — genuinely garbled/corrupted LLM output that
    # regex-based cleanup can't reliably fix (e.g. a stray extra ':'/'"'
    # where a value should be, a missing comma, an unquoted key, etc.).
    # It may occasionally guess a value differently than intended for truly
    # ambiguous corruption, but it reliably avoids raising — which is the
    # actual goal here: never let a formatting slip crash the pipeline.
    if _HAS_JSON_REPAIR:
        try:
            repaired = json_repair.loads(cleaned)
            if isinstance(repaired, dict) and repaired:
                logger.warning(
                    "Standard JSON parsing failed — json_repair recovered a "
                    "result, but please verify it looks sane: %s", repaired,
                )
                return repaired
        except Exception as exc:
            logger.debug("json_repair also failed: %s", exc)

    logger.error("Could not parse JSON from Gemini response: %s", text[:500])
    return None


def parse_gemini_json(response_text: str) -> dict:
    r"""
    Public JSON-extraction entry point used by score_project(). Delegates
    to _extract_json() above rather than a plain regex, because a plain
    greedy boundary match like r'(\{[\s\S]*\}|\[[\s\S]*\])' matches from
    the FIRST '{' to the LAST '}' anywhere in the text — which breaks
    exactly on the anomalies Gemini actually produces (trailing prose
    after the JSON, a stray extra closing brace, etc.), swallowing
    everything in between into one invalid blob. _extract_json already
    does markdown-fence stripping + boundary extraction (what was asked
    for here) via a string-aware balanced-brace scan instead, PLUS several
    real-world cases found in production: smart/curly quotes, a leading
    BOM, raw control characters inside string values, trailing commas, and
    a json_repair last-resort pass for corruption too irregular for regex
    to fix deterministically.

    Raises ValueError (matching the originally-specified contract) instead
    of returning None, so callers that want a hard failure signal get one.
    """
    result = _extract_json(response_text)
    if result is None:
        raise ValueError(f"Failed to extract JSON from Gemini output: {response_text[:100]}")
    return result


# ---------------------------------------------------------------------------
# Local tag pre-filter (zero API cost)
# ---------------------------------------------------------------------------

def _skill_tokens(skill: str) -> List[str]:
    """Expands a skill entry into extra matchable tokens, e.g.
    'Object-Oriented Programming (OOP)' -> also match plain 'OOP'."""
    tokens = [skill.strip()]
    paren_match = re.search(r"\(([^)]+)\)", skill)
    if paren_match:
        tokens.append(paren_match.group(1).strip())
    stripped = re.sub(r"\([^)]*\)", "", skill).strip()
    if stripped and stripped not in tokens:
        tokens.append(stripped)
    return [t for t in tokens if len(t) >= 2]


def local_skill_prefilter(tags: List[str], title: str = None, description: str = None) -> bool:
    """
    Returns True if the project should proceed to Gemini evaluation, False
    if it should be skipped locally with zero API cost.

    Two independent checks, tried in order — either one finding an overlap
    is enough to proceed to Gemini. Biased toward failing OPEN throughout,
    since a false NEGATIVE here silently drops a lead with zero visibility
    (nothing logs "this might have been a good match"), while a false
    positive just costs one avoidable Gemini call:

      1. Official tag overlap — if `tags` is non-empty, this is the
         AUTHORITATIVE check: a non-empty tag list with zero overlap
         against config.MY_SKILLS returns False immediately, exactly as
         before this function had a second check. Mostaql's own tags are
         a more reliable signal than free-text keyword matching when
         they're actually available, so they're trusted on their own
         without falling through to check #2 below.
      2. Title/description keyword overlap — runs ONLY when tags are
         unavailable (empty/missing — e.g. FETCH_PROJECT_TAGS=false, or
         Mostaql simply didn't provide any for this project). Previously,
         an untagged project unconditionally proceeded to Gemini
         regardless of actual relevance — this applies the SAME substring
         matching used for tags to the project's own title+description
         text instead, so an untagged project with clearly zero skill-
         keyword overlap anywhere in its own text can also be skipped at
         zero API cost. Disable via config.TITLE_PREFILTER_ENABLED if this
         proves too aggressive for your skill list's phrasing.

    Fails open (returns True) if NEITHER tags NOR any usable title/
    description text is available at all — nothing to check against.
    """
    if tags:
        tag_texts = [t.lower() for t in tags if t]
        if tag_texts:
            for skill in config.MY_SKILLS:
                for token in _skill_tokens(skill):
                    token_l = token.lower()
                    for tag in tag_texts:
                        if token_l in tag or tag in token_l:
                            return True
            return False  # tags existed and were checked — authoritative "no"

    if not config.TITLE_PREFILTER_ENABLED:
        return True

    text = f"{title or ''} {description or ''}".strip().lower()
    if not text:
        return True  # nothing to check against — fail open

    for skill in config.MY_SKILLS:
        for token in _skill_tokens(skill):
            if token.lower() in text:
                return True

    return False


def smart_truncate_description(description: str, max_length: int = None) -> str:
    """
    Trims an oversized project description to control prompt token usage on
    outlier projects with massive descriptions. Title and tags are NEVER
    touched by this function — it only ever receives/returns the
    description text itself, and callers are responsible for keeping
    title/tags separate (which evaluate_project already does).

    Truncates to the first `max_length` characters (default:
    config.GEMINI_DESCRIPTION_MAX_CHARS) and appends a clear marker, so the
    cut is visible/auditable in the actual prompt and Gemini doesn't
    mistake the cut-off point for the description's natural ending.
    """
    if max_length is None:
        max_length = config.GEMINI_DESCRIPTION_MAX_CHARS
    if not description or len(description) <= max_length:
        return description
    return description[:max_length].rstrip() + "... [description truncated for evaluation]"


class ScoreCache:
    """
    Lightweight, fail-safe cache — backed by MongoDB Atlas's `score_cache`
    collection as of the Sept 2026 migration off local JSON files — mapping
    a hash of (title, the FULL untruncated description, current MY_SKILLS)
    -> the score result Gemini already produced for that exact content. A
    project re-evaluated with byte-for-byte identical text — most commonly
    the retry queue re-checking an entry whose earlier AI call failed, or a
    Mostaql repost with unchanged text — hits this cache and skips a fresh
    Gemini call entirely.

    Only the SCORING result is cached (match_score/reasoning/
    suggested_price/delivery_days) — NOT the proposal. See config.py's
    SCORE_CACHE_ENABLED comment for why proposal drafting always runs
    fresh regardless of cache hits.

    Silent/fail-safe by construction, matching TokenUsageTracker: any
    MongoDB read/write error degrades to "cache miss" rather than raising,
    since a broken cache must never interrupt evaluation. Uses the full,
    untruncated description as part of the key (not whatever truncated
    text a particular call happened to use) so the cache reflects the
    project's real identity, independent of GEMINI_SCORING_DESCRIPTION_MAX_CHARS.
    """

    def __init__(self, collection=None, max_entries: int = None):
        # `collection` is only ever passed explicitly by tests (an
        # injected mongomock collection, or a broken double to exercise
        # the fail-safe path) — production code always goes through
        # db.get_collection() so it picks up whatever database is active
        # at call time, not whatever it was at construction time.
        self._collection = collection
        self.max_entries = max_entries or config.SCORE_CACHE_MAX_ENTRIES

    def _coll(self):
        return self._collection if self._collection is not None else db.get_collection("score_cache")

    @staticmethod
    def _key(title: str, description: str) -> str:
        # Skills fingerprint included so a MY_SKILLS change naturally
        # invalidates every previously-cached score (different hash) —
        # this can never silently serve a score computed against a skill
        # list that no longer reflects config.py.
        skills_fingerprint = ",".join(sorted(config.MY_SKILLS))
        raw = f"{title or ''}\n{description or ''}\n{skills_fingerprint}"
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

    def get(self, title: str, description: str) -> Optional[dict]:
        if not config.SCORE_CACHE_ENABLED:
            return None
        try:
            key = self._key(title, description)
            doc = self._coll().find_one({"_id": key})
            if not doc:
                return None
            return {
                "match_score": doc.get("match_score"),
                "reasoning": doc.get("reasoning"),
                "matched_skills": doc.get("matched_skills"),
                "missing_skills": doc.get("missing_skills"),
                "suggested_price": doc.get("suggested_price"),
                "delivery_days": doc.get("delivery_days"),
                "budget_timeline_adjusted": doc.get("budget_timeline_adjusted", False),
                "budget_timeline_note": doc.get("budget_timeline_note", ""),
            }
        except Exception:
            return None  # unreachable Atlas, anything else -> cache miss

    def set(self, title: str, description: str, score_data: dict) -> None:
        if not config.SCORE_CACHE_ENABLED:
            return
        try:
            key = self._key(title, description)
            coll = self._coll()

            # Only the fields evaluate_project()/evaluate_projects_batch()
            # actually consume from a score result — never cache anything
            # else that might sneak into score_data.
            coll.update_one(
                {"_id": key},
                {"$set": {
                    "match_score": score_data.get("match_score"),
                    "reasoning": score_data.get("reasoning"),
                    "matched_skills": score_data.get("matched_skills"),
                    "missing_skills": score_data.get("missing_skills"),
                    "suggested_price": score_data.get("suggested_price"),
                    "delivery_days": score_data.get("delivery_days"),
                    "budget_timeline_adjusted": score_data.get("budget_timeline_adjusted", False),
                    "budget_timeline_note": score_data.get("budget_timeline_note", ""),
                    "cached_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            )

            # Evict oldest entries (by cached_at) once over the cap —
            # replaces the old "trim the dict to N keys" logic; a TTL
            # index (see db._ensure_indexes) is the second line of
            # defense in production, but this row-count cap is what the
            # tests exercise directly and keeps behavior deterministic.
            count = coll.count_documents({})
            if count > self.max_entries:
                overflow = count - self.max_entries
                oldest_ids = [
                    d["_id"] for d in
                    coll.find({}, {"_id": 1}).sort("cached_at", 1).limit(overflow)
                ]
                if oldest_ids:
                    coll.delete_many({"_id": {"$in": oldest_ids}})
        except Exception:
            pass  # silent/fail-safe, matching TokenUsageTracker


_score_cache = ScoreCache()


class DailyRequestTracker:
    """
    Tracks how many ACTUAL Gemini API requests succeeded today (UTC
    calendar day), across all keys/models combined. A batch scoring call
    counts as ONE request regardless of how many projects it scored —
    this is deliberately request-count-based (matching how Gemini's RPD
    quota itself works), not project-count-based.

    This is the source of truth for get_effective_match_threshold() below
    — NOT TokenUsageTracker, which logs one row per fully-processed
    PROJECT (0 calls on a cache hit, 1 for scoring only, up to 2 including
    a proposal), making it unsuitable for counting raw requests.

    Only counts calls that actually SUCCEEDED. Failed attempts (429s,
    transient errors) are deliberately not counted here — they're already
    handled by gemini_client's local proactive RPM limiter and its
    model/key cascade (reactive) separately, so this tracker stays
    focused on one question:
    "how much real scoring/drafting work got done today," which is what
    adaptive thresholding actually wants to protect.

    Persisted to MongoDB Atlas's `daily_request_count` collection, one
    document per UTC date (_id = "YYYY-MM-DD"), so a Render restart
    mid-day doesn't reset the count to zero — and, being keyed by date,
    "resets" for a new day automatically with no explicit reset logic
    needed (today's key simply doesn't exist yet). Silent/fail-safe like
    every other tracker in this file: any Mongo error just behaves as if
    zero requests have been made today, rather than raising. Uses an
    atomic `$inc` rather than a read-modify-write, so — unlike the old
    local-file version — concurrent increments from different
    threads/instances can never race and lose a count.
    """

    def __init__(self, collection=None):
        self._collection = collection

    def _coll(self):
        return self._collection if self._collection is not None else db.get_collection("daily_request_count")

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def increment(self) -> None:
        try:
            self._coll().update_one({"_id": self._today()}, {"$inc": {"count": 1}}, upsert=True)
        except Exception:
            pass

    def get_today_count(self) -> int:
        try:
            doc = self._coll().find_one({"_id": self._today()})
            return int(doc.get("count", 0)) if doc else 0
        except Exception:
            return 0


_daily_request_tracker = DailyRequestTracker()


def get_effective_match_threshold() -> float:
    """
    Returns the match-score threshold to actually use for THIS moment,
    which may be higher than config.MATCH_THRESHOLD if today's real
    Gemini request count (see DailyRequestTracker) has crossed
    config.ADAPTIVE_THRESHOLD_TRIGGER_RATIO of the estimated daily quota.

    Rationale: without this, the bot evaluates every new project at the
    same fixed bar all day, and once the daily quota is actually
    exhausted, EVERY project from that point on falls back to GitHub
    regardless of how good a match it might have been — the quota gets
    spent on a first-come-first-served basis rather than a
    best-candidates basis. Ramping the threshold up as the quota gets
    tight spends the LAST portion of the day's budget more selectively,
    on stronger matches only, instead of running out partway through an
    average project.

    Below the trigger ratio: returns config.MATCH_THRESHOLD unchanged (no
    behavior change at all under normal, non-quota-pressured conditions).
    Above it: ramps LINEARLY from MATCH_THRESHOLD up to
    config.ADAPTIVE_THRESHOLD_HARD_CAP as usage climbs from the trigger
    ratio to 100%+ of the estimated quota — the hard cap exists so the
    threshold can never climb so high that literally nothing could ever
    match, even once the quota is fully or over spent.

    Returns config.MATCH_THRESHOLD unchanged (no-op) if
    config.ADAPTIVE_THRESHOLD_ENABLED is False, or if the estimated quota
    is configured as 0/negative (nothing sensible to ramp against).
    """
    base = config.MATCH_THRESHOLD
    if not config.ADAPTIVE_THRESHOLD_ENABLED or config.GEMINI_ESTIMATED_DAILY_QUOTA <= 0:
        return base

    hard_cap = max(config.ADAPTIVE_THRESHOLD_HARD_CAP, base)  # never ramp below the configured base
    trigger = min(max(config.ADAPTIVE_THRESHOLD_TRIGGER_RATIO, 0.0), 0.999)  # avoid a zero-width ramp window

    usage_ratio = _daily_request_tracker.get_today_count() / config.GEMINI_ESTIMATED_DAILY_QUOTA
    if usage_ratio <= trigger:
        return base

    progress = min((usage_ratio - trigger) / (1.0 - trigger), 1.0)
    effective = round(min(base + progress * (hard_cap - base), hard_cap), 1)
    if effective > base:
        logger.info(
            "Adaptive threshold active: today's request count is at %.0f%% of the "
            "estimated daily quota (%s/%s) — effective threshold raised from %.0f%% to %.0f%%",
            usage_ratio * 100, _daily_request_tracker.get_today_count(),
            config.GEMINI_ESTIMATED_DAILY_QUOTA, base, effective,
        )
    return effective


class TokenUsageTracker:
    """
    Lightweight, fail-safe, SILENT analytics logger. Inserts ONE
    consolidated document per fully-processed project into MongoDB
    Atlas's `token_usage_stats` collection — not one record per raw
    Gemini call (a project can involve up to two calls: scoring, and
    optionally proposal drafting; their token counts and response times
    are summed into a single row by record_token_usage() below, since
    main.py needs to add sent_to_telegram AFTER both calls and the
    notification decision are already done).

    Writing directly to MongoDB on every call (rather than the old
    batched "sync to GitHub every N minutes" approach) is deliberate: a
    single insert_one() has none of the cost a git commit did, so there's
    no reason to batch it, and it means this data survives a restart
    immediately rather than up to TOKEN_STATS_SYNC_INTERVAL seconds late.

    SILENT means exactly that: this class never calls print(), logger.*,
    or anything else that writes to stdout/console — not even on failure.
    Any MongoDB error (connection hiccup, timeout, etc.) is caught and
    discarded without a trace, because the one hard requirement here is
    that a broken stats collection must NEVER interrupt the evaluation
    worker. If you need to debug this class, temporarily add logging
    yourself — by design it stays out of the way otherwise.
    """

    def __init__(self, collection=None):
        self._collection = collection

    def _coll(self):
        return self._collection if self._collection is not None else db.get_collection("token_usage_stats")

    def record(
        self,
        project_title: str,
        key_alias: str,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        original_desc_length: int = 0,
        truncated_desc_length: int = 0,
        match_score=None,
        sent_to_telegram: bool = False,
        proposal_generated: bool = False,
        response_time_sec: float = 0.0,
    ) -> None:
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "project_title": project_title,
                "key_alias": key_alias,
                "prompt_tokens": prompt_tokens or 0,
                "output_tokens": output_tokens or 0,
                "total_tokens": total_tokens or 0,
                "original_desc_length": original_desc_length or 0,
                "truncated_desc_length": truncated_desc_length or 0,
                "match_score": match_score,
                "sent_to_telegram": bool(sent_to_telegram),
                "proposal_generated": bool(proposal_generated),
                "response_time_sec": round(response_time_sec, 3) if response_time_sec else 0.0,
            }
            self._coll().insert_one(entry)
        except Exception:
            pass  # completely silent and fail-safe, by explicit requirement


_token_tracker = TokenUsageTracker()


def _current_key_alias() -> str:
    """e.g. 'Gemini key #1' — reflects whichever key _generate() actually
    used for the call that just completed (module-level _current_key_index
    is updated during rotation before a successful return)."""
    return f"Gemini key #{_current_key_index + 1}"


def _extract_call_stats(response, elapsed_sec: float) -> dict:
    """Pulls token counts out of a single Gemini response + records how
    long that one call took. Used by score_project/draft_proposal to
    surface per-call metrics up to evaluate_project(), which sums them into
    one consolidated Evaluation for the whole project."""
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(usage, "total_token_count", 0) or 0,
        "response_time_sec": elapsed_sec,
        "key_alias": _current_key_alias(),
    }


_EMPTY_CALL_STATS = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "response_time_sec": 0.0, "key_alias": None}


def record_token_usage(title: str, evaluation: "Evaluation", sent_to_telegram: bool) -> None:
    """
    Public entry point for logging ONE consolidated analytics record for a
    fully-processed project. Called from main.py once the whole pipeline —
    Gemini evaluation AND the Telegram notification decision — has
    completed, since sent_to_telegram can only be known at that point.
    Takes a plain title string (rather than a Project object) so it works
    uniformly whether the caller has a scraped Project, a GitHub queue
    entry dict, or a parsed GitHub issue. Silent/fail-safe by construction
    (delegates entirely to TokenUsageTracker.record, which never raises).
    """
    _token_tracker.record(
        project_title=title,
        key_alias=evaluation.key_alias or "unknown",
        prompt_tokens=evaluation.prompt_tokens,
        output_tokens=evaluation.output_tokens,
        total_tokens=evaluation.total_tokens,
        original_desc_length=evaluation.original_desc_length,
        truncated_desc_length=evaluation.truncated_desc_length,
        match_score=(None if evaluation.ai_failed else evaluation.match_score),
        sent_to_telegram=sent_to_telegram,
        proposal_generated=evaluation.proposal_generated,
        response_time_sec=evaluation.response_time_sec,
    )



class ProjectScoreSchema(BaseModel):
    """
    Structured-output schema for score_project()'s Gemini call — passed as
    response_schema so the SDK enforces this shape at generation time
    (guaranteed valid JSON, no markdown fences, no conversational prefix
    like "Here is the JSON requested:"), rather than only asking for it in
    the prompt and hoping. ge/le constraints on match_score are included in
    the JSON schema Gemini receives too, nudging it away from out-of-range
    values on top of the type enforcement.
    """
    match_score: int = Field(
        ge=0, le=100,
        description="Integer score from 0 to 100 for how well the freelancer's skills match this project.",
    )
    reasoning: str = Field(
        description="One short sentence in English explaining the score.",
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Short list (0-5 items) of skills FROM THE FREELANCER'S OWN LIST above that this specific "
                     "project genuinely calls for — exact names as given in the skill list, not paraphrased. "
                     "Empty list if none apply.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Short list (0-5 items) of skills/technologies the PROJECT clearly requires that are NOT "
                     "in the freelancer's skill list above — i.e. real gaps. Empty list if the freelancer's "
                     "skills fully cover what the project needs.",
    )
    suggested_price: str = Field(
        description="The bid price to actually use, as a short string including currency, e.g. '$150' or "
                     "'$300-400'. Budget-adherence rule: if the client stated a budget in the project "
                     "description AND that budget is a reasonable fit for the project's real scope, this MUST "
                     "be that exact client-stated figure (or a value inside a stated range) — do not inflate "
                     "or round it. Only propose a DIFFERENT figure if the client's stated budget is severely "
                     "unrealistic for the described scope (e.g. an amount that couldn't cover even the "
                     "cheapest credible execution), in which case use your own realistic estimate instead. If "
                     "no budget was stated at all, use your own realistic estimate.",
    )
    delivery_days: int = Field(
        ge=1,
        description="The delivery time to actually use, in days. Timeline-adherence rule: if the client "
                     "stated a timeframe/deadline in the project description AND it is a reasonable fit for "
                     "the scope, this MUST be that exact client-stated number of days — do not pad it. Only "
                     "deviate if the client's stated timeframe is severely unrealistic for the described scope "
                     "(e.g. days for what clearly needs weeks), in which case use your own realistic estimate "
                     "instead. If no timeframe was stated at all, use your own realistic estimate.",
    )
    budget_timeline_adjusted: bool = Field(
        default=False,
        description="True ONLY if suggested_price and/or delivery_days above were set to something DIFFERENT "
                     "from what the client explicitly stated, because the stated value was judged severely "
                     "unrealistic. False whenever the client's stated figure was used as-is, or when the "
                     "client didn't state a figure at all.",
    )
    budget_timeline_note: str = Field(
        default="",
        description="Empty string if budget_timeline_adjusted is False. If True, ONE short, polite sentence "
                     "in Arabic justifying the deviation, suitable for showing the freelancer before they send "
                     "the proposal, e.g. 'الميزانية المعلنة غير كافية لنطاق العمل الموصوف، وهذا تقدير أقرب "
                     "للواقع.' — this is NEVER inserted into the proposal text itself (see draft_proposal's "
                     "price/duration rules), it is only surfaced to the human via Telegram.",
    )


def score_project(title: str, description: str) -> tuple:
    """
    Step 1: ask Gemini for a match score, reasoning, a suggested bid price,
    and an estimated delivery time. Returns (data_or_None, call_stats) —
    call_stats (see _extract_call_stats) is always populated, even on
    failure, so evaluate_project() can still record an accurate
    response_time_sec/key_alias for the analytics log.

    Uses response_schema=ProjectScoreSchema (structured output) so the SDK
    enforces schema-compliant JSON at generation time — response.parsed is
    the primary extraction path (an already-validated ProjectScoreSchema
    instance when it succeeds). response.text + parse_gemini_json() is kept
    as a DEFENSIVE fallback, not removed: structured output significantly
    reduces malformed responses but doesn't guarantee zero edge cases
    (e.g. a response truncated by GEMINI_SCORING_MAX_OUTPUT_TOKENS mid-
    generation can still leave `.parsed` unset) — enforcing at the API
    level AND validating defensively in the app are complementary, not
    redundant.
    """
    skills_list = ", ".join(config.MY_SKILLS)

    prompt = f"""
You are an expert freelance-bidding assistant. Compare the project below
against the freelancer's skill set, estimate how good a match it is, and
recommend a realistic bid.

Freelancer skills: {skills_list}

Project title: {title}
Project description: {description}

Evaluate the match and provide a match score, brief reasoning, which of the
freelancer's OWN listed skills genuinely apply to this project, which
skills/technologies the project needs that are NOT in the freelancer's
list (if any), a suggested bid price, and an estimated delivery time in
days.

BUDGET AND TIMELINE — STRICT ADHERENCE RULE:
First, check whether the project description itself states a client budget
and/or a client timeframe/deadline.
- If it does, and that figure is a REASONABLE fit for the scope described,
  you MUST use that exact client-stated figure as suggested_price/
  delivery_days — do not inflate, round, or pad it "to be safe."
- Only override the client's stated figure if it is SEVERELY unrealistic
  for the scope (not just "a bit tight" or "a bit generous") — in that
  case use your own realistic estimate instead, set
  budget_timeline_adjusted=true, and give one short polite Arabic
  justification in budget_timeline_note.
- If the client stated no figure at all, use your own realistic estimate
  and leave budget_timeline_adjusted=false.
"""
    start = time.time()
    try:
        response = _generate(
            prompt,
            response_schema=ProjectScoreSchema,
            temperature=config.GEMINI_SCORING_TEMPERATURE,
            max_output_tokens=config.GEMINI_SCORING_MAX_OUTPUT_TOKENS,
        )
        stats = _extract_call_stats(response, time.time() - start)

        # Primary path: SDK-validated structured output.
        data = None
        try:
            parsed = response.parsed
            if parsed is not None:
                data = parsed.model_dump()
        except Exception as parsed_exc:
            # Being defensive about accessing .parsed itself, not just its
            # value — an unexpected SDK/validation error here should fall
            # through to the text-based path below, not propagate.
            logger.warning("response.parsed access failed, falling back to text parsing: %s", parsed_exc)

        if data is None:
            # Fallback: either .parsed was None (e.g. truncated response)
            # or accessing it failed above — try our own robust extractor
            # on the raw text before giving up entirely.
            try:
                data = parse_gemini_json(response.text)
            except ValueError as parse_exc:
                # The API call itself succeeded (we have real token/timing
                # stats) — only parsing failed. Return those real stats
                # rather than falling through to the generic except below,
                # which would otherwise discard them.
                logger.error("%s", parse_exc)
                return None, stats

        if "match_score" not in data:
            return None, stats
        return data, stats
    except Exception as exc:
        logger.error("Gemini scoring call failed: %s", exc, exc_info=True)
        stats = dict(_EMPTY_CALL_STATS, response_time_sec=time.time() - start, key_alias=_current_key_alias())
        return None, stats


class _BatchScoreItem(BaseModel):
    index: int = Field(
        description="The 0-based index of this project exactly as given in the input list — "
                    "used to map each result back to its project even if the model's array "
                    "order doesn't exactly match the input order.",
    )
    match_score: int = Field(
        ge=0, le=100,
        description="Integer score from 0 to 100 for how well the freelancer's skills match this project.",
    )
    reasoning: str = Field(
        description="One short sentence in English explaining the score.",
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Short list (0-5 items) of skills FROM THE FREELANCER'S OWN LIST that this specific "
                     "project genuinely calls for — exact names as given in the skill list, not paraphrased. "
                     "Empty list if none apply.",
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Short list (0-5 items) of skills/technologies THIS project clearly requires that are NOT "
                     "in the freelancer's skill list — i.e. real gaps. Empty list if fully covered.",
    )
    suggested_price: str = Field(
        description="A realistic recommended bid price/budget for this project's scope, as a short string "
                     "including currency, e.g. '$150' or '$300-400'.",
    )
    delivery_days: int = Field(
        ge=1,
        description="Realistic estimated number of days to complete the project based on its scope.",
    )


class BatchScoreSchema(BaseModel):
    """response_schema for score_projects_batch() — one ProjectScoreSchema-
    shaped entry per input project, tagged with `index` so results can be
    matched back to their project regardless of array order."""
    results: List[_BatchScoreItem] = Field(
        description="Exactly one result per input project, each tagged with its `index`.",
    )


def score_projects_batch(projects: List[dict]) -> tuple:
    """
    Batched counterpart to score_project(): scores MULTIPLE projects in a
    SINGLE Gemini call instead of one call per project — the single
    biggest lever for staying inside a free-tier daily request quota (RPD)
    when several new projects appear in the same poll cycle. `projects` is
    a list of {"title": str, "description": str} dicts (already truncated
    by the caller), in a fixed order that the caller cares about.

    Returns (results_by_index, call_stats):
      - results_by_index: {index: {"match_score", "reasoning",
        "suggested_price", "delivery_days"}}, one entry per project Gemini
        actually returned a result for. An index MISSING from this dict
        means Gemini dropped that entry — rare, but structured output
        doesn't guarantee every array item survives generation (e.g. the
        response hitting max_output_tokens mid-array). The caller MUST
        treat a missing index the same as a scoring failure for that one
        project specifically, not silently skip it.
      - call_stats: aggregate token/timing stats for the WHOLE call —
        Gemini doesn't report a per-item breakdown within one batched
        response, so callers apportion this across the batch themselves
        for analytics (see evaluate_projects_batch).

    Returns (None, call_stats) if the call itself fails outright (network
    error, every key exhausted, etc.) — same convention as score_project().
    """
    if not projects:
        return {}, dict(_EMPTY_CALL_STATS)

    skills_list = ", ".join(config.MY_SKILLS)
    numbered_projects = "\n\n".join(
        f"[Project index={i}]\nTitle: {p['title']}\nDescription: {p['description']}"
        for i, p in enumerate(projects)
    )
    prompt = f"""
You are an expert freelance-bidding assistant. Below are {len(projects)}
separate freelance projects, each labeled with its own index. Evaluate
EACH ONE independently against the freelancer's skill set below — do not
let one project's content influence another's score.

Freelancer skills: {skills_list}

{numbered_projects}

For EVERY project index above, provide a match score, brief reasoning,
which of the freelancer's OWN listed skills genuinely apply to it, which
skills/technologies it needs that are NOT in the freelancer's list (if
any), a suggested bid price, and an estimated delivery time in days.
Return exactly {len(projects)} results — one per index, none skipped or repeated.
"""
    start = time.time()
    try:
        # Scale the output budget with batch size: each result needs
        # roughly the same tokens as a single score_project() call, plus a
        # small buffer for JSON array overhead. Capped defensively so an
        # unexpectedly large batch can't request an unreasonable budget.
        max_tokens = min(config.GEMINI_SCORING_MAX_OUTPUT_TOKENS * len(projects) + 200, 8192)
        response = _generate(
            prompt,
            response_schema=BatchScoreSchema,
            temperature=config.GEMINI_SCORING_TEMPERATURE,
            max_output_tokens=max_tokens,
        )
        stats = _extract_call_stats(response, time.time() - start)

        data = None
        try:
            parsed = response.parsed
            if parsed is not None:
                data = parsed.model_dump()
        except Exception as parsed_exc:
            logger.warning("Batch response.parsed access failed, falling back to text parsing: %s", parsed_exc)

        if data is None:
            try:
                data = parse_gemini_json(response.text)
            except ValueError as parse_exc:
                logger.error("%s", parse_exc)
                return None, stats

        raw_results = data.get("results") or []
        results_by_index: Dict[int, dict] = {}
        for item in raw_results:
            try:
                idx = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            results_by_index[idx] = item

        if len(results_by_index) < len(projects):
            missing = [i for i in range(len(projects)) if i not in results_by_index]
            logger.warning(
                "Batch scoring returned %s/%s result(s) — missing index(es) %s "
                "will be treated as scoring failures for those specific projects",
                len(results_by_index), len(projects), missing,
            )

        return results_by_index, stats
    except Exception as exc:
        logger.error("Gemini BATCH scoring call failed: %s", exc, exc_info=True)
        stats = dict(_EMPTY_CALL_STATS, response_time_sec=time.time() - start, key_alias=_current_key_alias())
        return None, stats


def draft_proposal(
    title: str,
    description: str,
    budget: Optional[str] = None,
    client_info: Optional[dict] = None,
) -> tuple:
    """
    Step 2 (only called if score >= threshold): draft an Arabic proposal
    following Mostaql's professional-proposal standards. Returns
    (text_or_None, call_stats) — see score_project's docstring for why.

    client_info (see scraper.parse_client_info) is OPTIONAL context used
    only to adjust TONE — e.g. a slightly more assured, relationship-
    minded closing for an established, well-reviewed client vs. a warmer,
    more reassuring one for a brand-new client with no history yet. It is
    NEVER used to change what work is promised, and the prompt explicitly
    forbids stating or implying anything about the client's rating/review
    count/history in the proposal text itself (that would read as odd or
    presumptuous to the client) — it only shapes the freelancer's own tone.

    IMPORTANT — price/delivery time are DELIBERATELY NOT passed to this
    prompt and DELIBERATELY NOT mentioned anywhere in the proposal text.
    suggested_price/delivery_days (computed by score_project) still exist
    and are still sent to Telegram as their own dedicated fields — this
    function's prompt just never asks Gemini to restate them inside the
    proposal body, and explicitly forbids it from doing so on its own
    initiative, per Mostaql's rules against quoting price/duration inside
    proposal text (that information belongs only in the platform's
    dedicated bid fields, not embedded in free text).
    """
    skills_list = ", ".join(config.MY_SKILLS)
    budget_line = f"\n(للسياق فقط، لا تذكره: ميزانية العميل المعلنة هي {budget})" if budget else ""

    # Tone guidance derived from client_info — advisory only, never a claim
    # about the client that ends up IN the proposal text (see the explicit
    # rule below forbidding that). Fails open to no guidance at all if
    # client_info is missing/empty/inconclusive, which is the common case.
    tone_line = ""
    if client_info:
        rating = client_info.get("rating")
        reviews_count = client_info.get("reviews_count")
        if client_info.get("is_new"):
            tone_line = (
                "\n(ملاحظة أسلوب داخلية فقط، لا تُدرَج في النص: هذا عميل جديد على "
                "المنصة أو بدون سجل تقييمات — اكتب بأسلوب مرحّب وواضح يبني الثقة "
                "من الصفر، دون أي إشارة إلى كونه عميلاً جديداً.)"
            )
        elif rating is not None and rating >= config.STRONG_CLIENT_RATING_THRESHOLD and (reviews_count or 0) >= 3:
            tone_line = (
                "\n(ملاحظة أسلوب داخلية فقط، لا تُدرَج في النص: هذا عميل موثوق وله "
                "سجل تعاملات جيد — يمكنك الكتابة بنبرة أكثر ثقة ومهنية مباشرة، "
                "دون أي إشارة إلى تقييمه أو سجله.)"
            )

    prompt = f"""
أنت مستقل خبير تكتب عرضك الشخصي لتقديمه على مشروع في منصة مستقل (Mostaql).
مهاراتك الفعلية (استخدم منها فقط ما يخدم هذا المشروع تحديداً، وتجاهل الباقي
تماماً): {skills_list}

عنوان المشروع: {title}
وصف المشروع: {description}{budget_line}{tone_line}

اكتب عرضاً شخصياً بصوت مستقل بشري حقيقي وخبير، باللغة العربية الفصحى،
يغطي هذه العناصر بشكل متدفق وطبيعي (بدون كتابة عناوين الأقسام، وبدون أن
يبدو كقالب جامد مكرر):

- تحية ومقدمة موجزة تعرّف بك كمستقل مختص.
- إثبات فهم دقيق ومحدد لما يحتاجه هذا العميل تحديداً كما ورد في وصف
  المشروع فعلياً — وليس فهماً عاماً ينطبق على أي مشروع مشابه.
- خطة عمل مختصرة (2-4 خطوات) تُظهر منهجية واضحة تبني الثقة.
- لماذا أنت الخيار المناسب: اذكر فقط الخبرات المرتبطة مباشرة بما يطلبه
  هذا المشروع تحديداً. لا تسرد كل مهاراتك، ولا تذكر أي تقنية (بايثون،
  OOP، فلاتر، الخ) إلا إذا كانت مطلوبة صراحة في وصف المشروع أو ضرورية
  تقنياً وبشكل مباشر لحل المشكلة المطروحة — ذِكر تقنيات غير ذات صلة
  "لحشو" العرض ممنوع تماماً.
- خاتمة احترافية تدعو العميل للتواصل أو طرح الأسئلة.

قواعد صارمة وإلزامية:
1. ممنوع منعاً باتاً ذكر أي رقم أو إشارة تخص السعر، التكلفة، الميزانية،
   أو مدة التسليم/عدد الأيام في أي مكان من نص العرض — تحت أي ظرف ولأي
   سبب. هذه المعلومات موجودة في حقول منفصلة خارج نص العرض على المنصة،
   وذكرها داخل النص يخالف قواعد مستقل. لا تكتب حتى عبارات عامة تلمّح
   لذلك مثل "سعر مناسب" أو "خلال مدة قصيرة" — تجنب الموضوع كلياً.
2. ممنوع حشو المهارات أو ذكرها بشكل روتيني في كل عرض — فقط ما يرتبط
   تحديداً بهذا المشروع كما هو موضح أعلاه.
3. اكتب بأسلوب إنساني طبيعي ومرن كما يكتب مستقل محترف حقيقي، وليس بأسلوب
   جامد أو نمطي يبدو آلياً أو مولداً تلقائياً. تجنب الجمل الجاهزة
   المكررة والعبارات الفضفاضة.
4. لا تبالغ ولا تعد بما لا يمكنك تنفيذه بدقة — كن شفافاً وواقعياً.
5. طوله لا يتجاوز 180 كلمة.
6. لا تضع أي عناوين أقسام أو تنسيق ماركداون، فقط نص العرض جاهزاً للنسخ
   مباشرة.
7. ممنوع الإشارة من قريب أو بعيد إلى تقييم العميل أو عدد تقييماته أو كونه
   عميلاً جديداً أو له سجل أعمال سابق أم لا — أي ملاحظة أسلوب داخلية وردت
   أعلاه هي لضبط نبرتك أنت فقط، ولا يجوز أن تظهر كإشارة أو تلميح في نص
   العرض نفسه.
"""
    start = time.time()
    try:
        # Deliberately NOT passing temperature/max_output_tokens here (unlike
        # score_project) — this call needs natural, varied prose per the
        # "human, non-robotic tone" requirement; a low temperature would
        # make every proposal read identically, and a 300-token cap could
        # cut off a well-formed ~180-word Arabic proposal mid-sentence.
        response = _generate(prompt)
        stats = _extract_call_stats(response, time.time() - start)
        text = response.text.strip()
        return (text if text else None), stats
    except Exception as exc:
        logger.error("Gemini proposal drafting failed: %s", exc, exc_info=True)
        stats = dict(_EMPTY_CALL_STATS, response_time_sec=time.time() - start, key_alias=_current_key_alias())
        return None, stats


class _ScreeningAnswerItem(BaseModel):
    question: str = Field(description="The screening question, repeated back EXACTLY as given in the input.")
    answer: str = Field(
        description="A specific, technically accurate answer to this exact question, grounded in the actual "
                     "project description and the freelancer's real skills — never generic boilerplate that "
                     "could apply to any project. Plain text, no markdown, ready to paste directly into a "
                     "platform text box. Concise (roughly 2-6 sentences) but substantive enough to demonstrate "
                     "real understanding, not a one-liner.",
    )


class ScreeningAnswersSchema(BaseModel):
    answers: List[_ScreeningAnswerItem] = Field(
        default_factory=list,
        description="One entry per input question, in the SAME order as given.",
    )


def draft_screening_answers(
    title: str,
    description: str,
    questions: List[str],
) -> tuple:
    """
    Drafts one specific, technically-grounded answer per explicit client
    screening question (see scraper.parse_screening_questions) — e.g. "How
    do you propose to implement this?", "What architecture will you use?".
    Only called when a project actually has screening questions (see
    _finalize_score_result), so this adds zero extra Gemini cost to every
    other project.

    Returns (list_of_{"question","answer"}_dicts, call_stats) — mirrors
    score_project()/draft_proposal()'s (data_or_empty, stats) contract.
    Never raises: on any failure, returns ([], stats) so a screening-
    question drafting problem can never block the rest of the evaluation
    (the match score and main proposal are computed independently and
    already returned by the time this runs).

    Deliberately a SEPARATE Gemini call from draft_proposal() rather than
    folded into one bigger prompt: screening answers need to be long-form,
    fact-specific, and individually addressable (Telegram formats each one
    as its own copy-paste block — see notifier.build_screening_section),
    which is a different shape of output than the single flowing cover-
    letter paragraph draft_proposal() produces, and mixing the two in one
    prompt risks the model blending proposal prose into an answer or vice
    versa.
    """
    if not questions:
        return [], dict(_EMPTY_CALL_STATS)

    skills_list = ", ".join(config.MY_SKILLS)
    numbered_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""
You are an expert freelancer answering a client's explicit screening
questions on a project-bidding platform, so the client can evaluate your
proposal. These answers appear separately from the main cover letter, so
each one must fully stand on its own.

Freelancer's real skills (use only what's actually relevant to each
question — never claim something not in this list): {skills_list}

Project title: {title}
Project description: {description}

The client asked these screening questions (answer EVERY one, in the same
order, and repeat each question exactly as given):
{numbered_questions}

For each question, write a specific, technically accurate, and confident
answer grounded in the actual project description above — not a generic
answer that could apply to any project. If a question asks about
implementation approach or architecture, name concrete steps/technologies
that are genuinely appropriate for THIS project's stated requirements. Do
not invent capabilities outside the freelancer's real skill list. Do not
mention price, budget, or delivery time/deadline in any answer. Plain
text only, no markdown formatting, ready to paste directly into a text
box.
"""
    start = time.time()
    try:
        response = _generate(
            prompt,
            response_schema=ScreeningAnswersSchema,
            max_output_tokens=config.GEMINI_SCREENING_MAX_OUTPUT_TOKENS,
        )
        stats = _extract_call_stats(response, time.time() - start)

        data = None
        try:
            parsed = response.parsed
            if parsed is not None:
                data = parsed.model_dump()
        except Exception as parsed_exc:
            logger.warning("Screening-answers response.parsed access failed, falling back to text parsing: %s", parsed_exc)

        if data is None:
            try:
                data = parse_gemini_json(response.text)
            except ValueError as parse_exc:
                logger.error("%s", parse_exc)
                return [], stats

        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, list):
            return [], stats

        result = []
        for item in answers:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or "").strip()
            a = str(item.get("answer") or "").strip()
            if q and a:
                result.append({"question": q, "answer": a})
        return result, stats
    except Exception as exc:
        logger.error("Gemini screening-answers call failed: %s", exc, exc_info=True)
        stats = dict(_EMPTY_CALL_STATS, response_time_sec=time.time() - start, key_alias=_current_key_alias())
        return [], stats


def _ai_failed_evaluation(reason: str, original_desc_length: int, truncated_desc_length: int, score_stats: dict) -> Evaluation:
    """Shared 'scoring didn't produce a usable result' Evaluation, used by
    both evaluate_project() and evaluate_projects_batch() so this shape
    exists in exactly one place."""
    return Evaluation(
        match_score=0.0,
        reasoning=reason,
        ai_failed=True,
        original_desc_length=original_desc_length,
        truncated_desc_length=truncated_desc_length,
        prompt_tokens=score_stats["prompt_tokens"],
        output_tokens=score_stats["output_tokens"],
        total_tokens=score_stats["total_tokens"],
        response_time_sec=round(score_stats["response_time_sec"], 3),
        key_alias=score_stats["key_alias"],
    )


def _finalize_score_result(
    title: str,
    full_description: str,
    budget: Optional[str],
    score_data: dict,
    score_stats: dict,
    original_desc_length: int,
    truncated_desc_length: int,
    client_info: Optional[dict] = None,
    screening_questions: Optional[List[str]] = None,
) -> Evaluation:
    """
    Shared tail logic that turns a raw score_data dict — regardless of
    whether it came from a fresh score_project()/score_projects_batch()
    call or a ScoreCache hit — into a complete Evaluation: validates
    match_score, and if it clears MATCH_THRESHOLD, drafts a proposal
    (always fresh, never cached — see ScoreCache's docstring) using the
    FULL description at its own, longer truncation length, AND (also only
    above threshold, also always fresh) drafts answers for any explicit
    screening_questions the project has (see draft_screening_answers) — no
    point spending a screening-answers call on a project that won't be
    notified anyway. Used by both evaluate_project() and
    evaluate_projects_batch() so this logic exists in exactly one place.
    """
    try:
        raw_score = float(score_data.get("match_score", 0))
    except (TypeError, ValueError):
        logger.error(
            "Gemini returned a non-numeric match_score (%r) for '%s' — "
            "treating as a scoring failure rather than crashing or "
            "silently reporting 0%%",
            score_data.get("match_score"), title,
        )
        return _ai_failed_evaluation(
            "AI scoring unavailable (malformed match_score in response).",
            original_desc_length, truncated_desc_length, score_stats,
        )
    # Round to a whole number ONCE, immediately, before any comparison or
    # logging happens anywhere downstream (here, and in main.py) — see the
    # detailed rationale that used to live inline here: this guarantees
    # the score used in the threshold check, the one shown in logs, and
    # the one sent to Telegram are always the exact same number.
    score = float(round(raw_score))

    reasoning = score_data.get("reasoning", "")

    def _safe_skill_list(raw) -> List[str]:
        # Defensive: Gemini's structured output enforces the schema at the
        # top level, but a ScoreCache hit replays a plain dict we wrote
        # ourselves — still worth guarding against anything other than a
        # list of strings ending up here rather than crashing downstream
        # Telegram formatting.
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if item and str(item).strip()]

    matched_skills = _safe_skill_list(score_data.get("matched_skills"))
    missing_skills = _safe_skill_list(score_data.get("missing_skills"))

    suggested_price = score_data.get("suggested_price")
    # Defensive: Gemini occasionally returns this as a number instead of
    # the requested string (e.g. 150 instead of "$150") — coerce so
    # notifier.py's Telegram formatting always gets a clean string rather
    # than a raw Python repr.
    if suggested_price is not None and not isinstance(suggested_price, str):
        suggested_price = str(suggested_price)
    delivery_days = score_data.get("delivery_days")
    try:
        delivery_days = int(delivery_days) if delivery_days is not None else None
    except (TypeError, ValueError):
        delivery_days = None

    budget_timeline_adjusted = bool(score_data.get("budget_timeline_adjusted", False))
    budget_timeline_note = str(score_data.get("budget_timeline_note") or "").strip()

    proposal = None
    proposal_generated = False
    proposal_stats = dict(_EMPTY_CALL_STATS)
    screening_answers: List[dict] = []
    screening_stats = dict(_EMPTY_CALL_STATS)
    # Computed ONCE here and reused for both the decision below and the
    # returned Evaluation, so a threshold that ramps up mid-batch under
    # quota pressure can't produce an inconsistent picture for one project
    # (e.g. deciding with one value but reporting against another).
    threshold = get_effective_match_threshold()
    # >= : a score exactly equal to the threshold must be treated as a match,
    # not skipped. This must match main.py's notification-gate comparison
    # exactly (main.py now compares against evaluation.effective_threshold,
    # not the static config.MATCH_THRESHOLD, for exactly this reason) —
    # since proposal_ar only gets set here, if that gate used a different
    # bar than this one, a boundary-score project could clear main.py's
    # check but still have no proposal to send, or vice versa.
    if score >= threshold:
        proposal_generated = True  # a drafting call was executed, regardless of its outcome below
        # Proposal drafting uses its OWN (longer) truncation of the FULL
        # description — deliberately re-truncated here rather than reusing
        # whatever shorter text scoring used, since a cache hit means no
        # scoring-truncated text was even computed this time around.
        proposal_desc = smart_truncate_description(full_description, max_length=config.GEMINI_DESCRIPTION_MAX_CHARS)
        # Deliberately NOT passing suggested_price/delivery_days here — see
        # draft_proposal()'s docstring. They still flow to Telegram via the
        # Evaluation object below, just never into the proposal text itself.
        proposal, proposal_stats = draft_proposal(title, proposal_desc, budget, client_info)

        if screening_questions:
            screening_answers, screening_stats = draft_screening_answers(
                title, proposal_desc, screening_questions,
            )

    return Evaluation(
        match_score=score,
        reasoning=reasoning,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggested_price=suggested_price,
        delivery_days=delivery_days,
        proposal_ar=proposal,
        budget_timeline_adjusted=budget_timeline_adjusted,
        budget_timeline_note=budget_timeline_note,
        screening_answers=screening_answers,
        original_desc_length=original_desc_length,
        truncated_desc_length=truncated_desc_length,
        prompt_tokens=score_stats["prompt_tokens"] + proposal_stats["prompt_tokens"] + screening_stats["prompt_tokens"],
        output_tokens=score_stats["output_tokens"] + proposal_stats["output_tokens"] + screening_stats["output_tokens"],
        total_tokens=score_stats["total_tokens"] + proposal_stats["total_tokens"] + screening_stats["total_tokens"],
        response_time_sec=round(
            score_stats["response_time_sec"] + proposal_stats["response_time_sec"] + screening_stats["response_time_sec"], 3,
        ),
        # Whichever key was actually used LAST (screening call if it ran,
        # else proposal call, else the scoring call) — usually all the same
        # key anyway. A cache hit leaves score_stats["key_alias"] as None,
        # so this naturally falls back further down the chain, or None if
        # nothing actually ran (below-threshold cache hit).
        key_alias=screening_stats["key_alias"] or proposal_stats["key_alias"] or score_stats["key_alias"],
        proposal_generated=proposal_generated,
        effective_threshold=threshold,
    )


def evaluate_project(
    title: str,
    description: str,
    budget: Optional[str] = None,
    tags: Optional[List[str]] = None,
    client_info: Optional[dict] = None,
    screening_questions: Optional[List[str]] = None,
) -> Evaluation:
    """
    Full pipeline for one project:
      0. Local tag pre-filter — zero-cost skip if tags exist and don't
         overlap with config.MY_SKILLS at all.
      1. Score-cache lookup — zero-cost skip of the Gemini scoring call if
         this exact (title, description, MY_SKILLS) was already scored
         before (see ScoreCache).
      2. Score it via Gemini if not cached (including price/duration
         estimates, and budget/timeline adherence — see
         ProjectScoreSchema), using the SHORTER scoring-specific
         truncation.
      3. If it clears the threshold, draft a proposal too, using the full
         (longer-truncated) description — client_info (see
         scraper.parse_client_info), if provided, lets draft_proposal()
         adjust TONE ONLY (see its docstring); it never changes scoring.
         Also, only above threshold, drafts an answer for each entry in
         screening_questions (see scraper.parse_screening_questions /
         draft_screening_answers), if any were found on the project.
    Always returns an Evaluation object — never raises — so main.py's loop
    can rely on it unconditionally. Also populates the analytics fields
    (token counts, response time, desc lengths, etc.) that main.py passes
    to ai_agent.record_token_usage() once it also knows sent_to_telegram.
    """
    original_desc_length = len(description) if description else 0

    if not local_skill_prefilter(tags, title, description):
        logger.info(
            "Local pre-filter: no skill overlap found (tags=%s) — "
            "skipping Gemini entirely (zero API cost)",
            tags,
        )
        return Evaluation(
            match_score=0.0,
            reasoning="No matching skills found locally (filtered, zero API cost).",
            original_desc_length=original_desc_length,
            truncated_desc_length=0,
        )

    # Scoring uses a SHORTER truncation than proposal drafting — see
    # config.GEMINI_SCORING_DESCRIPTION_MAX_CHARS's comment. The cache
    # lookup below uses the FULL, untruncated `description` as its key —
    # a project's identity shouldn't depend on this truncation length.
    scoring_desc = smart_truncate_description(description, max_length=config.GEMINI_SCORING_DESCRIPTION_MAX_CHARS)
    truncated_desc_length = len(scoring_desc) if scoring_desc else 0

    cached = _score_cache.get(title, description)
    if cached is not None:
        logger.info(
            "Score cache HIT for '%s' — identical content already scored, "
            "skipping the Gemini scoring call entirely",
            title,
        )
        score_data, score_stats = cached, dict(_EMPTY_CALL_STATS)
    else:
        score_data, score_stats = score_project(title, scoring_desc)
        if score_data is not None:
            _score_cache.set(title, description, score_data)

    if score_data is None:
        return _ai_failed_evaluation(
            "AI scoring unavailable (error).", original_desc_length, truncated_desc_length, score_stats,
        )

    return _finalize_score_result(
        title, description, budget, score_data, score_stats, original_desc_length, truncated_desc_length,
        client_info=client_info, screening_questions=screening_questions,
    )


def evaluate_projects_batch(projects: List[dict]) -> List[Evaluation]:
    """
    Batched counterpart to evaluate_project(): scores ALL given projects in
    ONE Gemini call (via score_projects_batch), then drafts a proposal
    individually — still one call each — for whichever ones clear
    MATCH_THRESHOLD. Proposal drafting is deliberately NOT batched: free-
    form prose for several unrelated projects in a single call risks
    quality bleed between them (tone/details from one leaking into
    another's proposal), so it stays one call per accepted match. Since
    match_score >= MATCH_THRESHOLD is usually the minority of any batch,
    this still collapses what used to be N scoring calls into 1.

    `projects` is a list of dicts shaped like {"title", "description",
    "budget", "tags"}. Returns a list of Evaluation objects in the EXACT
    same order/length as the input, so callers can zip() it against their
    own project objects.

    Two zero-Gemini-cost skips happen before anything is sent to Gemini,
    per project:
      1. Local tag pre-filter (identical rule to evaluate_project()).
      2. Score-cache lookup (see ScoreCache) — a project whose exact
         (title, description, MY_SKILLS) was already scored before skips
         the batch entirely for that project.
    Only the remaining projects are sent to Gemini in ONE scoring call,
    using the shorter scoring-specific truncation (see config.py).
    """
    n = len(projects)
    results: List[Optional[Evaluation]] = [None] * n
    original_lengths = [len(p.get("description") or "") for p in projects]
    # Scoring uses the SHORTER truncation — see config.GEMINI_SCORING_DESCRIPTION_MAX_CHARS.
    scoring_descs = [
        smart_truncate_description(p.get("description") or "", max_length=config.GEMINI_SCORING_DESCRIPTION_MAX_CHARS)
        for p in projects
    ]

    # batch_positions[i] = this project's position within `to_score` (the
    # subset actually sent to Gemini), or None if it was already resolved
    # below (local pre-filter or cache hit) without ever needing an API call.
    to_score: List[dict] = []
    batch_positions: List[Optional[int]] = [None] * n

    for i, p in enumerate(projects):
        tags = p.get("tags") or []
        if not local_skill_prefilter(tags, p.get("title"), p.get("description")):
            logger.info(
                "Local pre-filter: no skill overlap found (tags=%s) for '%s' — "
                "skipping Gemini entirely (zero API cost)",
                tags, p.get("title"),
            )
            results[i] = Evaluation(
                match_score=0.0,
                reasoning="No matching skills found locally (filtered, zero API cost).",
                original_desc_length=original_lengths[i],
                truncated_desc_length=0,
            )
            continue

        # Cache lookup uses the FULL, untruncated description — a
        # project's identity shouldn't depend on this batch's truncation.
        cached = _score_cache.get(p["title"], p.get("description") or "")
        if cached is not None:
            logger.info(
                "Score cache HIT for '%s' — identical content already "
                "scored, skipping this project's slot in the batch call entirely",
                p["title"],
            )
            results[i] = _finalize_score_result(
                p["title"], p.get("description") or "", p.get("budget"),
                cached, dict(_EMPTY_CALL_STATS),
                original_lengths[i], len(scoring_descs[i] or ""),
                client_info=p.get("client_info"),
                screening_questions=p.get("screening_questions"),
            )
            continue

        batch_positions[i] = len(to_score)
        to_score.append({"title": p["title"], "description": scoring_descs[i]})

    if to_score:
        score_results, batch_stats = score_projects_batch(to_score)

        # Apportion the ONE batch call's aggregate stats evenly across the
        # projects actually sent to Gemini — Gemini doesn't report a
        # per-item token breakdown within a single batched response, so
        # this is a reasonable approximation for analytics, not exact
        # per-project accounting. key_alias isn't numeric, so it's shared
        # as-is rather than divided.
        share = max(len(to_score), 1)
        per_item_stats = dict(batch_stats or _EMPTY_CALL_STATS)
        per_item_stats["prompt_tokens"] = (per_item_stats.get("prompt_tokens") or 0) // share
        per_item_stats["output_tokens"] = (per_item_stats.get("output_tokens") or 0) // share
        per_item_stats["total_tokens"] = (per_item_stats.get("total_tokens") or 0) // share
        per_item_stats["response_time_sec"] = (per_item_stats.get("response_time_sec") or 0.0) / share

        for i in range(n):
            pos = batch_positions[i]
            if pos is None:
                continue  # already filled in above (pre-filter or cache hit)

            score_data = None if score_results is None else score_results.get(pos)
            if score_data is None:
                results[i] = _ai_failed_evaluation(
                    "AI scoring unavailable (error).",
                    original_lengths[i], len(scoring_descs[i] or ""), per_item_stats,
                )
                continue

            # Cache this fresh result under the FULL, untruncated
            # description — so a future retry of this exact project (e.g.
            # via the GitHub fallback queue) can skip Gemini entirely.
            _score_cache.set(projects[i]["title"], projects[i].get("description") or "", score_data)

            results[i] = _finalize_score_result(
                projects[i]["title"], projects[i].get("description") or "", projects[i].get("budget"),
                score_data, per_item_stats, original_lengths[i], len(scoring_descs[i] or ""),
                client_info=projects[i].get("client_info"),
                screening_questions=projects[i].get("screening_questions"),
            )

    return results
