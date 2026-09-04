"""
ai_agent.py
-----------
Uses Google's Gen AI SDK (`from google import genai`) to:
  1. Score how well a project matches the freelancer's skill set (0-100),
     and estimate a suggested bid price and delivery time.
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's details.

Three reliability features on top of that:

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

API KEY ROTATION (survive per-key free-tier quota limits)
-------------------------------------------------------------------
`config.GEMINI_API_KEYS` is a list. On a 429 / RESOURCE_EXHAUSTED quota
error, `_generate()` immediately rotates to the next key, rebuilds the
client, and retries — no backoff wait, since a different key's quota is
unrelated to how long we wait on this one.

TRANSIENT-ERROR RETRY (504 Gateway Timeout / DEADLINE_EXCEEDED / 503)
-------------------------------------------------------------------
These are different from quota errors: switching keys doesn't fix a
timed-out gateway, so `_generate()` instead retries the SAME key with
short exponential backoff, up to config.GEMINI_MAX_TRANSIENT_RETRIES times,
before giving up on that key and (only then) also rotating to the next key
as a last resort. Any error that is neither a quota error nor a recognized
transient error is raised immediately without retrying or rotating, so a
genuinely broken prompt/request doesn't waste time or keys.

Both retry paths are bounded (at most n_keys * (1 + max_transient_retries)
attempts total), so `_generate()` always eventually returns or raises —
it cannot loop forever. score_project()/draft_proposal() catch whatever it
raises and return None, and evaluate_project() turns that into a safe
fallback Evaluation (match_score=0.0, suggested_price=None,
delivery_days=None) rather than letting the exception propagate — so one
bad project can never take down main.py's loop.

SDK NOTE: single stable model (config.GEMINI_MODEL, default
"gemini-3.5-flash"), no model fallback chain, per current requirements.

TIMEOUT CAVEAT: google-genai's http_options timeout has known upstream
reliability issues (requests can occasionally hang despite a timeout being
set — see googleapis/python-genai#1893, #911). We still set it below as a
first line of defense, but the real guarantee against a permanent hang is
main.py's CYCLE_TIMEOUT watchdog (ThreadPoolExecutor + future.result
timeout), which is untouched by this file and must stay in place.
"""

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
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import config
import db

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


class AllKeysRateLimited(Exception):
    """
    Raised by _generate() when EVERY configured key is already at its local
    rate-limit cap (config.GEMINI_MAX_RPM_PER_KEY requests in the last 60s)
    — meaning _generate() bypassed the Gemini API entirely and made ZERO
    network calls for this attempt. This is the PROACTIVE counterpart to a
    real 429: instead of firing a request we already know will likely be
    rejected (or, worse, one that pushes Google's own rate limiter into
    penalizing us further), we skip it and let the caller fall back
    immediately. score_project()/draft_proposal() catch this via their
    existing broad `except Exception`, so no special handling is required
    there — it flows into Evaluation(ai_failed=True) exactly like a real
    API failure would.
    """
    pass


class AllKeysExhaustedError(RuntimeError):
    """
    Raised by _generate() when at least one (key, model) attempt was
    actually made — as opposed to AllKeysRateLimited, where none were —
    and every single one of them failed with a real error (429, transient,
    or otherwise). Distinct from AllKeysRateLimited so logs/callers can
    tell "we never even tried, everything was locally rate-limited" apart
    from "we tried the entire fallback chain and it all genuinely failed".
    The original underlying exception is chained via `__cause__` (see
    `raise ... from last_exc`), so nothing about the root cause is lost.
    """
    pass


# Global mutex: guarantees only ONE Gemini API call is ever in flight at a
# time, process-wide, regardless of which thread calls _generate() — not
# just relying on "only the consumer thread happens to call this today".
# Cheap (the critical section is bounded by the same per-task watchdog
# main.py already enforces) and directly prevents concurrent callers from
# multiplying instantaneous demand against the RPM limits, which is the
# actual failure mode a race between multiple callers would produce.
_generate_lock = threading.Lock()


class KeyRateLimiter:
    """
    In-memory sliding-window rate tracker. Originally one window per API
    key index; now generalized to track ANY hashable "bucket" — in
    practice a (key_index, model_name) tuple, since Gemini's actual RPM
    quotas are per-model, not shared across every model used under one
    key. Enforces a hard cap of `max_per_minute` requests (default, or an
    explicit per-call override) in any trailing 60-second window —
    proactively, BEFORE a request is sent, rather than reactively after
    Google returns a 429.

    Thread-safety: in this codebase _generate() is only ever called from
    the single consumer thread (see main.py's producer/consumer split), so
    a lock isn't strictly required for correctness today — but it's kept
    here anyway since this class represents shared mutable state and the
    cost of the lock is negligible, in case that assumption ever changes.
    """

    def __init__(self, default_max_per_minute: int):
        self.default_max_per_minute = default_max_per_minute
        self._timestamps: dict = {}  # bucket_key -> [timestamp, ...]
        self._lock = threading.Lock()

    def _prune_locked(self, bucket_key, now: float) -> list:
        """Must be called while holding self._lock. Drops timestamps older
        than the 60s window and returns the (mutated in place) list."""
        window_start = now - 60.0
        ts = self._timestamps.setdefault(bucket_key, [])
        while ts and ts[0] < window_start:
            ts.pop(0)
        return ts

    def available(self, bucket_key, max_per_minute: int = None) -> bool:
        """True if this bucket has capacity for at least one more request
        right now (fewer than max_per_minute requests in the last 60s)."""
        cap = max_per_minute if max_per_minute is not None else self.default_max_per_minute
        with self._lock:
            ts = self._prune_locked(bucket_key, time.time())
            return len(ts) < cap

    def record(self, bucket_key) -> None:
        """Records a request attempt against this bucket's window. Call
        this immediately before actually making the API call (not after),
        so capacity is reserved even if the call is still in flight."""
        with self._lock:
            ts = self._prune_locked(bucket_key, time.time())
            ts.append(time.time())

    def remaining(self, bucket_key, max_per_minute: int = None) -> int:
        """How many more requests this bucket can make right now before
        hitting the cap. Useful for logging/diagnostics."""
        cap = max_per_minute if max_per_minute is not None else self.default_max_per_minute
        with self._lock:
            ts = self._prune_locked(bucket_key, time.time())
            return max(0, cap - len(ts))


# Default cap (used when a model isn't listed in config.MODEL_RPM_LIMITS).
# Hard cap of 14 req/60s per key (free tier is 15 RPM — staying one under
# leaves headroom for clock/measurement drift between our tracker and
# Google's). Override via GEMINI_MAX_RPM_PER_KEY if your tier differs.
_rate_limiter = KeyRateLimiter(config.GEMINI_MAX_RPM_PER_KEY)


def _model_rpm_cap(model: str) -> int:
    """The local rate-limit cap to apply for this specific model — from
    config.MODEL_RPM_LIMITS if listed, else the global default."""
    return config.MODEL_RPM_LIMITS.get(model, config.GEMINI_MAX_RPM_PER_KEY)


# ---------------------------------------------------------------------------
# Client + API key rotation
# ---------------------------------------------------------------------------
# Module-level state is safe without locking because main.py runs cycles one
# at a time on a single worker thread (ThreadPoolExecutor(max_workers=1)).
_current_key_index = 0

if len(config.GEMINI_API_KEYS) > 1:
    logger.info(
        "Gemini: %s API key(s) configured for rotation on quota exhaustion.",
        len(config.GEMINI_API_KEYS),
    )
else:
    logger.warning(
        "Gemini: only 1 API key configured — GEMINI_API_KEYS is not set (or "
        "only contains one entry), so there is nothing to rotate to on a 429. "
        "Set GEMINI_API_KEYS as a comma-separated list of keys FROM DIFFERENT "
        "GOOGLE CLOUD PROJECTS to actually get separate quota pools — keys "
        "created under the same project can share the same underlying quota, "
        "in which case rotating between them will not avoid 429s either."
    )


def _build_client(key_index: int) -> genai.Client:
    http_options_kwargs = dict(
        # timeout is in MILLISECONDS for this SDK.
        timeout=config.GEMINI_TIMEOUT * 1000,
        # CRITICAL: the SDK's own default retry behavior is up to 5
        # attempts with exponential backoff (up to ~60s), and 429 is in
        # its default retryable status list. Left at the default, a
        # single generate_content() call would silently retry the SAME
        # already-exhausted key up to 5 times internally — taking up to
        # ~a minute — before our exception handler in _generate() ever
        # sees it and gets a chance to rotate to the next key. Setting
        # attempts=1 disables the SDK's internal retry entirely, so a
        # 429 raises immediately and OUR rotation logic (which is what
        # actually knows about the other keys) takes over right away.
        retry_options=types.HttpRetryOptions(attempts=1),
    )

    if config.GEMINI_PROXY_URL:
        # Scoped to ONLY this genai.Client's underlying httpx.Client —
        # scraper.py's cloudscraper/requests session and notifier.py's/
        # github_fallback.py's plain `requests` calls are entirely separate
        # HTTP stacks and never see this proxy. That separation is the
        # whole point: Gemini may require a supported-region IP while
        # Mostaql's Cloudflare WAF flags/blocks that same proxy IP as
        # datacenter/VPN traffic, so only Gemini's traffic goes through it.
        http_options_kwargs["client_args"] = {"proxy": config.GEMINI_PROXY_URL}

    return genai.Client(
        api_key=config.GEMINI_API_KEYS[key_index],
        http_options=types.HttpOptions(**http_options_kwargs),
    )


def _build_client_safe(key_index: int) -> Optional[genai.Client]:
    """Same as _build_client, but never raises — used when rotating to a
    new key mid-retry-loop, since a single malformed/invalid key in the
    list must not abort rotation to the REST of the list."""
    try:
        return _build_client(key_index)
    except Exception as exc:
        logger.error(
            "Failed to build Gemini client for key #%s (is it malformed?): %s",
            key_index + 1, exc,
        )
        return None


if config.GEMINI_PROXY_URL:
    logger.info(
        "Gemini: routing ONLY Gemini API traffic through proxy %s "
        "(scraper/Telegram/GitHub traffic is unaffected and stays direct).",
        config.GEMINI_PROXY_URL,
    )

_client = _build_client(_current_key_index)


def _is_quota_error(exc: Exception) -> bool:
    """Detects a 429 / RESOURCE_EXHAUSTED quota error across SDK versions,
    since relying on a single exception type/attribute is fragile."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


def _is_transient_error(exc: Exception) -> bool:
    """
    Detects gateway/server-side transient errors — 504 Gateway Timeout,
    DEADLINE_EXCEEDED, 503 Service Unavailable, and generic 500s — which are
    worth retrying on the SAME key after a short backoff (unlike quota
    errors, a different key doesn't fix a timed-out gateway).
    """
    code = getattr(exc, "code", None)
    if code in (500, 502, 503, 504):
        return True
    text = str(exc)
    markers = (
        "504", "DEADLINE_EXCEEDED", "Gateway Timeout",
        "503", "UNAVAILABLE", "Service Unavailable",
        "500", "Internal error", "Server disconnected",
    )
    return any(marker.lower() in text.lower() for marker in markers)


@dataclass
class Evaluation:
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


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    """
    Reads Google's own suggested wait time out of a 429 error's response
    body (a google.rpc.RetryInfo detail with a retryDelay like '13s'), when
    present, so _generate() can respect the server's actual guidance
    instead of guessing blindly with pure exponential backoff. Returns None
    if not present or not parseable — never raises.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    try:
        error_body = details.get("error", details)
        for item in error_body.get("details", []) or []:
            if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
                match = re.match(r"([\d.]+)\s*s", str(item.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    except (AttributeError, TypeError):
        pass
    return None


class _KeyLocallyRateLimited(Exception):
    """
    Internal marker: raised by _call_gemini_once() when the LOCAL rate
    limiter says this key has no capacity right this instant — e.g. a
    burst of tenacity's own transient-error retries pushed it over the cap
    mid-attempt, not just across separate calls. Deliberately excluded from
    _is_tenacity_retryable() so tenacity stops immediately rather than
    retrying against a key we already know is at its local cap; _generate()
    catches this specifically to move on to the next candidate key.
    """
    pass


def _is_tenacity_retryable(exc: BaseException) -> bool:
    """
    The retry PREDICATE tenacity uses to decide whether to retry at all.
    This is where requirement #1 actually lives: 429/RESOURCE_EXHAUSTED is
    explicitly excluded here, so tenacity NEVER retries a quota error —
    it's caught, logged, and routed to our own key-rotation/cooldown logic
    in _generate() on the very first occurrence, with zero extra requests
    burned trying the same exhausted key again. Only genuine
    transient/network-level errors (5xx, timeouts) are retryable.
    """
    if isinstance(exc, _KeyLocallyRateLimited):
        return False
    if _is_quota_error(exc):
        return False
    return _is_transient_error(exc)


@retry(
    retry=retry_if_exception(_is_tenacity_retryable),
    # +1 because stop_after_attempt counts the FIRST attempt too —
    # GEMINI_MAX_TRANSIENT_RETRIES retries after that, so e.g. a default of
    # 2 means 3 total attempts against this one key for a transient error,
    # matching "a maximum of 2 or 3 attempts."
    stop=stop_after_attempt(config.GEMINI_MAX_TRANSIENT_RETRIES + 1),
    wait=wait_exponential(
        multiplier=config.GEMINI_RETRY_BACKOFF_BASE,
        max=config.GEMINI_QUOTA_BACKOFF_MAX,
    ),
    reraise=True,  # re-raise the ORIGINAL exception, not tenacity's own RetryError wrapper
)
def _call_gemini_once(key_index: int, model: str, prompt: str, gen_config):
    """
    Exactly ONE logical Gemini call attempt against `key_index`'s client,
    for the given `model`. Rate-limit capacity is tracked per (key, model)
    pair — not per key alone — since Gemini's actual RPM quotas are
    per-model, not shared across every model callable under one key.
    tenacity re-invokes this whole function (including the rate-limiter
    check and record() below) on each retry, but ONLY for transient errors
    (see _is_tenacity_retryable). A 429 raised from here propagates
    immediately, unretried, straight out of the @retry decorator.
    """
    bucket = (key_index, model)
    cap = _model_rpm_cap(model)
    if not _rate_limiter.available(bucket, max_per_minute=cap):
        raise _KeyLocallyRateLimited(
            f"Key #{key_index + 1} has no local rate-limit capacity remaining for model '{model}'"
        )
    _rate_limiter.record(bucket)  # reserve capacity before sending
    return _client.models.generate_content(model=model, contents=prompt, config=gen_config)


def _generate(
    prompt: str,
    json_mode: bool = False,
    response_schema=None,
    temperature: float = None,
    max_output_tokens: int = None,
):
    """
    Immediate fallback chain (Key -> its models, RPM-descending) with a
    single global mutex and light inter-request throttling, plus PROACTIVE
    rate-limit avoidance — retry responsibilities cleanly split between
    tenacity and our own code:

    0. FALLBACK CHAIN ORDER: Key #1 first, ALL of its models tried
       (config.GEMINI_MODEL_CASCADE, highest-RPM first) before EVER moving
       to Key #2 — Key #1 is exhausted (every model, every real error)
       before Key #2 is touched at all. Within a key, a 429 on one model
       does NOT retry that model locally at all — it immediately moves to
       the next model in the cascade. No same-key/same-model wait-and-
       retry on quota errors, by design: waiting out an RPM window blocks
       whichever thread is running this, and moving on immediately is
       almost always faster than waiting for one specific bucket to reset
       when other buckets may already have room.
    1. PROACTIVE: for each key, computes which of ITS models currently
       have LOCAL rate-limit capacity (see KeyRateLimiter — tracked per
       (key, model) pair, since RPM quotas are per-model on Gemini, not
       shared across models under one key) before attempting anything.
       Models with no local room are skipped with zero API calls, not
       attempted-and-rejected.
    2. Per (key, model) pair, _call_gemini_once() is wrapped in @retry from
       tenacity — but its retry PREDICATE (_is_tenacity_retryable)
       explicitly excludes 429/RESOURCE_EXHAUSTED, so tenacity ONLY ever
       retries genuine transient/network errors (504/DEADLINE_EXCEEDED/
       503/500 — this also covers ReadTimeouts, which the SDK surfaces as
       connection/deadline errors, not quota errors), with exponential
       backoff, capped at GEMINI_MAX_TRANSIENT_RETRIES + 1 total attempts
       (default 3) against that one pair. A 429 always propagates out of
       tenacity immediately, unretried.
    3. GLOBAL MUTEX (_generate_lock): only one Gemini call is ever in
       flight process-wide, so concurrent callers (if this codebase is
       ever changed to have more than one) can't multiply instantaneous
       demand against the RPM limits. LIGHT THROTTLING
       (config.GEMINI_INTER_REQUEST_DELAY, default 1s) is applied between
       consecutive attempts within one _generate() call — small on
       purpose, meant to space out a burst of backlog items being drained
       in quick succession, not to wait out a rate-limit window.
    4. AUTOMATIC FUNCTION CALLING is explicitly disabled on every request
       (automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)).
       This codebase never passes `tools=` to Gemini, so AFC cannot
       actually trigger today regardless — this is an explicit, visible
       guarantee rather than an implicit "well, we just never call it that
       way" one, so a future change can't accidentally introduce hidden
       remote calls without this line having to change too.

    Terminal conditions — TWO distinct outcomes, deliberately different
    exceptions:
      - AllKeysRateLimited: NO (key, model) pair anywhere had local
        capacity — zero Gemini API calls were made at all this call.
      - AllKeysExhaustedError("ALL_KEYS_EXHAUSTED"): at least one call was
        actually attempted, and every attempted (key, model) pair failed
        with a real error. The original exception is chained via `from
        last_exc` so the root cause is never lost.
    Either way, callers (score_project/draft_proposal) catch it via their
    existing broad `except Exception` and return None, which
    evaluate_project() turns into Evaluation(ai_failed=True) — the one and
    only point where main.py's GitHub fallback (Telegram alert +
    persistent GitHub-hosted queue) gets triggered. That queue is ALSO
    where the "give the RPM window time to reset, then retry" behavior
    actually lives (main.py's consumer_loop periodically re-attempts
    queued items — see retry_pending_queue) — deliberately NOT
    implemented as a blocking sleep-and-retry inside this function, since
    that would freeze the one thread responsible for draining the rest of
    the backlog for the full wait duration. See generate_with_outer_backoff()
    below for the literal blocking-retry version, kept available but not
    wired into the default pipeline for that reason.
    """
    global _client, _current_key_index

    gen_config_kwargs = {
        # Explicit, visible guarantee — see docstring point 4 above.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    # response_schema requires response_mime_type="application/json" to be
    # set too (the SDK validates this — see types.GenerateContentConfig's
    # own field description) — implied automatically here so callers only
    # need to pass response_schema, not both.
    if json_mode or response_schema is not None:
        gen_config_kwargs["response_mime_type"] = "application/json"
    if response_schema is not None:
        gen_config_kwargs["response_schema"] = response_schema
    if temperature is not None:
        gen_config_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        gen_config_kwargs["max_output_tokens"] = max_output_tokens
    gen_config = types.GenerateContentConfig(**gen_config_kwargs)
    n_keys = len(config.GEMINI_API_KEYS)

    with _generate_lock:
        last_exc: Optional[Exception] = None
        any_capacity_found = False  # True once ANY (key, model) pair ever had local rate-limit room
        attempted_any_call = False  # gates the inter-request throttle (never sleeps before the first attempt)

        # Key #1 first, wrapping from the current key so a previously-
        # successful key stays "sticky" across calls rather than always
        # restarting at index 0.
        key_order = [(_current_key_index + offset) % n_keys for offset in range(n_keys)]

        for key_index in key_order:
            if key_index != _current_key_index or _client is None:
                candidate_client = _build_client_safe(key_index)
                if candidate_client is None:
                    logger.warning("Key #%s could not be initialized — skipping to next key", key_index + 1)
                    continue  # malformed key — skip this whole key
                _current_key_index = key_index
                _client = candidate_client

            key_had_capacity = False

            for model in config.GEMINI_MODEL_CASCADE:
                model_cap = _model_rpm_cap(model)
                if not _rate_limiter.available((key_index, model), max_per_minute=model_cap):
                    logger.debug(
                        "Key #%s / model '%s': no local rate-limit capacity "
                        "(%s req/60s cap) — skipping to next model",
                        key_index + 1, model, model_cap,
                    )
                    continue

                any_capacity_found = True
                key_had_capacity = True

                if attempted_any_call:
                    time.sleep(config.GEMINI_INTER_REQUEST_DELAY)  # light throttling, not a rate-limit-recovery wait
                attempted_any_call = True

                logger.debug(
                    "Using Gemini key #%s with model '%s' (%s req remaining in its local window)",
                    key_index + 1, model, _rate_limiter.remaining((key_index, model), max_per_minute=model_cap),
                )

                try:
                    result = _call_gemini_once(key_index, model, prompt, gen_config)
                except _KeyLocallyRateLimited:
                    logger.debug(
                        "Key #%s ran out of local rate-limit capacity for "
                        "model '%s' mid-attempt — moving to the next model",
                        key_index + 1, model,
                    )
                    continue
                except Exception as exc:
                    last_exc = exc

                    if _is_quota_error(exc):
                        # Real 429, immediately propagated by tenacity (never
                        # retried by it). Per the fallback-chain design: NO
                        # local retry, NO wait — immediately try the next
                        # model. (Logged for visibility only — the server's
                        # suggested delay is surfaced but not obeyed, since
                        # obeying it would block this thread.)
                        retry_delay = _extract_retry_delay_seconds(exc)
                        logger.warning(
                            "Key #%s hit quota on model '%s' (429 "
                            "RESOURCE_EXHAUSTED%s) — immediately moving to "
                            "the next model, no local retry/wait",
                            key_index + 1, model,
                            f", server-suggested retryDelay={retry_delay:.1f}s (not obeyed)" if retry_delay else "",
                        )
                        continue  # next model, same key

                    if _is_transient_error(exc):
                        # tenacity already retried this internally
                        # (GEMINI_MAX_TRANSIENT_RETRIES + 1 attempts,
                        # exponential backoff) and it still failed — this
                        # also covers ReadTimeouts. Nothing more to do on
                        # this (key, model) pair.
                        logger.warning(
                            "Transient Gemini error/timeout persisted through "
                            "tenacity's retry budget on key #%s, model '%s' "
                            "— moving to the next model",
                            key_index + 1, model,
                        )
                        continue

                    # Not a quota error, and not a (recognized) transient
                    # error at all — not worth retrying or rotating for.
                    logger.error("Non-retryable Gemini error on model '%s': %s", model, exc, exc_info=True)
                    raise
                else:
                    # Only a genuinely successful call counts toward RPD —
                    # see DailyRequestTracker's docstring for why failed
                    # attempts are deliberately excluded.
                    _daily_request_tracker.increment()
                    return result

            if key_had_capacity:
                logger.warning(
                    "All %s model(s) in GEMINI_MODEL_CASCADE exhausted for "
                    "key #%s — falling back to the next key, if any remain",
                    len(config.GEMINI_MODEL_CASCADE), key_index + 1,
                )

        if not any_capacity_found:
            logger.warning(
                "Every (key, model) pair is at its local rate limit — "
                "bypassing the Gemini API entirely (zero requests sent) "
                "and triggering the fallback immediately",
            )
            raise AllKeysRateLimited(
                f"All {n_keys} key(s) x {len(config.GEMINI_MODEL_CASCADE)} "
                f"model(s) at their local rate limits"
            )

        logger.error(
            "ALL_KEYS_EXHAUSTED: every key (%s) and every model (%s) was "
            "attempted and failed — THIS is the only point where the "
            "GitHub fallback (Telegram alert + persistent queue) gets "
            "triggered. Last error: %s",
            n_keys, ", ".join(config.GEMINI_MODEL_CASCADE), last_exc,
        )
        raise AllKeysExhaustedError("ALL_KEYS_EXHAUSTED") from last_exc


@retry(
    retry=retry_if_exception(lambda exc: isinstance(exc, AllKeysExhaustedError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=15, max=60),
    reraise=True,
)
def generate_with_outer_backoff(prompt: str, **kwargs):
    """
    NOT called anywhere in the default pipeline — provided because it was
    explicitly requested (tenacity-wrapped outer retry, triggering only on
    AllKeysExhaustedError, wait_exponential(multiplier=2, min=15, max=60)),
    but intentionally NOT wired into score_project()/draft_proposal().

    Why: this BLOCKS the calling thread for 15-60+ seconds (possibly twice,
    across up to 3 attempts — worst case ~2 minutes) every time the entire
    key x model grid is genuinely exhausted. In this codebase, _generate()
    is only ever called from main.py's single consumer thread — the same
    thread responsible for draining the REST of the backlog queue. Under
    the exact "burst backlog" scenario this was meant to help with, a
    blocking wait here would freeze the consumer for the whole backoff
    duration instead of moving on to the next queued project, making
    backlog throughput WORSE, not better.

    The non-blocking equivalent already exists in this codebase: when
    _generate() raises, evaluate_project() returns ai_failed=True,
    main.py's handle_ai_unavailable() queues the project to the
    GitHub-hosted persistent queue (see github_fallback.py) and sends an
    instant Telegram notice, and main.py's consumer_loop() ALREADY
    periodically re-attempts everything in that queue (every
    config.GITHUB_RETRY_CHECK_INTERVAL — default 300s — see
    retry_pending_queue()). That's "wait for the RPM window to
    reset, then retry the whole thing" — just implemented so the consumer
    thread stays free to keep processing other items while it waits,
    rather than blocking on one project.

    If you have a specific reason to want blocking backoff for some other
    call site (e.g. a one-off synchronous script, not the producer/
    consumer pipeline), this function is ready to use as-is.
    """
    return _generate(prompt, **kwargs)


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
    handled by KeyRateLimiter (proactive) and the key/model fallback chain
    (reactive) separately, so this tracker stays focused on one question:
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
        description="A realistic recommended bid price/budget for this project's scope, as a short string "
                     "including currency, e.g. '$150' or '$300-400'.",
    )
    delivery_days: int = Field(
        ge=1,
        description="Realistic estimated number of days to complete the project based on its scope.",
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
) -> Evaluation:
    """
    Shared tail logic that turns a raw score_data dict — regardless of
    whether it came from a fresh score_project()/score_projects_batch()
    call or a ScoreCache hit — into a complete Evaluation: validates
    match_score, and if it clears MATCH_THRESHOLD, drafts a proposal
    (always fresh, never cached — see ScoreCache's docstring) using the
    FULL description at its own, longer truncation length. Used by both
    evaluate_project() and evaluate_projects_batch() so this logic exists
    in exactly one place.
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

    proposal = None
    proposal_generated = False
    proposal_stats = dict(_EMPTY_CALL_STATS)
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

    return Evaluation(
        match_score=score,
        reasoning=reasoning,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggested_price=suggested_price,
        delivery_days=delivery_days,
        proposal_ar=proposal,
        original_desc_length=original_desc_length,
        truncated_desc_length=truncated_desc_length,
        prompt_tokens=score_stats["prompt_tokens"] + proposal_stats["prompt_tokens"],
        output_tokens=score_stats["output_tokens"] + proposal_stats["output_tokens"],
        total_tokens=score_stats["total_tokens"] + proposal_stats["total_tokens"],
        response_time_sec=round(score_stats["response_time_sec"] + proposal_stats["response_time_sec"], 3),
        # Whichever key was actually used LAST (proposal call if it ran,
        # otherwise the scoring call) — both usually the same key anyway.
        # A cache hit leaves score_stats["key_alias"] as None, so this
        # naturally falls back to the proposal call's key, or None if
        # neither call actually ran (below-threshold cache hit).
        key_alias=proposal_stats["key_alias"] or score_stats["key_alias"],
        proposal_generated=proposal_generated,
        effective_threshold=threshold,
    )


def evaluate_project(
    title: str,
    description: str,
    budget: Optional[str] = None,
    tags: Optional[List[str]] = None,
    client_info: Optional[dict] = None,
) -> Evaluation:
    """
    Full pipeline for one project:
      0. Local tag pre-filter — zero-cost skip if tags exist and don't
         overlap with config.MY_SKILLS at all.
      1. Score-cache lookup — zero-cost skip of the Gemini scoring call if
         this exact (title, description, MY_SKILLS) was already scored
         before (see ScoreCache).
      2. Score it via Gemini if not cached (including price/duration
         estimates), using the SHORTER scoring-specific truncation.
      3. If it clears the threshold, draft a proposal too, using the full
         (longer-truncated) description — client_info (see
         scraper.parse_client_info), if provided, lets draft_proposal()
         adjust TONE ONLY (see its docstring); it never changes scoring.
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
        client_info=client_info,
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
            )

    return results
