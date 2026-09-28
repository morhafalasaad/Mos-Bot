"""
gemini_client.py
----------------
Quota-aware, fully asynchronous Gemini client: a MODEL-major cascade with
round-robin API-key rotation.

WHY THIS MODULE EXISTS
----------------------
Free-tier Gemini quotas are tracked PER MODEL, in two independent windows:
RPM (requests/minute) and RPD (requests/day). The previous implementation
was KEY-major (exhaust every model on key #1 before touching key #2),
which burns the scarce low-quota models on one key while other keys still
have plenty of headroom on the high-quota ones. This module inverts it:

    for model in cascade:                # high-quota models first
        for key in round_robin(keys):    # rotate keys within that model
            try (model, key)

so a 15 RPM / 500 RPD "lite" model is drained across ALL keys before the
5 RPM / 20 RPD models are ever touched.

MODEL TIERS (see MODEL_REGISTRY below)
--------------------------------------
  PRIMARY   gemini-3.5-flash-lite, gemini-3.1-flash-lite  (15 RPM / 500 RPD)
  FALLBACK  gemini-3.8-flash -> 3.7 -> 3.6 -> 3.5 -> 3-flash -> 2.5-flash-lite
            (5 RPM / 20 RPD each)
  BLACKLIST 0 RPM / 0 RPD models — can never be configured or routed to.

HTTP 429 HANDLING
-----------------
A 429 is classified from Google's own `google.rpc.QuotaFailure` detail
(`violations[].quotaId`, e.g. "...RequestsPerMinute..." vs
"...RequestsPerDay..."), NOT from `RetryInfo.retryDelay` — the delay does
not distinguish the two (a per-DAY 429 can still say retryDelay "34s").

  * RPM  -> cool that (model, key) pair down for a short window and
            IMMEDIATELY try the next key on the SAME model.
  * RPD  -> mark that (model, key) exhausted until the next midnight
            Pacific time (when Google resets RPD). Once every key is
            exhausted/cooling for a model, the cascade moves to the next
            model automatically.
  * Anything ambiguous (no parsable quotaId) is treated as RPM, the
    cheaper-to-recover-from assumption; a mis-classified RPD simply
    re-classifies itself on the next 429, which does carry the quotaId.

When EVERY model x key pair is unavailable, the whole sweep is retried with
exponential backoff + jitter (bounded), then QuotaExhaustedError is raised.

IMPORTANT — QUOTAS ARE PER PROJECT, NOT PER KEY
-----------------------------------------------
Google applies rate limits per Cloud *project*, not per API key. Several
keys minted inside ONE project share ONE quota pool, so rotating between
them cannot avoid a 429. For rotation to help, each key must come from a
DIFFERENT Google Cloud project. (The pool logs a reminder at startup.)

CONCURRENCY MODEL
-----------------
Everything here is `async`. Shared mutable state (cooldowns, the
round-robin cursor, the local RPM windows) is only ever touched on the
event-loop thread with no `await` between read and write, so it is
race-free without locks. `ai_agent._generate()` bridges the synchronous,
thread-based rest of the application onto ONE dedicated event-loop thread
(see `ai_agent._run_async`), preserving the single-in-flight-call
guarantee the old global mutex provided.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

logger = logging.getLogger("gemini_client")

# Google resets RPD quotas at midnight Pacific time (ai.google.dev rate-limit docs).
_PACIFIC = ZoneInfo("America/Los_Angeles")


# ===========================================================================
# 1. Model registry + blacklist
# ===========================================================================
class Tier(str, Enum):
    PRIMARY = "primary"    # high quota: 15 RPM / 500 RPD
    FALLBACK = "fallback"  # low quota:   5 RPM /  20 RPD


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: Tier
    rpm: int
    rpd: int


# The authoritative, ordered cascade. Order here IS the routing order.
# RPM/RPD figures are the free-tier limits you specified; they are only
# used for LOCAL proactive throttling and logging — Google's live 429s
# remain the source of truth. Override per model via env (see config.py).
_DEFAULT_CASCADE: Tuple[ModelSpec, ...] = (
    # --- Primary tier (high quota) -------------------------------------
    ModelSpec("gemini-3.5-flash-lite", Tier.PRIMARY, rpm=15, rpd=500),
    ModelSpec("gemini-3.1-flash-lite", Tier.PRIMARY, rpm=15, rpd=500),
    # --- Fallback tier (low quota) -------------------------------------
    ModelSpec("gemini-3.8-flash", Tier.FALLBACK, rpm=5, rpd=20),
    ModelSpec("gemini-3.7-flash", Tier.FALLBACK, rpm=5, rpd=20),
    ModelSpec("gemini-3.6-flash", Tier.FALLBACK, rpm=5, rpd=20),
    ModelSpec("gemini-3.5-flash", Tier.FALLBACK, rpm=5, rpd=20),
    ModelSpec("gemini-3-flash", Tier.FALLBACK, rpm=5, rpd=20),
    ModelSpec("gemini-2.5-flash-lite", Tier.FALLBACK, rpm=5, rpd=20),
)

# 0 RPM / 0 RPD — deprecated or disabled. These must NEVER be routable,
# not via the default cascade, not via a GEMINI_MODEL_CASCADE env override,
# not via a legacy GEMINI_MODEL env var. Canonical (lower-case) names of
# the five blacklisted families you listed, plus the API-id spellings and
# common variants ("-preview", "-exp", "-latest", "-001", "models/" prefix)
# that resolve to the same disabled model. Matching is done by
# `is_blacklisted()` on a NORMALISED name so a variant can't slip through.
BLACKLISTED_MODELS: frozenset = frozenset({
    "gemini-2-flash",          # Gemini 2 Flash
    "gemini-2.0-flash",        #   (API-id spelling)
    "gemini-2-flash-lite",     # Gemini 2 Flash Lite
    "gemini-2.0-flash-lite",   #   (API-id spelling)
    "gemini-2.5-pro",          # Gemini 2.5 Pro
    "gemini-3.1-pro",          # Gemini 3.1 Pro
    "gemini-omni-flash",       # Gemini Omni Flash
})

# Suffixes Google appends to a base model id for dated/preview/alias builds.
_VARIANT_SUFFIX = re.compile(
    r"(?:-(?:preview|exp|experimental|latest|lite-preview)"
    r"|-\d{2}-\d{2}(?:-\d{2,4})?"       # -05-20 / -05-20-2025
    r"|-\d{3}"                          # -001
    r"|-\d{4,8}"                        # -20250514
    r")+$"
)


def _normalise_model_name(name: str) -> str:
    """'models/Gemini-2.5-Pro-Preview-05-06' -> 'gemini-2.5-pro'."""
    n = (name or "").strip().lower()
    if n.startswith("models/"):
        n = n[len("models/"):]
    # Strip variant suffixes repeatedly ("-preview-05-06", "-001", ...).
    prev = None
    while prev != n:
        prev = n
        n = _VARIANT_SUFFIX.sub("", n)
    return n


def is_blacklisted(name: str) -> bool:
    """True if `name` (in any common spelling/variant) is a zero-quota
    model that must never be routed to."""
    return _normalise_model_name(name) in BLACKLISTED_MODELS


class BlacklistedModelError(ValueError):
    """Raised when configuration tries to route to a blacklisted model."""


def build_cascade(
    overrides: Optional[Sequence[str]] = None,
    rpm_overrides: Optional[Dict[str, int]] = None,
    rpd_overrides: Optional[Dict[str, int]] = None,
    strict: bool = False,
) -> List[ModelSpec]:
    """
    Builds the ordered, validated cascade.

    `overrides` (e.g. from a GEMINI_MODEL_CASCADE env var) replaces the
    ORDER/MEMBERSHIP of the default cascade. Every entry is checked
    against the blacklist. A blacklisted entry is dropped with a loud
    warning (default) or raises BlacklistedModelError (`strict=True`).
    Unknown-but-not-blacklisted names are allowed (Google ships new
    models constantly) and treated as FALLBACK tier at the conservative
    5 RPM / 20 RPD, so an unrecognised name can never accidentally be
    over-driven.

    Guarantees on the returned list: non-empty, no duplicates, contains
    no blacklisted model. Raises ValueError if nothing routable remains.
    """
    known = {m.name: m for m in _DEFAULT_CASCADE}
    names: List[str] = list(overrides) if overrides else [m.name for m in _DEFAULT_CASCADE]

    specs: List[ModelSpec] = []
    seen: set = set()
    for raw in names:
        name = (raw or "").strip()
        if not name:
            continue
        if is_blacklisted(name):
            msg = (
                f"Model '{name}' is BLACKLISTED (0 RPM / 0 RPD — deprecated or "
                f"disabled) and can never be routed to."
            )
            if strict:
                raise BlacklistedModelError(msg)
            logger.error("%s Dropping it from the cascade.", msg)
            continue
        if name in seen:
            continue
        seen.add(name)
        base = known.get(name) or ModelSpec(name, Tier.FALLBACK, rpm=5, rpd=20)
        specs.append(ModelSpec(
            name=base.name,
            tier=base.tier,
            rpm=(rpm_overrides or {}).get(name, base.rpm),
            rpd=(rpd_overrides or {}).get(name, base.rpd),
        ))

    if not specs:
        raise ValueError(
            "Gemini cascade is empty after removing blacklisted/blank models. "
            "Check GEMINI_MODEL_CASCADE."
        )
    return specs


# ===========================================================================
# 2. Errors + 429 classification
# ===========================================================================
class QuotaKind(str, Enum):
    RPM = "rpm"        # per-minute window: recovers in seconds -> rotate KEY
    RPD = "rpd"        # per-day window: recovers at PT midnight -> next MODEL
    UNKNOWN = "unknown"


class QuotaExhaustedError(RuntimeError):
    """Every (model, key) pair is exhausted/cooling even after the bounded
    exponential-backoff sweeps. `__cause__` carries the last API error."""


# Back-compat aliases: the previous ai_agent.py implementation raised these
# two distinct exception types (AllKeysRateLimited when NO call was ever
# attempted because every pair was locally rate-limited, vs
# AllKeysExhaustedError when at least one call was attempted and all
# failed). The new cascade engine collapses both outcomes into
# QuotaExhaustedError (the distinction stopped being actionable once RPM
# vs RPD is classified from the 429 body itself), but the names are kept
# importable — as aliases of QuotaExhaustedError — for any code or test
# that still references ai_agent.AllKeysExhaustedError / AllKeysRateLimited.
AllKeysExhaustedError = QuotaExhaustedError
AllKeysRateLimited = QuotaExhaustedError


class ModelUnavailableError(RuntimeError):
    """Internal: a model returned a permanent 404/NOT_FOUND-style failure."""


def _is_api_error_code(exc: BaseException, *codes: int) -> bool:
    return getattr(exc, "code", None) in codes


def is_quota_error(exc: BaseException) -> bool:
    """HTTP 429 / RESOURCE_EXHAUSTED (SDK-version tolerant)."""
    if _is_api_error_code(exc, 429):
        return True
    status = str(getattr(exc, "status", "") or "")
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in status or "RESOURCE_EXHAUSTED" in text or " 429" in text[:12]


def is_transient_error(exc: BaseException) -> bool:
    """5xx / timeout-style errors: worth a bounded retry on the same pair."""
    if _is_api_error_code(exc, 500, 502, 503, 504):
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    text = str(exc).lower()
    return any(m in text for m in (
        "deadline_exceeded", "gateway timeout", "service unavailable",
        "server disconnected", "internal error", "readtimeout", "connecttimeout",
    ))


def is_model_gone_error(exc: BaseException) -> bool:
    """404 / NOT_FOUND / 'no longer available' — a retired or unknown model.
    Unlike a quota error this can NEVER succeed later, so the model is
    parked for a long time instead of being re-tried every request (this is
    the exact failure mode of the gemini-2.5-flash-lite incident)."""
    if _is_api_error_code(exc, 404):
        return True
    text = str(exc)
    return "NOT_FOUND" in text or "no longer available" in text.lower()


def _iter_violations(exc: BaseException) -> Iterable[dict]:
    """Yields every QuotaFailure violation dict in a 429's error body."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return
    body = details.get("error", details)
    if not isinstance(body, dict):
        return
    for item in body.get("details", []) or []:
        if isinstance(item, dict) and "QuotaFailure" in str(item.get("@type", "")):
            for v in item.get("violations", []) or []:
                if isinstance(v, dict):
                    yield v


def classify_quota_error(exc: BaseException) -> QuotaKind:
    """
    Distinguishes an RPD (daily) 429 from an RPM (per-minute) 429.

    Uses `QuotaFailure.violations[].quotaId`. A single 429 body may list
    BOTH a PerMinute and a PerDay violation at once (observed in the
    wild); in that case RPD wins — if the daily bucket is empty, retrying
    on another key of the same project is pointless, moving on to the
    next model is the only correct action.

    Falls back to scanning the human-readable message when no structured
    detail is present. Returns UNKNOWN when nothing is conclusive.
    """
    ids = [str(v.get("quotaId", "")) for v in _iter_violations(exc)]
    joined = " ".join(ids).lower()
    if "perday" in joined or "daily" in joined:
        return QuotaKind.RPD
    if "perminute" in joined or "persecond" in joined:
        return QuotaKind.RPM

    text = str(exc).lower()
    if "perday" in text or "per day" in text or "daily" in text:
        return QuotaKind.RPD
    if "perminute" in text or "per minute" in text:
        return QuotaKind.RPM
    return QuotaKind.UNKNOWN


def extract_retry_delay_seconds(exc: BaseException) -> Optional[float]:
    """Google's suggested wait from `RetryInfo.retryDelay` ('13s'). Only a
    HINT for sizing an RPM cooldown — never used to tell RPM from RPD."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        body = details.get("error", details)
        for item in (body.get("details", []) if isinstance(body, dict) else []) or []:
            if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
                m = re.match(r"\s*([\d.]+)\s*s", str(item.get("retryDelay", "")))
                if m:
                    return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def seconds_until_pacific_midnight(now: Optional[datetime] = None) -> float:
    """Seconds until the next 00:00 America/Los_Angeles (when RPD resets).
    DST-safe: computed on real aware datetimes (subtracting two zoneinfo-
    aware instants yields true elapsed wall-clock time across a DST
    transition), not a naive fixed-24h assumption."""
    now = now or datetime.now(_PACIFIC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_PACIFIC)
    now = now.astimezone(_PACIFIC)
    tomorrow = (now + timedelta(days=1)).date()
    reset = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=_PACIFIC)
    return max(1.0, (reset - now).total_seconds())


# ===========================================================================
# 3. Key pool: round-robin + per-(model, key) cooldowns + local RPM window
# ===========================================================================
class KeyPool:
    """
    Owns API keys and all per-(model, key) availability state.

    Round-robin: a per-model cursor advances on every successful pick, so
    load is spread evenly across keys instead of always hammering key #1.
    Unavailable pairs are skipped during selection.

    State is only mutated on the event-loop thread with no `await` between
    read and write, so no locks are needed.
    """

    def __init__(self, keys: Sequence[str], clock: Callable[[], float] = time.monotonic):
        cleaned = [k for k in (keys or []) if k]
        if not cleaned:
            raise ValueError("KeyPool needs at least one API key")
        self._keys: List[str] = list(cleaned)
        self._clock = clock
        self._cursor: Dict[str, int] = {}                       # model -> next start index
        self._blocked_until: Dict[Tuple[str, int], float] = {}  # (model, key_idx) -> monotonic ts
        self._blocked_kind: Dict[Tuple[str, int], QuotaKind] = {}
        self._rpm_window: Dict[Tuple[str, int], List[float]] = {}
        self._model_dead_until: Dict[str, float] = {}           # permanent-404 breaker
        self._last_key_index: int = 0

    # -- basic info ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    def key(self, index: int) -> str:
        return self._keys[index]

    @property
    def last_key_index(self) -> int:
        """Index of the key that served the most recent successful call."""
        return self._last_key_index

    # -- model-level circuit breaker (404 / retired) -------------------
    def model_is_dead(self, model: str) -> bool:
        return self._clock() < self._model_dead_until.get(model, 0.0)

    def mark_model_dead(self, model: str, seconds: float) -> None:
        self._model_dead_until[model] = self._clock() + seconds

    # -- per-pair blocking ----------------------------------------------
    def _is_blocked(self, model: str, idx: int) -> bool:
        pair = (model, idx)
        until = self._blocked_until.get(pair)
        if until is None:
            return False
        if self._clock() >= until:
            self._blocked_until.pop(pair, None)
            self._blocked_kind.pop(pair, None)
            return False
        return True

    def block(self, model: str, idx: int, seconds: float, kind: QuotaKind) -> None:
        pair = (model, idx)
        self._blocked_until[pair] = self._clock() + max(0.0, seconds)
        self._blocked_kind[pair] = kind

    def blocked_kind(self, model: str, idx: int) -> Optional[QuotaKind]:
        return self._blocked_kind.get((model, idx)) if self._is_blocked(model, idx) else None

    # -- proactive local RPM window ------------------------------------
    def _local_rpm_ok(self, model: str, idx: int, rpm_cap: int) -> bool:
        pair = (model, idx)
        now = self._clock()
        window = self._rpm_window.setdefault(pair, [])
        while window and window[0] < now - 60.0:
            window.pop(0)
        return len(window) < rpm_cap

    def record_request(self, model: str, idx: int) -> None:
        self._rpm_window.setdefault((model, idx), []).append(self._clock())

    def local_rpm_wait(self, model: str, idx: int, rpm_cap: int) -> float:
        """Seconds until this pair's local 60s window has room again."""
        window = self._rpm_window.get((model, idx), [])
        now = self._clock()
        live = [t for t in window if t >= now - 60.0]
        if len(live) < rpm_cap:
            return 0.0
        return max(0.0, live[0] + 60.0 - now)

    # -- selection -------------------------------------------------------
    def next_key_for(self, model: str, rpm_cap: int, skip: Optional[set] = None) -> Optional[int]:
        """
        Next usable key index for `model` in round-robin order, or None if
        every key is blocked / locally at its RPM cap / already tried this
        pass (`skip`). Advances the per-model cursor past the chosen key.
        """
        n = len(self._keys)
        start = self._cursor.get(model, 0) % n
        for offset in range(n):
            idx = (start + offset) % n
            if skip and idx in skip:
                continue
            if self._is_blocked(model, idx):
                continue
            if not self._local_rpm_ok(model, idx, rpm_cap):
                continue
            self._cursor[model] = (idx + 1) % n
            return idx
        return None

    def soonest_recovery(self, models: Iterable[str], rpm_caps: Dict[str, int]) -> Optional[float]:
        """Shortest wait (s) until ANY currently-unavailable non-dead pair
        recovers — used to size backoff. None if nothing is waiting."""
        now = self._clock()
        best: Optional[float] = None
        for model in models:
            if self.model_is_dead(model):
                continue
            for idx in range(len(self._keys)):
                waits = []
                until = self._blocked_until.get((model, idx))
                if until is not None and until > now:
                    waits.append(until - now)
                cap = rpm_caps.get(model, 0)
                if cap and not self._local_rpm_ok(model, idx, cap):
                    waits.append(self.local_rpm_wait(model, idx, cap))
                if waits:
                    w = max(waits)
                    best = w if best is None else min(best, w)
        return best

    def mark_success(self, idx: int) -> None:
        self._last_key_index = idx

    def status(self, models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Diagnostic snapshot for logs / health endpoints."""
        out: Dict[str, Dict[str, Any]] = {}
        for m in models:
            out[m] = {
                "dead": self.model_is_dead(m),
                "keys": {
                    i + 1: (self.blocked_kind(m, i).value if self.blocked_kind(m, i) else "ok")
                    for i in range(len(self._keys))
                },
            }
        return out


# ===========================================================================
# 4. Cascade engine
# ===========================================================================
@dataclass
class GenerationResult:
    """A successful response plus which (model, key) actually served it."""
    response: Any
    model: str
    key_index: int          # 0-based
    attempts: int           # total API calls made for this request

    @property
    def key_alias(self) -> str:
        return f"Gemini key #{self.key_index + 1}"


@dataclass
class _Tunables:
    """All timing knobs in one place (mapped from config.py)."""
    rpm_cooldown_floor: float = 5.0       # min cooldown after an RPM 429 (s)
    rpm_cooldown_cap: float = 65.0        # max cooldown after an RPM 429 (s)
    dead_model_seconds: float = 6 * 3600  # park a 404'd model this long
    transient_retries: int = 2            # extra tries per pair on 5xx/timeout
    transient_backoff_base: float = 2.0
    transient_backoff_cap: float = 20.0
    backoff_sweeps: int = 3               # full-cascade sweeps before giving up
    backoff_base: float = 2.0             # exp. backoff base (s) between sweeps
    backoff_cap: float = 60.0             # per-sweep wait cap (s)
    total_deadline: float = 180.0         # hard ceiling for one request (s)
    inter_request_delay: float = 0.0      # optional gentle spacing (s)


class GeminiCascadeClient:
    """
    Async, MODEL-major cascade with round-robin key rotation.

    Construct once (per process), call `await client.generate(...)`.
    The SDK client for each key is built lazily and cached.
    """

    def __init__(
        self,
        api_keys: Sequence[str],
        cascade: Optional[Sequence[ModelSpec]] = None,
        *,
        timeout_seconds: float = 30.0,
        proxy_url: Optional[str] = None,
        tunables: Optional[_Tunables] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
        rng: Optional[random.Random] = None,
        now_provider: Callable[[], datetime] = lambda: datetime.now(_PACIFIC),
    ):
        self.pool = KeyPool(api_keys, clock=clock)
        self.cascade: List[ModelSpec] = list(cascade) if cascade else build_cascade()
        # Defence in depth: even a caller-supplied cascade is re-validated.
        for spec in self.cascade:
            if is_blacklisted(spec.name):
                raise BlacklistedModelError(
                    f"Model '{spec.name}' is BLACKLISTED (0 RPM / 0 RPD) and cannot be routed to."
                )
        self._timeout = timeout_seconds
        self._proxy = proxy_url
        self.t = tunables or _Tunables()
        self._client_factory = client_factory or self._default_client_factory
        self._clients: Dict[str, Any] = {}
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._rpm_caps = {m.name: m.rpm for m in self.cascade}
        self._now = now_provider
        self.last_model: Optional[str] = None

        if len(self.pool) > 1:
            logger.info(
                "Gemini: %d API keys, round-robin rotation on RPM 429s. NOTE: Google "
                "applies quotas per Cloud PROJECT, not per key — keys from the SAME "
                "project share one pool and rotation will not avoid 429s. Use keys "
                "from DIFFERENT projects.", len(self.pool),
            )
        else:
            logger.warning(
                "Gemini: only 1 API key configured — there is nothing to rotate to on "
                "an RPM 429; the cascade will move straight to the next model instead."
            )
        logger.info(
            "Gemini cascade (%d models): %s",
            len(self.cascade), " -> ".join(f"{m.name}[{m.tier.value}]" for m in self.cascade),
        )

    # -- SDK client construction --------------------------------------
    def _default_client_factory(self, api_key: str) -> genai.Client:
        kwargs: Dict[str, Any] = dict(
            timeout=int(self._timeout * 1000),  # SDK timeout is in MILLISECONDS
            # The SDK's own retry defaults would silently re-hit an exhausted
            # key up to 5x before WE ever see the 429. attempts=1 disables
            # it so this module's rotation/cascade logic is the only retry.
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        if self._proxy:
            # Async calls use the ASYNC client args (client.aio), so the
            # proxy must be set here too or Gemini traffic would bypass it.
            kwargs["client_args"] = {"proxy": self._proxy}
            kwargs["async_client_args"] = {"proxy": self._proxy}
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(**kwargs))

    def _client_for(self, key_index: int) -> Any:
        key = self.pool.key(key_index)
        cli = self._clients.get(key)
        if cli is None:
            cli = self._client_factory(key)
            self._clients[key] = cli
        return cli

    # -- one API call ---------------------------------------------------
    async def _call(self, key_index: int, model: str, prompt: Any, gen_config: Any) -> Any:
        cli = self._client_for(key_index)
        return await asyncio.wait_for(
            cli.aio.models.generate_content(model=model, contents=prompt, config=gen_config),
            timeout=self._timeout + 5.0,  # belt-and-braces over the SDK's own timeout
        )

    async def _attempt(
        self, key_index: int, model: str, prompt: Any, gen_config: Any
    ) -> Tuple[Any, int]:
        """
        One (model, key) pair, with a bounded retry for TRANSIENT errors
        only. 429s and 404s are NEVER retried here — they propagate at
        once so the caller can rotate key / move model. Returns
        (response, api_calls_made).
        """
        calls = 0
        for attempt in range(self.t.transient_retries + 1):
            self.pool.record_request(model, key_index)  # reserve local RPM capacity
            calls += 1
            try:
                return await self._call(key_index, model, prompt, gen_config), calls
            except Exception as exc:  # noqa: BLE001 - classified below
                if is_quota_error(exc) or is_model_gone_error(exc):
                    exc.__dict__["_calls_made"] = calls
                    raise
                if is_transient_error(exc) and attempt < self.t.transient_retries:
                    delay = min(
                        self.t.transient_backoff_cap,
                        self.t.transient_backoff_base * (2 ** attempt),
                    )
                    delay *= 0.5 + self._rng.random() / 2  # jitter in [0.5, 1.0)
                    logger.warning(
                        "Transient error on %s / key #%d (%s) — retry %d/%d in %.1fs",
                        model, key_index + 1, type(exc).__name__,
                        attempt + 1, self.t.transient_retries, delay,
                    )
                    await self._sleep(delay)
                    continue
                exc.__dict__["_calls_made"] = calls
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    # -- 429 bookkeeping -----------------------------------------------
    def _apply_quota_block(self, model: str, key_index: int, exc: BaseException) -> QuotaKind:
        kind = classify_quota_error(exc)
        if kind is QuotaKind.RPD:
            secs = seconds_until_pacific_midnight(self._now())
            self.pool.block(model, key_index, secs, QuotaKind.RPD)
            logger.warning(
                "RPD (daily) quota exhausted: %s / key #%d — parked until PT midnight (~%.1fh).",
                model, key_index + 1, secs / 3600,
            )
        else:
            hint = extract_retry_delay_seconds(exc)
            secs = hint if hint is not None else 60.0
            secs = min(self.t.rpm_cooldown_cap, max(self.t.rpm_cooldown_floor, secs))
            # UNKNOWN is deliberately handled as RPM (cheapest to recover from).
            self.pool.block(model, key_index, secs, QuotaKind.RPM)
            logger.warning(
                "RPM 429 (%s): %s / key #%d — cooling %.0fs, rotating to next key.",
                "classified" if kind is QuotaKind.RPM else "unclassified->RPM",
                model, key_index + 1, secs,
            )
        return kind

    # -- one full pass over the cascade ---------------------------------
    async def _sweep(
        self, prompt: Any, gen_config: Any, budget: List[int]
    ) -> Tuple[Optional[GenerationResult], Optional[BaseException]]:
        """
        A single model-major pass. Returns (result, last_exc). result is
        None if nothing succeeded this pass. `budget[0]` accumulates the
        total API calls made.
        """
        last_exc: Optional[BaseException] = None
        first = True
        for spec in self.cascade:
            model = spec.name
            if self.pool.model_is_dead(model):
                continue
            tried: set = set()
            while True:
                idx = self.pool.next_key_for(model, spec.rpm, skip=tried)
                if idx is None:
                    break  # every key for this model is blocked/capped/tried -> next model
                tried.add(idx)
                if not first and self.t.inter_request_delay > 0:
                    await self._sleep(self.t.inter_request_delay)
                first = False
                try:
                    response, calls = await self._attempt(idx, model, prompt, gen_config)
                    budget[0] += calls
                    self.pool.mark_success(idx)
                    self.last_model = model
                    return GenerationResult(response, model, idx, budget[0]), None
                except Exception as exc:  # noqa: BLE001
                    budget[0] += getattr(exc, "_calls_made", 1)
                    last_exc = exc
                    if is_quota_error(exc):
                        kind = self._apply_quota_block(model, idx, exc)
                        if kind is QuotaKind.RPD:
                            # A daily bucket is per (project, model): the
                            # sibling keys almost certainly share it, but
                            # they may be separate projects, so still try
                            # them — but each gets its own 429 quickly.
                            pass
                        continue  # next key, SAME model
                    if is_model_gone_error(exc):
                        self.pool.mark_model_dead(model, self.t.dead_model_seconds)
                        logger.error(
                            "Model '%s' returned NOT_FOUND/retired (%s) — parking it "
                            "for %.0fh and moving on.", model, exc, self.t.dead_model_seconds / 3600,
                        )
                        break  # next model
                    if is_transient_error(exc):
                        logger.warning(
                            "Transient failure persisted on %s / key #%d — trying next key/model.",
                            model, idx + 1,
                        )
                        continue
                    # Non-quota, non-transient, non-404 (e.g. 400 bad request,
                    # 401/403 auth, safety block): another key/model will not
                    # fix a broken request. Fail fast.
                    logger.error("Non-retryable Gemini error on %s / key #%d: %s", model, idx + 1, exc)
                    raise
        return None, last_exc

    # -- public entry point ----------------------------------------------
    async def generate(self, prompt: Any, gen_config: Any) -> GenerationResult:
        """
        Cascade + rotate until success, or exhaust everything.

        When a full sweep finds NOTHING usable, waits with exponential
        backoff (+ full jitter, sized toward the soonest real recovery,
        capped) and sweeps again, up to `backoff_sweeps` times or the hard
        `total_deadline`, then raises QuotaExhaustedError.
        """
        started = self._clock()
        budget = [0]
        last_exc: Optional[BaseException] = None

        for sweep_no in range(1, self.t.backoff_sweeps + 1):
            result, exc = await self._sweep(prompt, gen_config, budget)
            if result is not None:
                if sweep_no > 1:
                    logger.info("Recovered on sweep %d via %s / %s", sweep_no, result.model, result.key_alias)
                return result
            last_exc = exc or last_exc

            if sweep_no == self.t.backoff_sweeps:
                break

            # Nothing usable anywhere: wait. Exponential in the sweep number,
            # but never longer than the soonest genuine recovery (no point
            # sleeping past the moment something frees up), never beyond the
            # cap, and never past the request's hard deadline.
            expo = min(self.t.backoff_cap, self.t.backoff_base * (2 ** (sweep_no - 1)))
            recovery = self.pool.soonest_recovery(
                [m.name for m in self.cascade], self._rpm_caps
            )
            wait = expo if recovery is None else min(expo, recovery + 0.25)
            wait = self._rng.uniform(wait / 2, wait) if wait > 1 else wait  # jitter
            remaining = self.t.total_deadline - (self._clock() - started)
            if remaining <= wait:
                logger.error("Total deadline (%.0fs) would be exceeded — giving up.", self.t.total_deadline)
                break
            logger.warning(
                "Cascade exhausted (sweep %d/%d): every model x key is rate-limited. "
                "Backing off %.1fs before retrying.", sweep_no, self.t.backoff_sweeps, wait,
            )
            await self._sleep(wait)

        logger.error(
            "QUOTA_EXHAUSTED: all %d model(s) x %d key(s) exhausted after %d API call(s). "
            "State: %s | last error: %s",
            len(self.cascade), len(self.pool), budget[0],
            self.pool.status([m.name for m in self.cascade]), last_exc,
        )
        raise QuotaExhaustedError("ALL_KEYS_EXHAUSTED") from last_exc
