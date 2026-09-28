"""
Tests for gemini_client.py — the model-major cascade + round-robin key
rotation + RPM/RPD-aware 429 handling + blacklist guard.

No network, no real SDK calls: a scriptable fake replaces the SDK client,
and time/sleep are virtualised so exponential-backoff paths run instantly
and deterministically.
"""

import asyncio
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import gemini_client as gc
from gemini_client import (
    BLACKLISTED_MODELS,
    BlacklistedModelError,
    GeminiCascadeClient,
    KeyPool,
    ModelSpec,
    QuotaExhaustedError,
    QuotaKind,
    Tier,
    _Tunables,
    build_cascade,
    classify_quota_error,
    is_blacklisted,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
class FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError's shape (code/status/details)."""

    def __init__(self, code, status="", details=None, message=""):
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status
        self.details = details


def quota_429(kind: str, retry_delay: str = "13s", both: bool = False) -> FakeAPIError:
    """Builds a realistic 429 body. kind: 'rpm' | 'rpd' | 'none'."""
    violations = []
    if kind in ("rpm", "rpd") and not both:
        qid = (
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
            if kind == "rpm"
            else "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        violations.append({"quotaId": qid})
    if both:
        violations = [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"},
        ]
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    if violations:
        details.insert(0, {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": violations})
    return FakeAPIError(429, "RESOURCE_EXHAUSTED", {"error": {"details": details}}, "quota")


class VirtualTime:
    """Deterministic monotonic clock (for cooldowns) plus a matching
    virtual wall-clock `now()` (for RPD/Pacific-midnight math) that
    advances in lockstep — sleeping/advancing one advances both."""

    def __init__(self, wall_start=None):
        self.now = 1000.0
        self.slept = []
        self._wall = wall_start or datetime(2026, 1, 1, tzinfo=ZoneInfo("America/Los_Angeles"))

    def clock(self):
        return self.now

    def wall_now(self):
        from datetime import timedelta
        return self._wall + timedelta(seconds=self.now - 1000.0)

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    async def generate_content(self, model, contents, config):
        return await self.owner._dispatch(model)


class FakeAio:
    def __init__(self, owner):
        self.models = FakeModels(owner)


class FakeClient:
    """One per API key. `script(model, key, call_no)` returns a response or raises."""

    def __init__(self, key, script, calls):
        self.key = key
        self._script = script
        self._calls = calls
        self.aio = FakeAio(self)

    async def _dispatch(self, model):
        n = sum(1 for (m, k) in self._calls if m == model and k == self.key) + 1
        self._calls.append((model, self.key))
        outcome = self._script(model, self.key, n)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(script, keys=("k1", "k2", "k3"), cascade=None, vt=None, **tune):
    vt = vt or VirtualTime()
    calls = []
    tunables = _Tunables(
        transient_retries=tune.pop("transient_retries", 1),
        transient_backoff_base=tune.pop("transient_backoff_base", 1.0),
        **tune,
    )
    client = GeminiCascadeClient(
        list(keys),
        cascade=cascade,
        tunables=tunables,
        client_factory=lambda key: FakeClient(key, script, calls),
        clock=vt.clock,
        sleep=vt.sleep,
        rng=random.Random(0),
        now_provider=vt.wall_now,
    )
    return client, calls, vt


def run(coro):
    return asyncio.run(coro)


OK = object()  # sentinel "success" payload


def ok_script(model, key, n):
    return f"ok:{model}:{key}"


# ---------------------------------------------------------------------------
# 1. Model hierarchy
# ---------------------------------------------------------------------------
def test_default_cascade_exact_order_and_tiers():
    names = [m.name for m in build_cascade()]
    assert names == [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash",
        "gemini-2.5-flash-lite",
    ]


def test_primary_tier_quotas_and_fallback_quotas():
    specs = {m.name: m for m in build_cascade()}
    for n in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
        assert (specs[n].tier, specs[n].rpm, specs[n].rpd) == (Tier.PRIMARY, 15, 500)
    for n in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
              "gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash-lite"):
        assert (specs[n].tier, specs[n].rpm, specs[n].rpd) == (Tier.FALLBACK, 5, 20)


def test_primary_tier_always_precedes_fallback_tier():
    tiers = [m.tier for m in build_cascade()]
    first_fallback = tiers.index(Tier.FALLBACK)
    assert all(t is Tier.PRIMARY for t in tiers[:first_fallback])
    assert all(t is Tier.FALLBACK for t in tiers[first_fallback:])


# ---------------------------------------------------------------------------
# 2. Blacklist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "gemini-2-flash", "gemini-2.0-flash", "gemini-2-flash-lite", "gemini-2.0-flash-lite",
    "gemini-2.5-pro", "gemini-3.1-pro", "gemini-omni-flash",
    # variants must not slip through:
    "models/gemini-2.5-pro", "Gemini-2.5-Pro", "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash-001", "gemini-2.0-flash-lite-001", "gemini-3.1-pro-preview",
    "gemini-omni-flash-exp", "gemini-2.0-flash-latest",
])
def test_blacklisted_names_detected_including_variants(name):
    assert is_blacklisted(name)


@pytest.mark.parametrize("name", [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash",
    "gemini-3-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash",
    # near-misses that are NOT blacklisted:
    "gemini-2.5-flash", "gemini-3.1-flash", "gemini-3.2-pro",
])
def test_legitimate_models_not_blacklisted(name):
    assert not is_blacklisted(name)


def test_default_cascade_contains_no_blacklisted_model():
    assert not any(is_blacklisted(m.name) for m in build_cascade())


def test_blacklisted_override_entries_are_dropped_not_routed():
    specs = build_cascade(["gemini-2.5-pro", "gemini-3.5-flash-lite", "gemini-omni-flash"])
    assert [m.name for m in specs] == ["gemini-3.5-flash-lite"]


def test_strict_mode_raises_on_blacklisted_entry():
    with pytest.raises(BlacklistedModelError):
        build_cascade(["gemini-3.5-flash-lite", "gemini-3.1-pro"], strict=True)


def test_cascade_of_only_blacklisted_models_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        build_cascade(["gemini-2.5-pro", "gemini-2-flash"])


def test_client_rejects_blacklisted_spec_even_if_passed_directly():
    bad = [ModelSpec("gemini-2.5-pro", Tier.FALLBACK, 5, 20)]
    with pytest.raises(BlacklistedModelError):
        GeminiCascadeClient(["k1"], cascade=bad, client_factory=lambda k: None)


def test_blacklisted_model_is_never_called_even_when_everything_else_fails():
    script = lambda m, k, n: quota_429("rpd")
    client, calls, _ = make_client(script, backoff_sweeps=2)
    with pytest.raises(QuotaExhaustedError):
        run(client.generate("p", None))
    called = {m for (m, _k) in calls}
    assert called and not any(is_blacklisted(m) for m in called)
    assert not (called & BLACKLISTED_MODELS)


# ---------------------------------------------------------------------------
# 3. 429 classification
# ---------------------------------------------------------------------------
def test_classify_rpm_rpd_unknown():
    assert classify_quota_error(quota_429("rpm")) is QuotaKind.RPM
    assert classify_quota_error(quota_429("rpd")) is QuotaKind.RPD
    assert classify_quota_error(quota_429("none")) is QuotaKind.UNKNOWN


def test_classify_rpd_wins_when_both_windows_listed():
    """A single real 429 can list PerMinute AND PerDay — daily is stricter."""
    assert classify_quota_error(quota_429("rpm", both=True)) is QuotaKind.RPD


def test_retry_delay_does_not_decide_rpm_vs_rpd():
    """A per-DAY 429 can still say retryDelay '34s' — must stay RPD."""
    assert classify_quota_error(quota_429("rpd", retry_delay="34s")) is QuotaKind.RPD


def test_classify_falls_back_to_message_text():
    e = FakeAPIError(429, "RESOURCE_EXHAUSTED", None, "Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel")
    assert classify_quota_error(e) is QuotaKind.RPD


def test_pacific_midnight_is_dst_safe_and_positive():
    pt = ZoneInfo("America/Los_Angeles")
    # 23:00 PT -> exactly 1h to midnight (no DST transition that night).
    assert seconds_until(datetime(2026, 9, 28, 23, 0, tzinfo=pt)) == pytest.approx(3600)
    # Spring-forward night (Mar 8 2026, clocks skip 02:00->03:00): from 22:00 the wall
    # clock still reads exactly 2h to midnight — the skipped hour falls AFTER midnight,
    # so it doesn't shorten this particular window. Real elapsed time is 2h.
    assert seconds_until(datetime(2026, 3, 7, 22, 0, tzinfo=pt)) == pytest.approx(7200)
    # Fall-back day (Nov 1 2026, clocks repeat 01:00->01:00): from 00:30 the wall
    # clock has 23.5h left, but the repeated hour falls before this window's
    # midnight-to-midnight span already ended, so real elapsed time is 23.5h.
    assert seconds_until(datetime(2026, 11, 1, 0, 30, tzinfo=pt)) == pytest.approx(23.5 * 3600)


def seconds_until(dt):
    return gc.seconds_until_pacific_midnight(dt)


# ---------------------------------------------------------------------------
# 4. Routing / cascade order
# ---------------------------------------------------------------------------
def test_standard_request_goes_to_first_primary_model():
    client, calls, _ = make_client(ok_script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.5-flash-lite"
    assert calls == [("gemini-3.5-flash-lite", "k1")]
    assert res.attempts == 1


def test_round_robin_spreads_load_across_keys_on_same_model():
    client, calls, _ = make_client(ok_script)
    for _ in range(6):
        run(client.generate("p", None))
    keys_used = [k for (_m, k) in calls]
    assert keys_used == ["k1", "k2", "k3", "k1", "k2", "k3"]
    assert {m for (m, _k) in calls} == {"gemini-3.5-flash-lite"}


def test_rpm_429_rotates_to_next_key_on_SAME_model():
    def script(model, key, n):
        return quota_429("rpm") if key == "k1" else f"ok:{model}:{key}"

    client, calls, _ = make_client(script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.5-flash-lite"          # did NOT change model
    assert res.key_index == 1                            # rotated k1 -> k2
    assert calls == [("gemini-3.5-flash-lite", "k1"), ("gemini-3.5-flash-lite", "k2")]


def test_rpm_cooldown_prevents_reusing_the_limited_key():
    def script(model, key, n):
        return quota_429("rpm", retry_delay="30s") if key == "k1" else "ok"

    client, calls, vt = make_client(script)
    run(client.generate("p", None))
    calls.clear()
    for _ in range(4):
        run(client.generate("p", None))
    assert "k1" not in [k for (_m, k) in calls]          # still cooling, never retried


def test_rpm_cooldown_expires_and_key_returns_to_rotation():
    def script(model, key, n):
        return quota_429("rpm", retry_delay="10s") if (key == "k1" and n == 1) else "ok"

    client, calls, vt = make_client(script)
    run(client.generate("p", None))          # k1 429s once, k2 serves
    vt.now += 30                             # cooldown (floor..cap) elapses
    calls.clear()
    for _ in range(3):
        run(client.generate("p", None))
    assert "k1" in [k for (_m, k) in calls]


def test_rpd_on_all_keys_fails_over_to_next_model_in_hierarchy():
    def script(model, key, n):
        return quota_429("rpd") if model == "gemini-3.5-flash-lite" else f"ok:{model}"

    client, calls, _ = make_client(script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.1-flash-lite"          # 2nd primary, NOT a fallback
    assert [m for (m, _k) in calls].count("gemini-3.5-flash-lite") == 3   # tried all 3 keys


def test_both_primary_models_rpd_exhausted_cascades_into_fallback_tier_in_order():
    def script(model, key, n):
        if model in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
            return quota_429("rpd")
        return f"ok:{model}"

    client, _calls, _ = make_client(script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.8-flash"               # first fallback


def test_full_cascade_walks_fallbacks_in_exact_specified_order():
    order_seen = []

    def script(model, key, n):
        if key == "k1":
            order_seen.append(model)
        return quota_429("rpd")

    client, _c, _ = make_client(script, backoff_sweeps=1)
    with pytest.raises(QuotaExhaustedError):
        run(client.generate("p", None))
    assert order_seen == [m.name for m in build_cascade()]


def test_primary_models_are_fully_drained_before_any_fallback_is_touched():
    seen = []

    def script(model, key, n):
        seen.append(model)
        return quota_429("rpd") if model.endswith("flash-lite") and model != "gemini-2.5-flash-lite" else "ok"

    client, _c, _ = make_client(script)
    run(client.generate("p", None))
    fallback_idx = next(i for i, m in enumerate(seen) if m == "gemini-3.8-flash")
    assert all(m in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite") for m in seen[:fallback_idx])


def test_rpd_exhausted_model_is_skipped_on_subsequent_requests():
    def script(model, key, n):
        return quota_429("rpd") if model == "gemini-3.5-flash-lite" else "ok"

    client, calls, _ = make_client(script)
    run(client.generate("p", None))
    calls.clear()
    run(client.generate("p", None))
    # Parked until PT midnight -> zero wasted calls on the exhausted model.
    assert "gemini-3.5-flash-lite" not in [m for (m, _k) in calls]


def test_rpd_park_expires_after_pacific_midnight():
    """The daily quota genuinely resets at PT midnight — after that instant
    the model must be tried again (and succeed), not stay parked forever."""
    day = {"n": 0}

    def script(model, key, n):
        if model == "gemini-3.5-flash-lite" and day["n"] == 0:
            return quota_429("rpd")
        return "ok"

    client, calls, vt = make_client(script)
    r1 = run(client.generate("p", None))
    assert r1.model == "gemini-3.1-flash-lite"            # RPD-exhausted -> cascades onward
    day["n"] = 1                                          # simulate the new day's quota being fresh
    vt.now += 26 * 3600                                   # comfortably past any PT midnight
    calls.clear()
    r2 = run(client.generate("p", None))
    assert r2.model == "gemini-3.5-flash-lite"            # back to the top of the hierarchy
    assert calls == [("gemini-3.5-flash-lite", "k1")]     # succeeded on the very first try


def test_unclassifiable_429_is_treated_as_rpm_rotates_key_not_model():
    def script(model, key, n):
        return quota_429("none") if key == "k1" else "ok"

    client, calls, _ = make_client(script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.5-flash-lite" and res.key_index == 1


def test_single_key_rpm_429_goes_straight_to_next_model():
    def script(model, key, n):
        return quota_429("rpm") if model == "gemini-3.5-flash-lite" else "ok"

    client, _c, _ = make_client(script, keys=("only",))
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# 5. Non-quota errors
# ---------------------------------------------------------------------------
def test_404_retired_model_is_parked_and_never_retried():
    """The gemini-2.5-flash-lite incident: permanent 404 must not be re-hit."""
    def script(model, key, n):
        if model == "gemini-3.5-flash-lite":
            return FakeAPIError(404, "NOT_FOUND", None, "no longer available to new users")
        return "ok"

    client, calls, _ = make_client(script)
    res = run(client.generate("p", None))
    assert res.model == "gemini-3.1-flash-lite"
    first_model_calls = [c for c in calls if c[0] == "gemini-3.5-flash-lite"]
    assert len(first_model_calls) == 1                   # tried ONCE, not once per key
    calls.clear()
    run(client.generate("p", None))
    assert "gemini-3.5-flash-lite" not in [m for (m, _k) in calls]


def test_transient_error_retries_same_pair_then_succeeds():
    def script(model, key, n):
        return FakeAPIError(503, "UNAVAILABLE") if n == 1 else "ok"

    client, calls, vt = make_client(script, transient_retries=2)
    res = run(client.generate("p", None))
    assert res.attempts == 2 and res.key_index == 0
    assert vt.slept                                       # a backoff actually happened


def test_persistent_transient_error_moves_on_to_next_key():
    def script(model, key, n):
        return FakeAPIError(503, "UNAVAILABLE") if key == "k1" else "ok"

    client, _c, _ = make_client(script, transient_retries=1)
    res = run(client.generate("p", None))
    assert res.key_index == 1


def test_non_retryable_error_fails_fast_without_burning_keys_or_models():
    def script(model, key, n):
        return FakeAPIError(400, "INVALID_ARGUMENT", None, "bad prompt")

    client, calls, _ = make_client(script)
    with pytest.raises(FakeAPIError):
        run(client.generate("p", None))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6. Exponential backoff + final exception
# ---------------------------------------------------------------------------
def test_backoff_grows_exponentially_between_sweeps_then_raises():
    script = lambda m, k, n: quota_429("rpm", retry_delay="600s")  # always cooling; cap will clamp
    client, _c, vt = make_client(
        script, backoff_sweeps=4, backoff_base=2.0, backoff_cap=1000.0,
        rpm_cooldown_cap=1000.0, total_deadline=10_000,
    )
    with pytest.raises(QuotaExhaustedError):
        run(client.generate("p", None))
    waits = [s for s in vt.slept]
    assert len(waits) == 3                                # sweeps-1 waits
    assert waits == sorted(waits) and waits[-1] > waits[0]   # monotonically growing


def test_recovery_after_backoff_succeeds_once_a_cooldown_elapses():
    """First sweep: every pair 429s with a short RPM cooldown. The backoff
    wait before sweep 2 is sized to outlast that cooldown, so sweep 2
    succeeds without a 3rd sweep or a final exception."""
    def script(model, key, n):
        return quota_429("rpm", retry_delay="2s") if n == 1 else "ok"

    client, _c, vt = make_client(
        script, backoff_sweeps=3, total_deadline=10_000,
        rpm_cooldown_floor=2.0, rpm_cooldown_cap=2.0,
    )
    res = run(client.generate("p", None))
    assert res.response.startswith("ok")
    assert vt.slept                                       # it waited before recovering


def test_exhaustion_raises_quota_exhausted_with_cause_chained():
    client, _c, _ = make_client(lambda m, k, n: quota_429("rpd"), backoff_sweeps=2)
    with pytest.raises(QuotaExhaustedError) as ei:
        run(client.generate("p", None))
    assert str(ei.value) == "ALL_KEYS_EXHAUSTED"
    assert isinstance(ei.value.__cause__, FakeAPIError)


def test_total_deadline_bounds_the_wait():
    client, _c, vt = make_client(
        lambda m, k, n: quota_429("rpm", retry_delay="600s"),
        backoff_sweeps=10, backoff_base=50.0, backoff_cap=500.0,
        rpm_cooldown_cap=10_000, total_deadline=30.0,
    )
    with pytest.raises(QuotaExhaustedError):
        run(client.generate("p", None))
    assert sum(vt.slept) <= 30.0                          # never sleeps past the deadline


# ---------------------------------------------------------------------------
# 7. Local proactive RPM limiter
# ---------------------------------------------------------------------------
def test_local_rpm_cap_skips_to_next_key_with_zero_wasted_api_calls():
    """The LOCAL proactive limiter must stop a request before it ever hits
    the network once a (model, key) pair's 60s window is full — it should
    move to the next model instead of firing a request certain to 429."""
    cascade = [ModelSpec("gemini-3.5-flash-lite", Tier.PRIMARY, rpm=2, rpd=500),
               ModelSpec("gemini-3.1-flash-lite", Tier.PRIMARY, rpm=2, rpd=500)]
    client, calls, _ = make_client(ok_script, keys=("k1",), cascade=cascade, backoff_sweeps=1)
    seq = [run(client.generate("p", None)).model for _ in range(4)]
    # 2 RPM cap x 1 key per model: 2 requests land on the first model, then
    # the local cap sends the next 2 to the second model — zero API calls
    # wasted on a pair already known to be at capacity.
    assert seq == ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite",
                   "gemini-3.1-flash-lite", "gemini-3.1-flash-lite"]
    assert len(calls) == 4                                # no extra/wasted network calls

    # A 5th request has nowhere to go (both models fully capped, only 1
    # key) — it must exhaust rather than hang or silently fabricate a call.
    with pytest.raises(QuotaExhaustedError):
        run(client.generate("p", None))


# ---------------------------------------------------------------------------
# 8. KeyPool unit tests
# ---------------------------------------------------------------------------
def test_keypool_rejects_empty_key_list():
    with pytest.raises(ValueError):
        KeyPool([])
    with pytest.raises(ValueError):
        KeyPool(["", None])


def test_keypool_round_robin_cursor_is_per_model():
    p = KeyPool(["a", "b", "c"])
    assert [p.next_key_for("m1", 15) for _ in range(4)] == [0, 1, 2, 0]
    assert p.next_key_for("m2", 15) == 0                  # independent cursor


def test_keypool_skips_blocked_and_reports_none_when_all_blocked():
    p = KeyPool(["a", "b"])
    p.block("m", 0, 100, QuotaKind.RPM)
    assert p.next_key_for("m", 15) == 1
    p.block("m", 1, 100, QuotaKind.RPD)
    assert p.next_key_for("m", 15) is None
    assert p.blocked_kind("m", 1) is QuotaKind.RPD
