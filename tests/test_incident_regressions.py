"""
Regression tests for two production incidents:

1. gemini-2.5-flash-lite was, at one point, permanently retired by Google
   (404 NOT_FOUND on every call, not a transient error) while it remained
   in the default model fallback cascade — every cascade that fell
   through past the first model wasted an attempt on something guaranteed
   to fail. Google has since made the model available again, and the
   current requirements explicitly specify it as the final fallback-tier
   entry — so the regression this suite now guards against is no longer
   "keep this specific model name out of the static config list forever"
   (that would just be wrong today), but the underlying MECHANISM: a
   model that starts returning a permanent 404/NOT_FOUND must be detected
   at runtime and automatically skipped for a cooldown period, rather
   than being retried on every single request. See
   gemini_client.is_model_gone_error() / KeyPool.mark_model_dead().
2. Telegram's getUpdates returned HTTP 409 Conflict (another process
   already long-polling the same bot token) and was being treated as a
   generic transient error — retried every 5s with a generic message,
   even though fast retries can't make a conflict resolve any sooner.
"""

import time

import config
import main
import gemini_client


def test_retired_model_is_still_a_valid_configured_fallback_today():
    """gemini-2.5-flash-lite is Google-available again and is explicitly
    part of the specified hierarchy (final fallback-tier entry) — it must
    NOT be excluded from the default cascade or its RPM table."""
    assert "gemini-2.5-flash-lite" in config.GEMINI_MODEL_CASCADE
    assert "gemini-2.5-flash-lite" in config.MODEL_RPM_LIMITS
    assert config.GEMINI_MODEL_CASCADE[-1] == "gemini-2.5-flash-lite"


def test_a_model_that_starts_404ing_is_detected_and_parked_at_runtime():
    """The ACTUAL regression protection: if any model in the cascade ever
    gets retired again, is_model_gone_error() must recognize the 404/
    NOT_FOUND shape Google used last time, so GeminiCascadeClient parks it
    (KeyPool.mark_model_dead) instead of wasting a request on it on every
    single future call."""
    class _FakeRetiredModelError(Exception):
        code = 404
        status = "NOT_FOUND"

        def __str__(self):
            return "404 NOT_FOUND. models/gemini-2.5-flash-lite is no longer available to new users"

    exc = _FakeRetiredModelError()
    assert gemini_client.is_model_gone_error(exc)

    pool = gemini_client.KeyPool(["k1"])
    assert not pool.model_is_dead("gemini-2.5-flash-lite")
    pool.mark_model_dead("gemini-2.5-flash-lite", seconds=3600)
    assert pool.model_is_dead("gemini-2.5-flash-lite")


def test_default_cascade_still_has_a_fast_lite_model_first():
    """The whole point of the cascade is highest-quota (RPM/RPD) model
    first — confirm the primary tier still leads."""
    assert config.GEMINI_MODEL_CASCADE[0] == "gemini-3.5-flash-lite"


def test_default_cascade_has_at_least_one_fallback():
    assert len(config.GEMINI_MODEL_CASCADE) >= 2


def test_409_conflict_backs_off_longer_than_generic_error(monkeypatch):
    """The 409-specific backoff must be longer than the generic 5s
    transient-error backoff — retrying fast doesn't help a conflict
    resolve any sooner."""
    assert config.TELEGRAM_CONFLICT_BACKOFF_SECONDS > 5


def test_409_response_triggers_conflict_specific_backoff(monkeypatch, caplog):
    """End-to-end: a 409 response from getUpdates must sleep for
    TELEGRAM_CONFLICT_BACKOFF_SECONDS (not the generic 5s) and log a
    message that actually explains the real cause, not just the raw
    status code."""
    monkeypatch.setattr(config, "TELEGRAM_CONFLICT_BACKOFF_SECONDS", 0.01)  # keep the test fast

    call_count = {"n": 0}

    class FakeResponse:
        status_code = 409
        text = '{"ok":false,"error_code":409,"description":"Conflict"}'

        def json(self):
            return {"result": []}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt("stop the loop after one 409 iteration")
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)
    monkeypatch.setattr(main, "_load_telegram_offset", lambda: 0)
    monkeypatch.setattr(main, "_save_telegram_offset", lambda offset: None)

    sleep_calls = []
    real_sleep = time.sleep

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        real_sleep(0)  # don't actually wait in the test

    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    try:
        main.telegram_feedback_loop()
    except KeyboardInterrupt:
        pass

    assert config.TELEGRAM_CONFLICT_BACKOFF_SECONDS in sleep_calls
    assert any("another process" in record.message.lower() for record in caplog.records)
