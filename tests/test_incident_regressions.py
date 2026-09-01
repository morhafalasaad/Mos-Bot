"""
Regression tests for two production incidents:

1. gemini-2.5-flash-lite was permanently retired by Google (404 NOT_FOUND
   on every call, not a transient error) but remained in the default
   model fallback cascade — every cascade that fell through past the
   first model wasted an attempt on something guaranteed to fail.
2. Telegram's getUpdates returned HTTP 409 Conflict (another process
   already long-polling the same bot token) and was being treated as a
   generic transient error — retried every 5s with a generic message,
   even though fast retries can't make a conflict resolve any sooner.
"""

import time

import config
import main


def test_retired_model_not_in_default_cascade():
    assert "gemini-2.5-flash-lite" not in config.GEMINI_MODEL_CASCADE


def test_retired_model_not_in_rpm_limits():
    assert "gemini-2.5-flash-lite" not in config.MODEL_RPM_LIMITS


def test_default_cascade_still_has_a_fast_lite_model_first():
    """The whole point of the cascade is cheapest/fastest model first —
    confirm removing the retired entry didn't also remove the model that
    should still lead it."""
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
