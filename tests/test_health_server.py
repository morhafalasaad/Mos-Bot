"""
Tests for health_server.py — the /status JSON snapshot, heartbeat
staleness detection, and thread-safe update_status().
"""

import time

import config
import health_server


def test_no_heartbeat_yet_reports_none_health():
    snapshot = health_server.get_status_snapshot()
    assert snapshot["producer"]["seconds_since_heartbeat"] is None
    assert snapshot["producer"]["healthy"] is None


def test_fresh_heartbeat_is_healthy():
    health_server.update_status(producer_last_heartbeat=time.time())
    snapshot = health_server.get_status_snapshot()
    assert snapshot["producer"]["healthy"] is True
    assert snapshot["producer"]["seconds_since_heartbeat"] < 5


def test_stale_producer_heartbeat_is_unhealthy(monkeypatch):
    monkeypatch.setattr(config, "POLL_INTERVAL_MAX", 60)  # stale_after = 2*60+120 = 240s
    stale_time = time.time() - 500  # well past the 240s threshold
    health_server.update_status(producer_last_heartbeat=stale_time)

    snapshot = health_server.get_status_snapshot()
    assert snapshot["producer"]["healthy"] is False


def test_stale_consumer_heartbeat_is_unhealthy(monkeypatch):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 60)  # stale_after = 60+120 = 180s
    stale_time = time.time() - 400
    health_server.update_status(consumer_last_heartbeat=stale_time)

    snapshot = health_server.get_status_snapshot()
    assert snapshot["consumer"]["healthy"] is False


def test_stale_feedback_heartbeat_is_unhealthy(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_FEEDBACK_POLL_TIMEOUT", 10)  # stale_after = 10+60 = 70s
    stale_time = time.time() - 200
    health_server.update_status(feedback_last_heartbeat=stale_time)

    snapshot = health_server.get_status_snapshot()
    assert snapshot["telegram_feedback"]["healthy"] is False


def test_queue_size_is_reported():
    health_server.update_status(queue_size=7)
    snapshot = health_server.get_status_snapshot()
    assert snapshot["queue_size"] == 7


def test_gemini_and_outcomes_sections_present():
    snapshot = health_server.get_status_snapshot()
    assert "requests_today" in snapshot["gemini"]
    assert "effective_match_threshold" in snapshot["gemini"]
    assert snapshot["outcomes"] == {"won": 0, "lost": 0, "total": 0}


def test_update_status_never_raises_on_bad_input():
    # Passing something update_status doesn't expect must not crash —
    # this is a low-stakes dashboard update, never allowed to take down a
    # worker loop.
    health_server.update_status(some_unexpected_kwarg=object())  # must not raise


def test_snapshot_never_raises_even_if_ai_agent_import_fails(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "ai_agent":
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    snapshot = health_server.get_status_snapshot()  # must not raise
    assert snapshot["gemini"] == {"error": "unavailable"}


def test_uptime_increases_over_time():
    snapshot1 = health_server.get_status_snapshot()
    time.sleep(0.05)
    snapshot2 = health_server.get_status_snapshot()
    assert snapshot2["uptime_seconds"] >= snapshot1["uptime_seconds"]
