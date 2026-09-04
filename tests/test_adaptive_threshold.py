"""
Tests for ai_agent.DailyRequestTracker and get_effective_match_threshold —
the adaptive-threshold-under-quota-pressure feature.
"""

import ai_agent
import config


# ---- DailyRequestTracker ----------------------------------------------------
# Backed by MongoDB Atlas's `daily_request_count` collection (one document
# per UTC date) as of the Sept 2026 migration off a local JSON file — see
# tests/conftest.py's `isolated_state` fixture for the fresh in-memory
# (`mongomock`) database each test gets.

def test_tracker_starts_at_zero():
    tracker = ai_agent.DailyRequestTracker()
    assert tracker.get_today_count() == 0


def test_tracker_increments():
    tracker = ai_agent.DailyRequestTracker()
    tracker.increment()
    tracker.increment()
    tracker.increment()
    assert tracker.get_today_count() == 3


def test_tracker_persists_across_new_instances():
    ai_agent.DailyRequestTracker().increment()
    ai_agent.DailyRequestTracker().increment()
    assert ai_agent.DailyRequestTracker().get_today_count() == 2


def test_tracker_resets_on_new_utc_day():
    tracker = ai_agent.DailyRequestTracker()
    # Seed a stale prior day's count directly — today's key ("_id") is a
    # different date string, so it must not be affected by this at all.
    tracker._coll().update_one({"_id": "2020-01-01"}, {"$set": {"count": 999}}, upsert=True)
    assert tracker.get_today_count() == 0
    tracker.increment()
    assert tracker.get_today_count() == 1


def test_tracker_backend_error_degrades_to_zero_and_never_raises():
    class BrokenCollection:
        def find_one(self, *a, **kw):
            raise RuntimeError("Atlas unreachable")

        def update_one(self, *a, **kw):
            raise RuntimeError("Atlas unreachable")

    tracker = ai_agent.DailyRequestTracker(collection=BrokenCollection())
    assert tracker.get_today_count() == 0  # must not raise
    tracker.increment()  # must not raise either


# ---- get_effective_match_threshold ------------------------------------------

def _set_today_count(monkeypatch, n):
    monkeypatch.setattr(ai_agent._daily_request_tracker, "get_today_count", lambda: n)


def test_below_trigger_ratio_returns_base_threshold(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_TRIGGER_RATIO", 0.7)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_HARD_CAP", 90)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 50)

    assert ai_agent.get_effective_match_threshold() == 60


def test_at_trigger_boundary_returns_base_threshold(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_TRIGGER_RATIO", 0.7)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_HARD_CAP", 90)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 70)

    assert ai_agent.get_effective_match_threshold() == 60


def test_ramp_halfway_through_window(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_TRIGGER_RATIO", 0.7)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_HARD_CAP", 90)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 85)  # halfway between 70% and 100%

    assert ai_agent.get_effective_match_threshold() == 75.0  # 60 + 0.5 * (90-60)


def test_at_full_quota_hits_hard_cap(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_TRIGGER_RATIO", 0.7)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_HARD_CAP", 90)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 100)

    assert ai_agent.get_effective_match_threshold() == 90.0


def test_over_quota_stays_capped(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_TRIGGER_RATIO", 0.7)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_HARD_CAP", 90)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 150)  # 150% of quota

    assert ai_agent.get_effective_match_threshold() == 90.0


def test_disabled_feature_always_returns_base(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", False)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 100)
    _set_today_count(monkeypatch, 150)

    assert ai_agent.get_effective_match_threshold() == 60


def test_zero_quota_configured_is_a_noop(monkeypatch):
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "ADAPTIVE_THRESHOLD_ENABLED", True)
    monkeypatch.setattr(config, "GEMINI_ESTIMATED_DAILY_QUOTA", 0)
    _set_today_count(monkeypatch, 999)

    assert ai_agent.get_effective_match_threshold() == 60


def test_only_successful_calls_increment_the_counter(monkeypatch, make_fake_response):
    """The counter must only reflect genuinely successful Gemini calls,
    not failed attempts."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])

    monkeypatch.setattr(ai_agent, "_call_gemini_once", lambda *a, **kw: make_fake_response(
        parsed=ai_agent.ProjectScoreSchema(match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1),
    ))
    ai_agent.score_project("T", "D")
    assert ai_agent._daily_request_tracker.get_today_count() == 1

    def always_fail(*a, **kw):
        raise RuntimeError("simulated non-retryable failure")

    monkeypatch.setattr(ai_agent, "_call_gemini_once", always_fail)
    try:
        ai_agent.score_project("T2", "D2")
    except Exception:
        pass
    assert ai_agent._daily_request_tracker.get_today_count() == 1  # unchanged
