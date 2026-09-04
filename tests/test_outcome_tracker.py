"""
Tests for outcome_tracker.py — records Won/Lost outcomes from Telegram
button taps.
"""

import outcome_tracker


def test_record_win_and_loss():
    assert outcome_tracker.record_outcome("proj123", title="Build a scraper", outcome="won") is True
    assert outcome_tracker.record_outcome("proj456", title="Design a logo", outcome="lost") is True

    stats = outcome_tracker.get_stats()
    assert stats == {"won": 1, "lost": 1, "total": 2}


def test_correction_overwrites_previous_outcome():
    outcome_tracker.record_outcome("proj123", title="Build a scraper", outcome="won")
    outcome_tracker.record_outcome("proj123", title="Build a scraper", outcome="lost")

    stats = outcome_tracker.get_stats()
    assert stats == {"won": 0, "lost": 1, "total": 1}


def test_invalid_outcome_value_is_rejected():
    assert outcome_tracker.record_outcome("proj789", title="X", outcome="maybe") is False
    assert outcome_tracker.get_stats() == {"won": 0, "lost": 0, "total": 0}


def test_empty_project_id_is_rejected():
    assert outcome_tracker.record_outcome("", title="X", outcome="won") is False


def test_get_stats_on_empty_history():
    assert outcome_tracker.get_stats() == {"won": 0, "lost": 0, "total": 0}


def test_get_stats_never_raises_on_backend_error(monkeypatch):
    """db.get_outcome_stats() (which outcome_tracker.get_stats() delegates
    straight to) is fail-safe by its own design — this exercises that
    fail-safety through a broken underlying collection, rather than
    bypassing it, so the test reflects a real backend outage rather than
    a monkeypatch that defeats the safety net it's meant to verify."""
    import db

    class BrokenCollection:
        def count_documents(self, *a, **kw):
            raise RuntimeError("Atlas unreachable")

    monkeypatch.setattr(db, "get_collection", lambda name: BrokenCollection())

    assert outcome_tracker.get_stats() == {"won": 0, "lost": 0, "total": 0}
