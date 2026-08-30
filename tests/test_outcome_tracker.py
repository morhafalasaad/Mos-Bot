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


def test_get_stats_never_raises_on_corrupt_file(monkeypatch):
    import config
    path = config.OUTCOME_LOG_FILE
    with open(path, "w", encoding="utf-8") as f:
        f.write("not valid json {{{")

    assert outcome_tracker.get_stats() == {"won": 0, "lost": 0, "total": 0}
