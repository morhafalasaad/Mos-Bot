"""
Tests for main.py's _load_telegram_offset / _save_telegram_offset — the
small persistence helpers behind the Telegram feedback listener's
long-poll cursor. Backed by MongoDB Atlas's `bot_state` collection (see
db.py) as of the Sept 2026 migration off a local text file. Both must be
fail-safe against ANY exception type, matching every other persistence
helper in this codebase (ScoreCache, DailyRequestTracker, outcome_tracker,
repost_detector).
"""

import db
import main


def test_save_offset_writes_correctly():
    main._save_telegram_offset(42)
    assert db.get_state(main._TELEGRAM_OFFSET_STATE_KEY) == 42


def test_load_offset_reads_back_saved_value():
    main._save_telegram_offset(99)
    assert main._load_telegram_offset() == 99


def test_load_offset_missing_value_returns_zero():
    assert main._load_telegram_offset() == 0


def test_save_telegram_offset_handles_unexpected_write_errors(monkeypatch):
    """Regression test: any exception during the write must be swallowed,
    not propagated — a non-critical best-effort persistence write must
    never be allowed to crash the Telegram feedback listener thread."""

    def broken_set_state(*args, **kwargs):
        raise TypeError("write failed")

    monkeypatch.setattr(db, "set_state", broken_set_state)

    main._save_telegram_offset(123)  # must not raise


def test_load_telegram_offset_handles_unexpected_read_errors(monkeypatch):
    """Same regression, read side: any exception type while reading must
    degrade to offset 0, not propagate."""

    def broken_get_state(*args, **kwargs):
        raise TypeError("read failed")

    monkeypatch.setattr(db, "get_state", broken_get_state)

    assert main._load_telegram_offset() == 0  # must not raise
