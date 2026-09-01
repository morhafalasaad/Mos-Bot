"""
Tests for main.py's _load_telegram_offset / _save_telegram_offset — the
small persistence helpers behind the Telegram feedback listener's
long-poll cursor. Both must be fail-safe against ANY exception type, not
just OSError, matching every other persistence helper in this codebase
(ScoreCache, DailyRequestTracker, outcome_tracker, repost_detector).
"""

import builtins

import main


def test_save_offset_writes_correctly(tmp_path, monkeypatch):
    offset_file = tmp_path / "telegram_update_offset.txt"
    monkeypatch.setattr(main, "_TELEGRAM_OFFSET_FILE", str(offset_file))

    main._save_telegram_offset(42)

    assert offset_file.read_text().strip() == "42"


def test_load_offset_reads_back_saved_value(tmp_path, monkeypatch):
    offset_file = tmp_path / "telegram_update_offset.txt"
    monkeypatch.setattr(main, "_TELEGRAM_OFFSET_FILE", str(offset_file))

    main._save_telegram_offset(99)
    assert main._load_telegram_offset() == 99


def test_load_offset_missing_file_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_TELEGRAM_OFFSET_FILE", str(tmp_path / "does_not_exist.txt"))
    assert main._load_telegram_offset() == 0


def test_load_offset_corrupt_content_returns_zero(tmp_path, monkeypatch):
    offset_file = tmp_path / "telegram_update_offset.txt"
    offset_file.write_text("not a number")
    monkeypatch.setattr(main, "_TELEGRAM_OFFSET_FILE", str(offset_file))

    assert main._load_telegram_offset() == 0  # must not raise ValueError


def test_save_telegram_offset_handles_unexpected_write_errors(monkeypatch):
    """Regression test: a TypeError (or any other exception type) during
    the write must be swallowed, not just OSError — a non-critical
    best-effort persistence write must never be allowed to crash the
    Telegram feedback listener thread."""

    class BrokenFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def write(self, *args, **kwargs):
            raise TypeError("write failed")

    def broken_open(*args, **kwargs):
        return BrokenFile()

    monkeypatch.setattr(builtins, "open", broken_open)

    main._save_telegram_offset(123)  # must not raise


def test_load_telegram_offset_handles_unexpected_read_errors(monkeypatch):
    """Same regression, read side: any exception type while reading must
    degrade to offset 0, not propagate."""

    class BrokenFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, *args, **kwargs):
            raise TypeError("read failed")

    def broken_open(*args, **kwargs):
        return BrokenFile()

    monkeypatch.setattr(builtins, "open", broken_open)

    assert main._load_telegram_offset() == 0  # must not raise
