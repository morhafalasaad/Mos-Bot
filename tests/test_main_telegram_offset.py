import builtins

import main


def test_save_telegram_offset_handles_unexpected_write_errors(monkeypatch):
    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def write(self, value):
            raise TypeError("write failed")

    def fake_open(*args, **kwargs):
        return FakeFile()

    monkeypatch.setattr(main, "_TELEGRAM_OFFSET_FILE", "telegram_update_offset.txt")
    monkeypatch.setattr(builtins, "open", fake_open)

    main._save_telegram_offset(42)
