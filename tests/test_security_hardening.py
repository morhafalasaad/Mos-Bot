"""Tests for the Oct 2026 advisory hardening: log redaction, startup
trust-store check, and blank-secret rejection."""

import logging

import pytest
import requests

import config
import secure_logging
import security_check

FAKE_TOKEN = "7654321098:AAFakeTokenForLeakDemo_abcdefghijklmnop"


def _format(msg, *args, exc_info=None):
    rec = logging.LogRecord("t", logging.ERROR, __file__, 1, msg, args, exc_info)
    return secure_logging.RedactingFormatter("%(message)s").format(rec)


def test_telegram_token_redacted_from_requests_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    try:
        requests.post(f"https://127.0.0.1:9/bot{FAKE_TOKEN}/sendMessage", timeout=2)
    except requests.exceptions.RequestException as exc:
        out = _format("Telegram request failed: %s", exc)
    assert FAKE_TOKEN not in out and "AAFakeToken" not in out
    assert "<redacted>" in out


def test_token_redacted_in_traceback(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    try:
        raise RuntimeError(f"failed https://api.telegram.org/bot{FAKE_TOKEN}/getUpdates")
    except RuntimeError:
        import sys
        out = _format("boom", exc_info=sys.exc_info())
    assert FAKE_TOKEN not in out


def test_url_credentials_and_keys_redacted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "key-one-secret,key-two-secret")
    out = _format("a %s b %s c %s", "mongodb+srv://bob:hunter2pw@cluster0.x.net/", "key-two-secret", "Bearer abcdef123456")
    assert "hunter2pw" not in out and "key-two-secret" not in out and "abcdef123456" not in out


def test_mask_url_credentials():
    assert secure_logging.mask_url_credentials("http://user:pass@host:8080") == "http://host:8080"
    assert secure_logging.mask_url_credentials("socks5://host:1080") == "socks5://host:1080"


def test_plain_text_untouched():
    assert _format("Producer: 3 new projects") == "Producer: 3 new projects"


def test_old_certifi_fails_closed(monkeypatch):
    monkeypatch.setattr(security_check, "_installed", lambda n: "2026.7.22" if n == "certifi" else None)
    with pytest.raises(RuntimeError, match="eMudhra"):
        security_check.check_environment()


def test_fixed_certifi_passes(monkeypatch):
    monkeypatch.setattr(security_check, "_installed", lambda n: "2026.10.23" if n == "certifi" else None)
    security_check.check_environment()


def test_vulnerable_brotli_logs_critical(monkeypatch, caplog):
    vers = {"certifi": "2026.10.23", "Brotli": "1.2.0"}
    monkeypatch.setattr(security_check, "_installed", lambda n: vers.get(n))
    with caplog.at_level(logging.CRITICAL, logger="security_check"):
        security_check.check_environment()
    assert any("CVE-2026-111497" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("blank", ["", "   "])
def test_require_rejects_blank(monkeypatch, blank):
    monkeypatch.setenv("SOME_SECRET_FOR_TEST", blank)
    with pytest.raises(EnvironmentError):
        config._require("SOME_SECRET_FOR_TEST")
