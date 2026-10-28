"""
Tests for notifier.send_telegram_message's exponential-backoff retry on
transient Telegram failures (HTTP 5xx / 429 / timeouts / connection resets).

requests.post and the sleep hook are both mocked: no network, no waiting.
"""

import requests

import config
import notifier


class _Resp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _install(monkeypatch, outcomes):
    """Makes requests.post return/raise each item of `outcomes` in order;
    returns (calls, sleeps) lists for assertions."""
    calls, sleeps = [], []
    queue = list(outcomes)

    def fake_post(url, data=None, timeout=None):
        calls.append(dict(data))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(notifier.requests, "post", fake_post)
    monkeypatch.setattr(notifier, "_sleep", sleeps.append)
    monkeypatch.setattr(config, "TELEGRAM_RETRY_DELAYS", (2.0, 4.0, 8.0, 16.0))
    return calls, sleeps


def test_success_first_try_does_not_sleep(monkeypatch):
    calls, sleeps = _install(monkeypatch, [_Resp(200)])
    assert notifier.send_telegram_message("hi") is True
    assert len(calls) == 1
    assert sleeps == []


def test_retries_502_then_succeeds(monkeypatch):
    calls, sleeps = _install(monkeypatch, [_Resp(502, "Bad Gateway"), _Resp(502), _Resp(200)])
    assert notifier.send_telegram_message("hi") is True
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]  # exponential backoff between attempts


def test_retries_read_timeout_then_succeeds(monkeypatch):
    calls, sleeps = _install(monkeypatch, [requests.exceptions.ReadTimeout("slow"), _Resp(200)])
    assert notifier.send_telegram_message("hi") is True
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_retries_connection_error_then_succeeds(monkeypatch):
    calls, sleeps = _install(monkeypatch, [requests.exceptions.ConnectionError("reset"), _Resp(200)])
    assert notifier.send_telegram_message("hi") is True
    assert len(calls) == 2


def test_retries_429_flood_control(monkeypatch):
    calls, sleeps = _install(monkeypatch, [_Resp(429), _Resp(200)])
    assert notifier.send_telegram_message("hi") is True
    assert len(calls) == 2


def test_gives_up_after_max_retries_without_raising(monkeypatch, caplog):
    calls, sleeps = _install(monkeypatch, [_Resp(503)] * 5)
    assert notifier.send_telegram_message("hi") is False
    assert len(calls) == 5  # 1 initial + 4 retries
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert "giving up" in caplog.text


def test_gives_up_after_repeated_timeouts_without_raising(monkeypatch):
    calls, sleeps = _install(monkeypatch, [requests.exceptions.ReadTimeout("t")] * 5)
    assert notifier.send_telegram_message("hi") is False
    assert len(calls) == 5


def test_non_transient_4xx_is_not_retried(monkeypatch):
    calls, sleeps = _install(monkeypatch, [_Resp(403, "Forbidden")])
    assert notifier.send_telegram_message("hi") is False
    assert len(calls) == 1
    assert sleeps == []


def test_400_markdown_fallback_still_works(monkeypatch):
    calls, sleeps = _install(monkeypatch, [_Resp(400, "can't parse entities"), _Resp(200)])
    assert notifier.send_telegram_message("bad_markdown") is True
    assert "parse_mode" in calls[0]
    assert "parse_mode" not in calls[1]


def test_unexpected_exception_never_propagates(monkeypatch):
    calls, sleeps = _install(monkeypatch, [RuntimeError("boom")])
    assert notifier.send_telegram_message("hi") is False


def test_retry_delays_env_parsing():
    assert config._parse_delays("1,2,3", (9.0,)) == (1.0, 2.0, 3.0)
    assert config._parse_delays("oops", (9.0,)) == (9.0,)
    assert config._parse_delays("-1,2", (9.0,)) == (9.0,)
    assert config._parse_delays("", (9.0,)) == (9.0,)
