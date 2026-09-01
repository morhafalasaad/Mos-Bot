"""
tests/conftest.py
------------------
Two things MUST happen here, in this order, before any test module does
`import ai_agent` / `import config` / etc:

1. The project root must be on sys.path — the app modules live at the
   repo root, not in an installed package, and pytest's default import
   mode only auto-adds the `tests/` directory itself (since it has no
   __init__.py), not its parent.

2. GEMINI_API_KEYS, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID must already
   be set in the environment — config.py calls _require()/raises
   EnvironmentError IMMEDIATELY at import time if they're missing (see
   config.py). Since conftest.py is imported by pytest before it collects
   and imports any test file in this directory, setting them here
   guarantees every test module's `import ai_agent` (which transitively
   imports config) succeeds regardless of what's in the real environment
   or .env file.

Fixtures below give each test an ISOLATED, disposable copy of every
on-disk state file this codebase uses (score cache, daily request
counter, outcomes log, repost history) — nothing a test does here ever
touches the real files in the project root.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEYS", "test-gemini-key-one,test-gemini-key-two")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-telegram-bot-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123456789")

import pytest  # noqa: E402  (must come after the sys.path fix above)

import ai_agent  # noqa: E402
import config  # noqa: E402
import health_server  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state_files(tmp_path, monkeypatch):
    """
    Applied to EVERY test automatically. Points every on-disk state file
    this codebase writes at a throwaway path under pytest's per-test
    tmp_path, and rebuilds ai_agent's module-level singletons (ScoreCache,
    DailyRequestTracker) so they pick up those throwaway paths instead of
    whatever they were constructed with at import time.

    Without this, tests would read/write score_cache.json,
    daily_request_count.json, outcomes.json, and repost_history.json in
    the actual project directory — polluting real data and making test
    runs interfere with each other (and with a real running bot, if one
    happened to be pointed at the same directory).
    """
    monkeypatch.setattr(config, "SCORE_CACHE_FILE", str(tmp_path / "score_cache.json"))
    monkeypatch.setattr(config, "DAILY_REQUEST_COUNT_FILE", str(tmp_path / "daily_request_count.json"))
    monkeypatch.setattr(config, "OUTCOME_LOG_FILE", str(tmp_path / "outcomes.json"))
    monkeypatch.setattr(config, "REPOST_HISTORY_FILE", str(tmp_path / "repost_history.json"))

    # These two are module-level singletons in ai_agent.py, each built
    # ONCE at import time from whatever config.SCORE_CACHE_FILE /
    # config.DAILY_REQUEST_COUNT_FILE was at that moment — monkeypatching
    # the config values above doesn't retroactively change an
    # already-constructed instance's stored path, so they're rebuilt here.
    monkeypatch.setattr(ai_agent, "_score_cache", ai_agent.ScoreCache())
    monkeypatch.setattr(ai_agent, "_daily_request_tracker", ai_agent.DailyRequestTracker())

    # health_server keeps its own module-level status dict (thread
    # heartbeats, queue depth) and start time — reset both so one test's
    # update_status() calls can never leak into another's assertions,
    # regardless of test execution order.
    monkeypatch.setattr(health_server, "_status", {
        "producer_last_heartbeat": None,
        "consumer_last_heartbeat": None,
        "feedback_last_heartbeat": None,
        "queue_size": None,
    })
    monkeypatch.setattr(health_server, "_started_at", time.time())

    yield


@pytest.fixture
def fake_usage():
    """A minimal stand-in for Gemini SDK response.usage_metadata."""
    class FakeUsage:
        prompt_token_count = 10
        candidates_token_count = 10
        total_token_count = 20
    return FakeUsage()


@pytest.fixture
def make_fake_response(fake_usage):
    """
    Factory fixture: make_fake_response(parsed=..., text=...) returns an
    object shaped enough like the real google-genai SDK's response object
    for ai_agent's parsing code (response.parsed / response.text /
    response.usage_metadata) to work against it unmodified.
    """
    def _make(parsed=None, text=""):
        class FakeResponse:
            usage_metadata = fake_usage

        resp = FakeResponse()
        resp._parsed = parsed
        resp.text = text
        # `parsed` is a property on the real SDK response; replicate that
        # shape rather than a plain attribute, since ai_agent's code does
        # `response.parsed` (not `response.parsed()`).
        type(resp).parsed = property(lambda self: self._parsed)
        return resp
    return _make
