"""
tests/conftest.py
------------------
Two things MUST happen here, in this order, before any test module does
`import ai_agent` / `import config` / etc:

1. The project root must be on sys.path — the app modules live at the
   repo root, not in an installed package, and pytest's default import
   mode only auto-adds the `tests/` directory itself (since it has no
   __init__.py), not its parent.

2. GEMINI_API_KEYS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and (as of the
   Sept 2026 MongoDB migration) MONGODB_URI must already be set in the
   environment — config.py calls _require()/raises EnvironmentError
   IMMEDIATELY at import time if they're missing (see config.py). Since
   conftest.py is imported by pytest before it collects and imports any
   test file in this directory, setting them here guarantees every test
   module's `import ai_agent` (which transitively imports config)
   succeeds regardless of what's in the real environment or .env file.
   MONGODB_URI's value here is never actually connected to — see the
   `mongomock` fixture below, which replaces the real connection with an
   in-memory fake before any test touches a collection.

Fixtures below give each test an ISOLATED, disposable in-memory MongoDB
(via `mongomock`, no real Atlas connection ever made in tests) plus a
fresh copy of every ai_agent singleton that used to be file-backed
(score cache, daily request counter) — nothing a test does here ever
touches a real database or the project's real files.
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
os.environ.setdefault("MONGODB_URI", "mongodb://not-actually-used-see-mongomock-fixture/")

import pytest  # noqa: E402  (must come after the sys.path fix above)
import mongomock  # noqa: E402

import ai_agent  # noqa: E402
import config  # noqa: E402
import db  # noqa: E402
import health_server  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """
    Applied to EVERY test automatically. Swaps db.py's live database
    handle for a brand-new `mongomock` in-memory database (so every test
    starts with completely empty collections, no real network call ever
    attempted), and rebuilds ai_agent's module-level singletons
    (ScoreCache, DailyRequestTracker) so they pick up the fresh mock
    database rather than whatever they were constructed with at import
    time or by a previous test.

    Without this, tests would either fail outright (no real MongoDB
    reachable in CI) or, worse, all share ONE mock database across the
    whole test run — polluting each other's assertions depending on
    execution order.
    """
    fake_client = mongomock.MongoClient()
    fake_db = fake_client["mosbot_test"]
    monkeypatch.setattr(db, "_client", fake_client)
    monkeypatch.setattr(db, "_db", fake_db)
    # TTL/uniqueness indexes are a real-MongoDB-only concept mongomock
    # doesn't fully support — tests care about the read/write behavior of
    # the functions above, not whether an index exists, so this is
    # skipped entirely rather than letting _ensure_indexes silently
    # swallow a mongomock incompatibility on every single test.
    monkeypatch.setattr(db, "_indexes_ensured", True)

    # These two are module-level singletons in ai_agent.py, each built
    # ONCE at import time (or by a previous test's monkeypatch) — rebuilt
    # here so they resolve db.get_collection() against the fresh mock
    # database above rather than a stale reference.
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
