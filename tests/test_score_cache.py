"""
Tests for ai_agent.ScoreCache — caches a project's score result keyed by
(title, full description, current MY_SKILLS), so re-evaluating identical
content skips a fresh Gemini call. Backed by MongoDB Atlas's `score_cache`
collection as of the Sept 2026 migration off a local JSON file — see
tests/conftest.py's `isolated_state` fixture for how each test gets a
fresh in-memory (`mongomock`) database.
"""

import ai_agent
import config


def test_cache_miss_on_first_lookup():
    cache = ai_agent.ScoreCache()
    assert cache.get("Some title", "Some description") is None


def test_set_then_get_returns_stored_fields():
    cache = ai_agent.ScoreCache()
    cache.set("T", "D", {
        "match_score": 77, "reasoning": "good fit",
        "matched_skills": ["python"], "missing_skills": [],
        "suggested_price": "$100", "delivery_days": 3,
    })
    assert cache.get("T", "D") == {
        "match_score": 77, "reasoning": "good fit",
        "matched_skills": ["python"], "missing_skills": [],
        "suggested_price": "$100", "delivery_days": 3,
    }


def test_different_description_is_a_cache_miss():
    cache = ai_agent.ScoreCache()
    cache.set("T", "Description A", {"match_score": 50, "reasoning": "x"})
    assert cache.get("T", "Description B") is None


def test_my_skills_change_invalidates_cache(monkeypatch):
    cache = ai_agent.ScoreCache()
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    cache.set("T", "D", {"match_score": 90, "reasoning": "x"})
    assert cache.get("T", "D") is not None

    monkeypatch.setattr(config, "MY_SKILLS", ["php"])
    assert cache.get("T", "D") is None, "changing MY_SKILLS must invalidate previously-cached scores"


def test_disabled_cache_always_misses(monkeypatch):
    monkeypatch.setattr(config, "SCORE_CACHE_ENABLED", False)
    cache = ai_agent.ScoreCache()
    cache.set("T", "D", {"match_score": 90, "reasoning": "x"})
    assert cache.get("T", "D") is None


def test_max_entries_eviction():
    cache = ai_agent.ScoreCache(max_entries=3)
    for i in range(5):
        cache.set(f"Title {i}", f"Description {i}", {"match_score": i, "reasoning": "x"})

    hits = sum(1 for i in range(5) if cache.get(f"Title {i}", f"Description {i}") is not None)
    assert hits <= 3, "cache should have evicted down to max_entries once over the cap"


def test_backend_error_on_get_degrades_to_miss_not_crash():
    class BrokenCollection:
        def find_one(self, *a, **kw):
            raise RuntimeError("Atlas unreachable")

    cache = ai_agent.ScoreCache(collection=BrokenCollection())
    assert cache.get("T", "D") is None  # must not raise


def test_backend_error_on_set_is_swallowed_not_raised():
    class BrokenCollection:
        def update_one(self, *a, **kw):
            raise RuntimeError("Atlas unreachable")

    cache = ai_agent.ScoreCache(collection=BrokenCollection())
    cache.set("T", "D", {"match_score": 1, "reasoning": "x"})  # must not raise


def test_set_only_persists_whitelisted_fields():
    cache = ai_agent.ScoreCache()
    cache.set("T", "D", {
        "match_score": 80, "reasoning": "x",
        "suggested_price": "$1", "delivery_days": 1,
        "unexpected_field": "should not be persisted",
    })
    result = cache.get("T", "D")
    assert "unexpected_field" not in result
