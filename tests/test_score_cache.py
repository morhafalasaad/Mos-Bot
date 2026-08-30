"""
Tests for ai_agent.ScoreCache — caches a project's score result keyed by
(title, full description, current MY_SKILLS), so re-evaluating identical
content skips a fresh Gemini call.
"""

import ai_agent
import config


def test_cache_miss_on_first_lookup(tmp_path):
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    assert cache.get("Some title", "Some description") is None


def test_set_then_get_returns_stored_fields(tmp_path):
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    cache.set("T", "D", {
        "match_score": 77, "reasoning": "good fit",
        "matched_skills": ["python"], "missing_skills": [],
        "suggested_price": "$100", "delivery_days": 3,
    })
    result = cache.get("T", "D")
    assert result == {
        "match_score": 77, "reasoning": "good fit",
        "matched_skills": ["python"], "missing_skills": [],
        "suggested_price": "$100", "delivery_days": 3,
    }


def test_different_description_is_a_cache_miss(tmp_path):
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    cache.set("T", "Description A", {"match_score": 50, "reasoning": "x"})
    assert cache.get("T", "Description B") is None


def test_my_skills_change_invalidates_cache(tmp_path, monkeypatch):
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    cache.set("T", "D", {"match_score": 90, "reasoning": "x"})
    assert cache.get("T", "D") is not None

    monkeypatch.setattr(config, "MY_SKILLS", ["php"])
    assert cache.get("T", "D") is None, "changing MY_SKILLS must invalidate previously-cached scores"


def test_disabled_cache_always_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SCORE_CACHE_ENABLED", False)
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    cache.set("T", "D", {"match_score": 90, "reasoning": "x"})
    assert cache.get("T", "D") is None


def test_max_entries_eviction(tmp_path):
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"), max_entries=3)
    for i in range(5):
        cache.set(f"Title {i}", f"Description {i}", {"match_score": i, "reasoning": "x"})

    hits = sum(1 for i in range(5) if cache.get(f"Title {i}", f"Description {i}") is not None)
    assert hits <= 3, "cache should have evicted down to max_entries"


def test_corrupt_cache_file_degrades_to_miss_not_crash(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("this is not valid json {{{")
    cache = ai_agent.ScoreCache(path=str(cache_path))
    assert cache.get("T", "D") is None  # must not raise


def test_set_only_persists_whitelisted_fields(tmp_path):
    """Guards against accidentally caching something unexpected (e.g. a
    stray proposal_ar) that snuck into a score_data dict."""
    cache = ai_agent.ScoreCache(path=str(tmp_path / "cache.json"))
    cache.set("T", "D", {
        "match_score": 80, "reasoning": "x",
        "suggested_price": "$1", "delivery_days": 1,
        "unexpected_field": "should not be persisted",
    })
    result = cache.get("T", "D")
    assert "unexpected_field" not in result
