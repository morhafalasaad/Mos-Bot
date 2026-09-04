"""
Tests for the matched_skills/missing_skills breakdown produced by
_finalize_score_result — including defensive parsing of malformed Gemini
output (wrong types, junk list entries).
"""

import ai_agent
import config


def test_matched_and_missing_skills_flow_through(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python", "Web Scraping"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)

    def fake_generate(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="Good fit",
                matched_skills=["Python", "Web Scraping"],
                missing_skills=["Selenium"],
                suggested_price="$200", delivery_days=3,
            ))
        return make_fake_response(text="proposal text")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project("Build a scraper", "Need a python web scraper using selenium", tags=[])

    assert result.matched_skills == ["Python", "Web Scraping"]
    assert result.missing_skills == ["Selenium"]


def test_cache_hit_preserves_matched_and_missing_skills(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)  # avoid a proposal call, isolate scoring/caching
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)  # isolate caching, not the pre-filter

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=20, reasoning="ok",
            matched_skills=["Python"], missing_skills=["Docker"],
            suggested_price="$1", delivery_days=1,
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    ai_agent.evaluate_project("T", "D", tags=[])  # cache miss, populates cache

    def fail_if_called(*a, **kw):
        raise AssertionError("should be a cache hit, Gemini must not be called again")

    monkeypatch.setattr(ai_agent, "_generate", fail_if_called)
    result = ai_agent.evaluate_project("T", "D", tags=[])  # cache hit

    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["Docker"]


def test_wrong_type_matched_skills_defaults_to_empty_list():
    """Gemini's structured output enforces the schema, but a cache hit
    replays a plain dict — defend against unexpected shapes there too."""
    score_stats = dict(ai_agent._EMPTY_CALL_STATS)
    score_data = {
        "match_score": 50, "reasoning": "ok",
        "matched_skills": "Python",  # wrong type: string, not list
        "missing_skills": ["Selenium", "", None, "  ", "Docker"],  # junk mixed in
        "suggested_price": "$1", "delivery_days": 1,
    }
    result = ai_agent._finalize_score_result("T", "D", None, score_data, score_stats, 10, 5)

    assert result.matched_skills == []
    assert result.missing_skills == ["Selenium", "Docker"]


def test_missing_fields_default_to_empty_lists():
    score_stats = dict(ai_agent._EMPTY_CALL_STATS)
    score_data = {"match_score": 50, "reasoning": "ok", "suggested_price": "$1", "delivery_days": 1}
    result = ai_agent._finalize_score_result("T", "D", None, score_data, score_stats, 10, 5)

    assert result.matched_skills == []
    assert result.missing_skills == []
