"""
Tests for the budget/timeline strict-adherence feature: ProjectScoreSchema's
budget_timeline_adjusted/budget_timeline_note fields, and how they flow
through _finalize_score_result into the returned Evaluation.

This feature does NOT change scoring logic itself — it's Gemini's own job
to decide whether to honor or override the client's stated budget/
timeline (see the prompt instructions in score_project/score_projects_batch).
These tests instead verify the PLUMBING: that whatever Gemini decides
(simulated via a fake structured response) is correctly threaded through
to the final Evaluation object, since that's the part a regression could
actually break silently.
"""

import ai_agent
import config


def test_schema_defaults_to_not_adjusted():
    schema = ai_agent.ProjectScoreSchema(
        match_score=80, reasoning="ok", suggested_price="$100", delivery_days=5,
    )
    assert schema.budget_timeline_adjusted is False
    assert schema.budget_timeline_note == ""


def test_schema_accepts_explicit_adjustment():
    schema = ai_agent.ProjectScoreSchema(
        match_score=80, reasoning="ok", suggested_price="$500", delivery_days=14,
        budget_timeline_adjusted=True,
        budget_timeline_note="الميزانية المعلنة غير كافية لنطاق العمل الموصوف.",
    )
    assert schema.budget_timeline_adjusted is True
    assert "غير كافية" in schema.budget_timeline_note


def test_evaluation_defaults_to_not_adjusted():
    ev = ai_agent.Evaluation(match_score=80.0, reasoning="ok")
    assert ev.budget_timeline_adjusted is False
    assert ev.budget_timeline_note == ""


def test_client_budget_honored_flows_through_finalize(monkeypatch, make_fake_response):
    """The common case: Gemini used the client's own stated figures as-is
    (budget_timeline_adjusted=False) — the Evaluation must reflect that,
    with no adjustment note."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)  # isolate scoring only, no proposal call
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=50, reasoning="ok",
            suggested_price="$300", delivery_days=7,
            budget_timeline_adjusted=False, budget_timeline_note="",
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project("T", "Client offered $300, needs it in 7 days", tags=[])

    assert result.suggested_price == "$300"
    assert result.delivery_days == 7
    assert result.budget_timeline_adjusted is False
    assert result.budget_timeline_note == ""


def test_severely_unrealistic_client_budget_gets_overridden(monkeypatch, make_fake_response):
    """When Gemini judges the client's stated figure severely unrealistic,
    the deviation and its justification must survive into the Evaluation
    so notifier.py can surface it to the human."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=50, reasoning="ok",
            suggested_price="$800", delivery_days=14,
            budget_timeline_adjusted=True,
            budget_timeline_note="الميزانية المعلنة ($20) غير كافية إطلاقاً لنطاق العمل الموصوف.",
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project("T", "Client offers only $20 for a full e-commerce platform", tags=[])

    assert result.suggested_price == "$800"
    assert result.budget_timeline_adjusted is True
    assert "غير كافية" in result.budget_timeline_note


def test_cache_hit_preserves_budget_timeline_fields(monkeypatch, make_fake_response):
    """A cached score result (a plain dict, not the Pydantic schema) must
    still carry budget_timeline_adjusted/note through correctly — cache
    entries replay as raw dicts, so this exercises the dict.get() path in
    _finalize_score_result rather than the schema's own defaults."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=30, reasoning="ok",
            suggested_price="$50", delivery_days=2,
            budget_timeline_adjusted=True, budget_timeline_note="تم التعديل.",
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    ai_agent.evaluate_project("T", "D", tags=[])  # cache miss, populates cache

    def fail_if_called(*a, **kw):
        raise AssertionError("should be a cache hit, Gemini must not be called again")

    monkeypatch.setattr(ai_agent, "_generate", fail_if_called)
    result = ai_agent.evaluate_project("T", "D", tags=[])  # cache hit

    assert result.budget_timeline_adjusted is True
    assert result.budget_timeline_note == "تم التعديل."


def test_missing_budget_timeline_fields_default_safely():
    """A malformed/old-format cached dict lacking these keys entirely must
    not raise, and must default to 'not adjusted'."""
    score_stats = dict(ai_agent._EMPTY_CALL_STATS)
    score_data = {"match_score": 50, "reasoning": "ok", "suggested_price": "$1", "delivery_days": 1}

    result = ai_agent._finalize_score_result("T", "D", None, score_data, score_stats, 10, 5)

    assert result.budget_timeline_adjusted is False
    assert result.budget_timeline_note == ""


def test_scoring_prompt_states_the_adherence_rule(monkeypatch, make_fake_response):
    """The actual prompt text sent to Gemini must contain the strict
    budget/timeline adherence instruction — a regression here (e.g.
    someone editing the prompt and dropping this) would silently degrade
    the feature to nothing, without any other test catching it."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1,
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    ai_agent.score_project("T", "D")

    assert "BUDGET AND TIMELINE" in seen["prompt"]
    assert "SEVERELY unrealistic" in seen["prompt"]
