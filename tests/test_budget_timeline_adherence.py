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


# ---------------------------------------------------------------------------
# Batched scoring path (the primary production path: consumer_loop ->
# process_project_batch -> evaluate_projects_batch) must carry the same
# budget/timeline adherence behavior as the single-project path.
# ---------------------------------------------------------------------------

def test_batch_item_schema_defaults_to_not_adjusted():
    item = ai_agent._BatchScoreItem(
        index=0, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1,
    )
    assert item.budget_timeline_adjusted is False
    assert item.budget_timeline_note == ""


def test_batch_item_schema_accepts_adjustment_fields():
    item = ai_agent._BatchScoreItem(
        index=0, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1,
        budget_timeline_adjusted=True,
        budget_timeline_note="الميزانية المعلنة غير كافية لنطاق العمل الموصوف.",
    )
    assert item.budget_timeline_adjusted is True
    assert "غير كافية" in item.budget_timeline_note


def test_batch_scoring_prompt_states_the_adherence_rule(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=ai_agent.BatchScoreSchema(results=[
            ai_agent._BatchScoreItem(index=0, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1),
        ]))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    ai_agent.score_projects_batch([{"title": "T", "description": "D"}])

    assert "BUDGET AND TIMELINE" in seen["prompt"]
    assert "SEVERELY unrealistic" in seen["prompt"]
    assert "budget_timeline_adjusted" in seen["prompt"]
    assert "budget_timeline_note" in seen["prompt"]


def test_adjustment_fields_flow_through_evaluate_projects_batch(monkeypatch, make_fake_response):
    """budget_timeline_adjusted/note returned by Gemini for ONE project in
    a batch must land on that project's Evaluation — and only that one."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    batch_result = ai_agent.BatchScoreSchema(results=[
        ai_agent._BatchScoreItem(
            index=0, match_score=90, reasoning="ok", suggested_price="$300", delivery_days=14,
            budget_timeline_adjusted=True,
            budget_timeline_note="الميزانية المعلنة ($20) غير كافية إطلاقاً لنطاق العمل الموصوف.",
        ),
        ai_agent._BatchScoreItem(
            index=1, match_score=88, reasoning="ok", suggested_price="$100", delivery_days=5,
        ),
    ])

    def fake_generate(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.BatchScoreSchema:
            return make_fake_response(parsed=batch_result)
        return make_fake_response(text="proposal text")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    results = ai_agent.evaluate_projects_batch([
        {"title": "A", "description": "python scraper", "budget": "$20", "tags": []},
        {"title": "B", "description": "python bot", "budget": "$100", "tags": []},
    ])

    assert results[0].budget_timeline_adjusted is True
    assert "غير كافية" in results[0].budget_timeline_note
    assert results[1].budget_timeline_adjusted is False
    assert results[1].budget_timeline_note == ""


def test_adjustment_fields_survive_a_cache_hit_in_a_later_batch(monkeypatch, make_fake_response):
    """The adjustment fields written to the ScoreCache by a batch result
    must be replayed on a later cache-hit evaluation."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)  # no proposal drafting needed
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    batch_result = ai_agent.BatchScoreSchema(results=[
        ai_agent._BatchScoreItem(
            index=0, match_score=90, reasoning="ok", suggested_price="$300", delivery_days=14,
            budget_timeline_adjusted=True, budget_timeline_note="تم التعديل.",
        ),
    ])
    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(prompt)
        return make_fake_response(parsed=batch_result)

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    project = {"title": "A", "description": "python scraper", "budget": "$20", "tags": []}

    first = ai_agent.evaluate_projects_batch([dict(project)])[0]
    second = ai_agent.evaluate_projects_batch([dict(project)])[0]

    assert len(calls) == 1  # second call was a cache hit
    assert first.budget_timeline_adjusted is True
    assert second.budget_timeline_adjusted is True
    assert second.budget_timeline_note == "تم التعديل."


# ---------------------------------------------------------------------------
# Scraped budget/duration metadata must reach the scoring prompt.
# ---------------------------------------------------------------------------

def _single_fake_generate(seen, make_fake_response):
    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
            match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1,
        ))
    return fake_generate


def test_score_project_includes_scraped_budget_and_duration(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = {}
    monkeypatch.setattr(ai_agent, "_generate", _single_fake_generate(seen, make_fake_response))

    ai_agent.score_project("T", "D", budget="$250 - $500", duration="7 أيام")

    assert "$250 - $500" in seen["prompt"]
    assert "7 أيام" in seen["prompt"]
    assert "Client-stated budget" in seen["prompt"]
    assert "Client-stated delivery time" in seen["prompt"]


def test_score_project_without_metadata_has_no_client_stated_lines(monkeypatch, make_fake_response):
    """Backward compat: (title, description)-only callers get the same
    prompt as before — no empty 'Client-stated ...' placeholder lines.
    (The adherence RULE text itself mentions those labels, so check for the
    label followed by a colon, which only the rendered lines have.)"""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = {}
    monkeypatch.setattr(ai_agent, "_generate", _single_fake_generate(seen, make_fake_response))

    ai_agent.score_project("T", "D")

    assert "Client-stated budget:" not in seen["prompt"]
    assert "Client-stated delivery time:" not in seen["prompt"]


def test_evaluate_project_forwards_budget_and_duration_to_scoring(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)
    seen = {}
    monkeypatch.setattr(ai_agent, "_generate", _single_fake_generate(seen, make_fake_response))

    ai_agent.evaluate_project("T", "python things", budget="$75", duration="3 أيام", tags=[])

    assert "Client-stated budget: $75" in seen["prompt"]
    assert "Client-stated delivery time: 3 أيام" in seen["prompt"]


def test_evaluate_projects_batch_includes_scraped_budget_and_duration_in_prompt(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=ai_agent.BatchScoreSchema(results=[
            ai_agent._BatchScoreItem(index=0, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1),
            ai_agent._BatchScoreItem(index=1, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1),
        ]))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    ai_agent.evaluate_projects_batch([
        {"title": "A", "description": "python one", "budget": "$250 - $500", "duration": "7 أيام", "tags": []},
        {"title": "B", "description": "python two", "budget": None, "tags": []},
    ])

    prompt = seen["prompt"]
    assert "$250 - $500" in prompt
    assert "7 أيام" in prompt
    # Only project A carries metadata, so exactly one rendered budget line.
    assert prompt.count("Client-stated budget:") == 1
    assert prompt.count("Client-stated delivery time:") == 1


def test_batch_prompt_without_metadata_has_no_client_stated_lines(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=ai_agent.BatchScoreSchema(results=[
            ai_agent._BatchScoreItem(index=0, match_score=50, reasoning="ok", suggested_price="$1", delivery_days=1),
        ]))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    ai_agent.score_projects_batch([{"title": "T", "description": "D"}])

    assert "Client-stated budget:" not in seen["prompt"]
    assert "Client-stated delivery time:" not in seen["prompt"]
