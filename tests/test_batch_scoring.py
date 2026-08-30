"""
Tests for ai_agent.score_projects_batch / evaluate_projects_batch — scores
multiple projects in ONE Gemini call instead of one call per project.
"""

import ai_agent
import config


def test_batch_scoring_reduces_call_count(monkeypatch, make_fake_response):
    """3 projects (2 matches, 1 non-match) should take 3 total Gemini
    calls: 1 batch score + 2 proposals — not 5 (3 scores + 2 proposals)."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python", "web scraping"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    # Disabled so this test isolates BATCH SCORING mechanics — otherwise
    # project B (no skill-overlapping words in its own title/description)
    # would get filtered locally before ever reaching the batch, shifting
    # the index mapping. The local pre-filter has its own dedicated tests.
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    fake_batch_result = ai_agent.BatchScoreSchema(results=[
        ai_agent._BatchScoreItem(index=0, match_score=85, reasoning="ok", suggested_price="$1", delivery_days=1),
        ai_agent._BatchScoreItem(index=1, match_score=20, reasoning="ok", suggested_price="$1", delivery_days=1),
        ai_agent._BatchScoreItem(index=2, match_score=70, reasoning="ok", suggested_price="$1", delivery_days=1),
    ])
    calls = []

    def fake_generate(prompt, response_schema=None, temperature=None, max_output_tokens=None):
        if response_schema is ai_agent.BatchScoreSchema:
            calls.append("batch_score")
            return make_fake_response(parsed=fake_batch_result)
        calls.append("draft_proposal")
        return make_fake_response(text="proposal text")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    projects = [
        {"title": "A", "description": "python scraper", "budget": None, "tags": []},
        {"title": "B", "description": "logo design", "budget": None, "tags": []},
        {"title": "C", "description": "python automation", "budget": None, "tags": []},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert calls == ["batch_score", "draft_proposal", "draft_proposal"]
    assert [r.match_score for r in results] == [85.0, 20.0, 70.0]
    assert results[0].proposal_ar and results[2].proposal_ar
    assert not results[1].proposal_ar


def test_batch_missing_index_marks_only_that_project_failed(monkeypatch, make_fake_response):
    """If Gemini's response drops an entry for one project in the batch,
    only that project should be ai_failed — the rest of the batch must be
    unaffected."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)  # nothing clears threshold; isolates the scoring behavior
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    # Only returns results for index 0 — index 1 is silently dropped.
    fake_batch_result = ai_agent.BatchScoreSchema(results=[
        ai_agent._BatchScoreItem(index=0, match_score=90, reasoning="ok", suggested_price="$1", delivery_days=1),
    ])

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=fake_batch_result)

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    projects = [
        {"title": "A", "description": "x", "budget": None, "tags": []},
        {"title": "B", "description": "y", "budget": None, "tags": []},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert results[0].ai_failed is False
    assert results[0].match_score == 90.0
    assert results[1].ai_failed is True


def test_total_batch_failure_marks_every_project_failed(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate_raises(*a, **kw):
        raise RuntimeError("simulated: all Gemini keys exhausted")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_raises)

    projects = [
        {"title": "A", "description": "x", "budget": None, "tags": []},
        {"title": "B", "description": "y", "budget": None, "tags": []},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert all(r.ai_failed for r in results)


def test_locally_prefiltered_project_never_reaches_batch_call(monkeypatch):
    """A project whose tags share zero overlap with MY_SKILLS should never
    be sent to Gemini at all, even inside a batch with other projects."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(prompt)
        raise AssertionError("should never be called for a fully-filtered batch")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    projects = [
        {"title": "Graphic design", "description": "logo work", "budget": None, "tags": ["graphic-design"]},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert calls == []
    assert results[0].match_score == 0.0
    assert results[0].ai_failed is False


def test_cached_project_is_excluded_from_the_batch_call(monkeypatch, make_fake_response):
    """A project whose score is already cached should be resolved from the
    cache and NOT sent to Gemini as part of the batch."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 200)  # isolate scoring behavior only
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    # Pre-warm the cache for project B via the single-project path.
    def fake_generate_prewarm(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=20, reasoning="cached-ok", suggested_price="$1", delivery_days=1,
            ))
        return make_fake_response(text="p")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_prewarm)
    ai_agent.evaluate_project("B", "desc B content", tags=[])

    batch_calls = []

    def fake_generate_batch(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.BatchScoreSchema:
            batch_calls.append(prompt)
            return make_fake_response(parsed=ai_agent.BatchScoreSchema(results=[
                ai_agent._BatchScoreItem(index=0, match_score=90, reasoning="A", suggested_price="$1", delivery_days=1),
                ai_agent._BatchScoreItem(index=1, match_score=15, reasoning="C", suggested_price="$1", delivery_days=1),
            ]))
        return make_fake_response(text="p")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_batch)

    projects = [
        {"title": "A", "description": "desc A content", "budget": None, "tags": []},
        {"title": "B", "description": "desc B content", "budget": None, "tags": []},  # cache hit
        {"title": "C", "description": "desc C content", "budget": None, "tags": []},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert len(batch_calls) == 1  # only ONE batch call, covering A and C
    assert results[1].reasoning == "cached-ok"  # B's result came from the cache, not the batch
