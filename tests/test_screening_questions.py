"""
Tests for the screening-questions feature:
  - ai_agent.draft_screening_answers() — drafts one answer per question
  - Its wiring into _finalize_score_result()/evaluate_project()/
    evaluate_projects_batch() — only called above threshold, only when
    questions exist, never blocks the rest of the evaluation on failure.

Scraper-side extraction (scraper.parse_screening_questions) has its own
dedicated test file — test_scraper_screening_questions.py.
"""

import ai_agent
import config


def test_no_questions_returns_empty_without_calling_gemini(monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("must not call Gemini when there are no questions")

    monkeypatch.setattr(ai_agent, "_generate", fail_if_called)

    answers, stats = ai_agent.draft_screening_answers("T", "D", [])
    assert answers == []
    assert stats == dict(ai_agent._EMPTY_CALL_STATS)


def test_answers_drafted_in_order(monkeypatch, make_fake_response):
    questions = ["How will you implement this?", "What is your relevant experience?"]

    def fake_generate(prompt, response_schema=None, **kw):
        assert response_schema is ai_agent.ScreeningAnswersSchema
        return make_fake_response(parsed=ai_agent.ScreeningAnswersSchema(answers=[
            {"question": questions[0], "answer": "I will use FastAPI and PostgreSQL."},
            {"question": questions[1], "answer": "I have 5 years of relevant experience."},
        ]))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    answers, stats = ai_agent.draft_screening_answers("T", "D", questions)

    assert len(answers) == 2
    assert answers[0]["question"] == questions[0]
    assert "FastAPI" in answers[0]["answer"]
    assert answers[1]["question"] == questions[1]


def test_malformed_answer_entries_are_skipped(monkeypatch, make_fake_response):
    """A junk/incomplete entry in Gemini's response (missing question or
    answer) must be dropped, not crash or produce a blank entry."""

    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=ai_agent.ScreeningAnswersSchema(answers=[
            {"question": "Real question?", "answer": "Real answer."},
        ]))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    answers, stats = ai_agent.draft_screening_answers("T", "D", ["Real question?"])
    assert len(answers) == 1


def test_gemini_failure_degrades_to_empty_list_without_raising(monkeypatch):
    def fake_generate_raises(*a, **kw):
        raise RuntimeError("simulated Gemini outage")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_raises)

    answers, stats = ai_agent.draft_screening_answers("T", "D", ["Some question?"])
    assert answers == []
    assert stats["response_time_sec"] >= 0


def test_unparseable_response_degrades_to_empty_list(monkeypatch, make_fake_response):
    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=None, text="not valid json at all {{{")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    answers, stats = ai_agent.draft_screening_answers("T", "D", ["Some question?"])
    assert answers == []


# ---- Wiring through _finalize_score_result / evaluate_project --------------

def test_above_threshold_with_questions_drafts_answers(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(response_schema)
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="ok", suggested_price="$1", delivery_days=1,
            ))
        if response_schema is ai_agent.ScreeningAnswersSchema:
            return make_fake_response(parsed=ai_agent.ScreeningAnswersSchema(answers=[
                {"question": "How will you implement this?", "answer": "Using FastAPI."},
            ]))
        return make_fake_response(text="proposal text")  # draft_proposal uses no response_schema

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project(
        "T", "D", tags=[], screening_questions=["How will you implement this?"],
    )

    assert ai_agent.ScreeningAnswersSchema in calls
    assert len(result.screening_answers) == 1
    assert result.screening_answers[0]["answer"] == "Using FastAPI."
    assert result.proposal_ar == "proposal text"


def test_above_threshold_without_questions_never_calls_screening(monkeypatch, make_fake_response):
    """The overwhelmingly common case — most projects have no screening
    questions — must add zero extra Gemini cost."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(response_schema)
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="ok", suggested_price="$1", delivery_days=1,
            ))
        assert response_schema is not ai_agent.ScreeningAnswersSchema, \
            "must not call draft_screening_answers when there are no questions"
        return make_fake_response(text="proposal text")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project("T", "D", tags=[])  # no screening_questions passed

    assert ai_agent.ScreeningAnswersSchema not in calls
    assert result.screening_answers == []


def test_below_threshold_never_drafts_screening_answers(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 90)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(response_schema)
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=30, reasoning="poor fit", suggested_price="$1", delivery_days=1,
            ))
        raise AssertionError("only the scoring call should happen below threshold")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project(
        "T", "D", tags=[], screening_questions=["Some question?"],
    )

    assert calls == [ai_agent.ProjectScoreSchema]
    assert result.screening_answers == []
    assert result.proposal_ar is None


def test_screening_answer_failure_does_not_block_proposal(monkeypatch, make_fake_response):
    """If draft_screening_answers itself fails, the project's score and
    proposal must still be returned normally — screening answers degrade
    to an empty list, nothing else is affected."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="ok", suggested_price="$1", delivery_days=1,
            ))
        if response_schema is ai_agent.ScreeningAnswersSchema:
            raise RuntimeError("simulated screening-answers failure")
        return make_fake_response(text="proposal text")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project(
        "T", "D", tags=[], screening_questions=["Some question?"],
    )

    assert result.screening_answers == []
    assert result.proposal_ar == "proposal text"  # proposal drafting is unaffected
    assert result.match_score == 80.0


def test_batch_evaluation_threads_screening_questions_per_project(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    fake_batch_result = ai_agent.BatchScoreSchema(results=[
        ai_agent._BatchScoreItem(index=0, match_score=90, reasoning="ok", suggested_price="$1", delivery_days=1),
        ai_agent._BatchScoreItem(index=1, match_score=90, reasoning="ok", suggested_price="$1", delivery_days=1),
    ])

    def fake_generate(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.BatchScoreSchema:
            return make_fake_response(parsed=fake_batch_result)
        if response_schema is ai_agent.ScreeningAnswersSchema:
            return make_fake_response(parsed=ai_agent.ScreeningAnswersSchema(answers=[
                {"question": "Q for project A?", "answer": "A's answer."},
            ]))
        return make_fake_response(text="proposal")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    projects = [
        {"title": "A", "description": "d1", "budget": None, "tags": [], "screening_questions": ["Q for project A?"]},
        {"title": "B", "description": "d2", "budget": None, "tags": [], "screening_questions": []},
    ]
    results = ai_agent.evaluate_projects_batch(projects)

    assert len(results[0].screening_answers) == 1
    assert results[0].screening_answers[0]["answer"] == "A's answer."
    assert results[1].screening_answers == []  # project B had no questions


def test_cache_hit_still_drafts_fresh_screening_answers(monkeypatch, make_fake_response):
    """Screening answers, like the proposal, must always be drafted FRESH
    even when the underlying score came from the cache — a cached score
    doesn't mean the screening questions were already answered."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    def fake_generate_prewarm(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="ok", suggested_price="$1", delivery_days=1,
            ))
        return make_fake_response(text="proposal")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_prewarm)
    # First call with NO screening questions -> populates the score cache.
    ai_agent.evaluate_project("T", "D", tags=[])

    screening_calls = []

    def fake_generate_second(prompt, response_schema=None, **kw):
        if response_schema is ai_agent.ScreeningAnswersSchema:
            screening_calls.append(1)
            return make_fake_response(parsed=ai_agent.ScreeningAnswersSchema(answers=[
                {"question": "New question this time?", "answer": "Fresh answer."},
            ]))
        return make_fake_response(text="proposal")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_second)
    # Second call, SAME title/description (cache hit on score) but WITH
    # screening questions this time.
    result = ai_agent.evaluate_project(
        "T", "D", tags=[], screening_questions=["New question this time?"],
    )

    assert len(screening_calls) == 1
    assert result.screening_answers[0]["answer"] == "Fresh answer."
