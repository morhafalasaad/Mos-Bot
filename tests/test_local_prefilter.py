"""
Tests for ai_agent.local_skill_prefilter — the zero-Gemini-cost local
check run before any project reaches the API. Covers both the original
tag-based check and the title/description fallback used when no tags are
available.
"""

import queue
from types import SimpleNamespace

import pytest

import ai_agent
import config
import main
import scraper


def test_tag_overlap_passes(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    assert ai_agent.local_skill_prefilter(["python-dev"], "x", "y") is True


def test_tag_no_overlap_is_authoritative_no(monkeypatch):
    """A non-empty tag list with zero overlap is a definitive 'no' — it
    must NOT fall through to checking title/description text, even if
    that text would otherwise match."""
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    result = ai_agent.local_skill_prefilter(
        ["graphic-design"], "A python scraper project", "build a python web scraper",
    )
    assert result is False


def test_no_tags_but_title_matches_passes(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)
    result = ai_agent.local_skill_prefilter([], "Need a Python web scraper built", "details here")
    assert result is True


def test_no_tags_and_no_text_overlap_fails(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)
    result = ai_agent.local_skill_prefilter([], "Need a logo designed", "creative branding work")
    assert result is False


def test_no_tags_and_no_text_at_all_fails_open(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)
    assert ai_agent.local_skill_prefilter([], None, None) is True


def test_title_prefilter_disabled_always_passes_when_no_tags(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["Python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)
    result = ai_agent.local_skill_prefilter([], "Need a logo designed", "creative branding")
    assert result is True


def test_evaluate_project_filters_irrelevant_untagged_project_at_zero_cost(monkeypatch):
    """End-to-end: an untagged, genuinely irrelevant project should never
    trigger a Gemini call at all."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)

    def fake_generate_should_not_be_called(*a, **kw):
        raise AssertionError("Gemini should never be called for a locally-filtered project")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_should_not_be_called)

    result = ai_agent.evaluate_project(
        "Design a wedding invitation", "Need creative graphic design for a wedding card", tags=[],
    )
    assert result.match_score == 0.0
    assert result.ai_failed is False


def test_evaluate_project_still_scores_relevant_untagged_project(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "MATCH_THRESHOLD", 60)
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)

    calls = []

    def fake_generate(prompt, response_schema=None, **kw):
        calls.append(response_schema)
        if response_schema is ai_agent.ProjectScoreSchema:
            return make_fake_response(parsed=ai_agent.ProjectScoreSchema(
                match_score=80, reasoning="ok", suggested_price="$1", delivery_days=1,
            ))
        return make_fake_response(text="proposal")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    result = ai_agent.evaluate_project(
        "Need a Python developer", "Build a python automation script", tags=[],
    )
    assert len(calls) >= 1
    assert result.match_score == 80.0


# ---------------------------------------------------------------------------
# Producer loop — must apply the FULL local pre-filter (tags AND the
# title/description fallback) BEFORE enqueuing, so irrelevant untagged
# projects never consume consumer batch slots.
# ---------------------------------------------------------------------------

class _StopProducer(Exception):
    """Raised from the stubbed safe_sleep to break out of producer_loop's
    otherwise-infinite `while True` after exactly one scrape iteration."""


def _scraped(pid, title, description, tags=None):
    return SimpleNamespace(
        id=pid, title=title, description=description, tags=tags or [],
        url=f"https://mostaql.com/project/{pid}-x", budget=None, duration=None,
        client_warning=None, client_info=None, screening_questions=[],
    )


def _run_one_producer_iteration(monkeypatch, projects):
    q = queue.Queue()
    monkeypatch.setattr(main, "task_queue", q)
    monkeypatch.setattr(scraper, "get_new_projects", lambda: list(projects))

    def stop(*a, **kw):
        raise _StopProducer()

    monkeypatch.setattr(main, "safe_sleep", stop)
    with pytest.raises(_StopProducer):
        main.producer_loop()
    return [q.get_nowait() for _ in range(q.qsize())]


def test_producer_skips_untagged_irrelevant_project_when_title_prefilter_enabled(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)

    irrelevant = _scraped("1", "Design a wedding invitation", "Need creative graphic design for a wedding card")
    relevant = _scraped("2", "Need a Python developer", "Build a python automation script")

    enqueued = _run_one_producer_iteration(monkeypatch, [irrelevant, relevant])

    assert [p.id for p in enqueued] == ["2"]


def test_producer_enqueues_untagged_project_when_title_prefilter_disabled(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", False)

    irrelevant = _scraped("1", "Design a wedding invitation", "Need creative graphic design for a wedding card")

    enqueued = _run_one_producer_iteration(monkeypatch, [irrelevant])

    assert [p.id for p in enqueued] == ["1"]  # prefilter is a no-op for untagged projects when disabled


def test_producer_still_skips_tagged_project_with_no_skill_overlap(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)

    tagged_irrelevant = _scraped("1", "Python scraper", "python", tags=["graphic-design"])
    tagged_relevant = _scraped("2", "Logo", "logo", tags=["python-dev"])

    enqueued = _run_one_producer_iteration(monkeypatch, [tagged_irrelevant, tagged_relevant])

    # Tags are authoritative: the first is skipped despite a matching title,
    # the second passes despite an unrelated title.
    assert [p.id for p in enqueued] == ["2"]


def test_producer_fails_open_for_untagged_project_with_no_text(monkeypatch):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "TITLE_PREFILTER_ENABLED", True)

    blank = _scraped("1", "", "")

    enqueued = _run_one_producer_iteration(monkeypatch, [blank])

    assert [p.id for p in enqueued] == ["1"]
