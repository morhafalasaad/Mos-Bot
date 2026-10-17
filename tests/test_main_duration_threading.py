"""
Tests that main.py threads each project's scraped `duration` (alongside
`budget`) into ai_agent's evaluation entry points, so Gemini sees the
detail-page figures during scoring.
"""

from types import SimpleNamespace

import ai_agent
import github_fallback
import main


def _project(**overrides):
    base = dict(
        id="42", title="Python scraper", description="Build a python scraper",
        url="https://mostaql.com/project/42-x", budget="$100", duration="5 أيام",
        tags=[], client_warning=None, client_info=None, screening_questions=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_process_project_batch_passes_duration_to_evaluate_projects_batch(monkeypatch):
    captured = {}

    def fake_batch(batch_input):
        captured["input"] = batch_input
        return []

    monkeypatch.setattr(ai_agent, "evaluate_projects_batch", fake_batch)

    main.process_project_batch([_project(), _project(id="43", duration=None)])

    assert captured["input"][0]["duration"] == "5 أيام"
    assert captured["input"][0]["budget"] == "$100"
    assert captured["input"][1]["duration"] is None


def test_retry_pending_queue_passes_duration_to_evaluate_project(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.db, "get_pending_projects", lambda: [{
        "id": "7", "title": "T", "description": "D", "budget": "$10", "duration": "3 أيام",
        "tags": [], "retry_count": 0,
    }])

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here — only the call arguments matter")

    monkeypatch.setattr(ai_agent, "evaluate_project", fake_evaluate)
    monkeypatch.setattr(main.db, "update_pending_project", lambda *a, **kw: None)

    main.retry_pending_queue()

    assert captured["duration"] == "3 أيام"
    assert captured["budget"] == "$10"


def test_retry_open_github_issues_passes_duration_to_evaluate_project(monkeypatch):
    captured = {}
    monkeypatch.setattr(github_fallback, "list_open_fallback_issues", lambda: [
        {"number": 9, "title": f"{github_fallback.ISSUE_TITLE_PREFIX}T", "body": "ignored"},
    ])
    monkeypatch.setattr(main.db, "get_pending_projects", lambda: [])
    monkeypatch.setattr(github_fallback, "parse_issue_body", lambda body: {
        "url": "https://x", "budget": "$20", "duration": "10 أيام", "tags": [], "description": "D",
    })

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here — only the call arguments matter")

    monkeypatch.setattr(ai_agent, "evaluate_project", fake_evaluate)

    main.retry_open_github_issues()

    assert captured["duration"] == "10 أيام"
    assert captured["budget"] == "$20"
