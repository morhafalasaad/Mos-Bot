"""
Tests for ai_agent.draft_proposal's client_info-driven tone adaptation —
must adjust TONE ONLY, and must never leak a mention of the client's
rating/history into the actual proposal text.
"""

import ai_agent
import config


def _capture_prompt(monkeypatch, make_fake_response):
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(text="proposal")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    return seen


def test_strong_established_client_gets_confidence_hint(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "STRONG_CLIENT_RATING_THRESHOLD", 4.5)
    seen = _capture_prompt(monkeypatch, make_fake_response)

    ai_agent.draft_proposal("T", "D", client_info={"rating": 4.9, "reviews_count": 10, "is_new": False})

    assert "عميل موثوق" in seen["prompt"]


def test_new_unrated_client_gets_warm_hint(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = _capture_prompt(monkeypatch, make_fake_response)

    ai_agent.draft_proposal("T", "D", client_info={"rating": None, "reviews_count": 0, "is_new": True})

    assert "عميل جديد" in seen["prompt"]


def test_no_client_info_has_neither_hint(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = _capture_prompt(monkeypatch, make_fake_response)

    ai_agent.draft_proposal("T", "D", client_info=None)

    assert "عميل موثوق" not in seen["prompt"]
    assert "عميل جديد" not in seen["prompt"]


def test_no_mention_rule_present_regardless_of_client_info(monkeypatch, make_fake_response):
    """The explicit instruction forbidding any mention of the client's
    rating/history in the actual proposal text must be present in every
    case — strong client, new client, and no info at all."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    seen = _capture_prompt(monkeypatch, make_fake_response)

    for client_info in (
        {"rating": 4.9, "reviews_count": 10, "is_new": False},
        {"rating": None, "reviews_count": 0, "is_new": True},
        None,
    ):
        ai_agent.draft_proposal("T", "D", client_info=client_info)
        assert "ممنوع الإشارة" in seen["prompt"]


def test_strong_client_requires_minimum_review_count(monkeypatch, make_fake_response):
    """A high rating with too few reviews shouldn't trigger the
    'established client' framing — could be a fluke rating."""
    monkeypatch.setattr(config, "MY_SKILLS", ["python"])
    monkeypatch.setattr(config, "STRONG_CLIENT_RATING_THRESHOLD", 4.5)
    seen = _capture_prompt(monkeypatch, make_fake_response)

    ai_agent.draft_proposal("T", "D", client_info={"rating": 5.0, "reviews_count": 1, "is_new": False})

    assert "عميل موثوق" not in seen["prompt"]
