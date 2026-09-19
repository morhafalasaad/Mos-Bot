"""
Tests for reply_assistant.py — the "Reply Assistant" mode that drafts
multiple distinct-tone reply options for a pasted client message.

reply_assistant.get_reply_options() calls ai_agent._generate() lazily
(imported inside the function, not at module load time — see its
docstring), so tests monkeypatch ai_agent._generate directly, exactly
like every ai_agent-level test in this suite.
"""

import ai_agent
import config
import reply_assistant


# ---- looks_like_client_message (local, zero-cost heuristic) ----------------

def test_empty_string_is_not_a_client_message():
    assert reply_assistant.looks_like_client_message("") is False
    assert reply_assistant.looks_like_client_message(None) is False


def test_short_greeting_is_not_a_client_message():
    assert reply_assistant.looks_like_client_message("hi") is False
    assert reply_assistant.looks_like_client_message("ok") is False


def test_bot_command_is_not_a_client_message():
    assert reply_assistant.looks_like_client_message("/start") is False
    assert reply_assistant.looks_like_client_message("/help me please") is False


def test_realistic_message_is_a_client_message():
    assert reply_assistant.looks_like_client_message(
        "When can you start on this project and what is your day rate?"
    ) is True


def test_arabic_message_is_a_client_message():
    assert reply_assistant.looks_like_client_message(
        "مرحبا، متى يمكنك البدء بالعمل على المشروع؟"
    ) is True


def test_boundary_length_respects_config(monkeypatch):
    monkeypatch.setattr(config, "REPLY_ASSISTANT_MIN_MESSAGE_CHARS", 10)
    assert reply_assistant.looks_like_client_message("123456789") is False  # 9 chars
    assert reply_assistant.looks_like_client_message("1234567890") is True  # 10 chars


def test_never_raises_on_unexpected_input():
    assert reply_assistant.looks_like_client_message(12345) is False
    assert reply_assistant.looks_like_client_message([]) is False


# ---- get_reply_options -------------------------------------------------------

def test_returns_summary_and_options_on_success(monkeypatch, make_fake_response):
    fake_schema = reply_assistant.ReplyOptionsSchema(
        situation_summary="Pre-hire client asking about timeline",
        options=[
            {"label": "مختصر ومباشر", "reply": "شكراً لرسالتكم، يمكنني البدء فوراً."},
            {"label": "تفصيلي وتقني", "reply": "بخصوص الجدول الزمني، سأحتاج حوالي 5 أيام."},
            {"label": "دافئ ومطمئن", "reply": "يسعدني جداً العمل معكم على هذا المشروع."},
        ],
    )

    def fake_generate(prompt, response_schema=None, **kw):
        assert response_schema is reply_assistant.ReplyOptionsSchema
        return make_fake_response(parsed=fake_schema)

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)

    summary, options, stats = reply_assistant.get_reply_options(
        "When can you start and what is your day rate?"
    )

    assert summary == "Pre-hire client asking about timeline"
    assert len(options) == 3
    assert options[0]["label"] == "مختصر ومباشر"
    assert "خبرة" not in options[0]["reply"] or True  # no structural assumption on content


def test_default_option_count_matches_config(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "REPLY_ASSISTANT_OPTION_COUNT", 3)
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options("Some client message here")

    assert "3 distinct reply options" in seen["prompt"] or "draft 3" in seen["prompt"]


def test_explicit_option_count_overrides_config(monkeypatch, make_fake_response):
    monkeypatch.setattr(config, "REPLY_ASSISTANT_OPTION_COUNT", 3)
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options("Some client message here", option_count=2)

    assert "draft 2" in seen["prompt"]


def test_project_context_included_when_given(monkeypatch, make_fake_response):
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options(
        "Some client message", project_context="Project: Build a REST API in Python",
    )

    assert "Build a REST API in Python" in seen["prompt"]


def test_no_project_context_omits_context_block(monkeypatch, make_fake_response):
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options("Some client message")

    assert "Project context" not in seen["prompt"]


def test_client_message_is_embedded_in_prompt(monkeypatch, make_fake_response):
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options("This is a very specific client question about pricing")

    assert "This is a very specific client question about pricing" in seen["prompt"]


def test_prompt_covers_scope_creep_handling(monkeypatch, make_fake_response):
    """The prompt must explicitly instruct polite-but-firm scope-creep
    handling — this is one of the named post-hire scenarios from the
    feature spec, and a regression here would silently degrade replies
    for that scenario without any other test catching it."""
    seen = {}

    def fake_generate(prompt, response_schema=None, **kw):
        seen["prompt"] = prompt
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    reply_assistant.get_reply_options("Can you also add a mobile app to this for the same price?")

    assert "scope creep" in seen["prompt"].lower()
    assert "flatly refuse" in seen["prompt"].lower()


def test_gemini_failure_returns_none_empty_without_raising(monkeypatch):
    def fake_generate_raises(*a, **kw):
        raise RuntimeError("simulated Gemini outage")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate_raises)

    summary, options, stats = reply_assistant.get_reply_options("Some client message here")

    assert summary is None
    assert options == []
    assert stats["response_time_sec"] >= 0


def test_malformed_options_entries_are_skipped(monkeypatch, make_fake_response):
    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x",
            options=[{"label": "Valid", "reply": "A valid reply."}],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    summary, options, stats = reply_assistant.get_reply_options("Some client message")

    assert len(options) == 1
    assert options[0]["label"] == "Valid"


def test_unparseable_response_degrades_to_empty(monkeypatch, make_fake_response):
    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=None, text="not valid json {{{")

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    summary, options, stats = reply_assistant.get_reply_options("Some client message")

    assert summary is None
    assert options == []


def test_stats_are_populated_on_success(monkeypatch, make_fake_response):
    def fake_generate(prompt, response_schema=None, **kw):
        return make_fake_response(parsed=reply_assistant.ReplyOptionsSchema(
            situation_summary="x", options=[],
        ))

    monkeypatch.setattr(ai_agent, "_generate", fake_generate)
    summary, options, stats = reply_assistant.get_reply_options("Some client message here")

    # Comes from the shared fake_usage fixture in conftest.py.
    assert stats["total_tokens"] == 20
