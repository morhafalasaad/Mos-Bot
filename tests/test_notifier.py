"""
Tests for notifier.py — Telegram message and inline-keyboard construction.
Pure string/dict assembly, no real network calls.
"""

import notifier


def test_build_message_includes_core_fields():
    msg = notifier.build_message("My Title", "https://mostaql.com/x", 85.0, "Proposal text")
    assert "My Title" in msg
    assert "85%" in msg
    assert "Proposal text" in msg


def test_build_message_skills_lines_appear_when_present():
    msg = notifier.build_message(
        "T", "u", 80.0, "p",
        matched_skills=["Python", "Web Scraping"], missing_skills=["Selenium"],
    )
    assert "Python, Web Scraping" in msg
    assert "Selenium" in msg
    assert "متطابقة" in msg
    assert "غير متوفرة" in msg


def test_build_message_skills_lines_absent_when_not_given():
    msg = notifier.build_message("T", "u", 80.0, "p")
    assert "متطابقة" not in msg
    assert "غير متوفرة لديك" not in msg


def test_build_message_repost_warning_appears_when_given():
    msg = notifier.build_message("T", "u", 80.0, "p", repost_warning="⚠️ يبدو أنه معاد نشره")
    assert "معاد نشره" in msg


def test_build_message_repost_warning_absent_when_not_given():
    msg = notifier.build_message("T", "u", 80.0, "p")
    assert "معاد نشره" not in msg


def test_build_message_client_warning_included():
    msg = notifier.build_message("T", "u", 80.0, "p", client_warning="⚠️ Low-rated client")
    assert "Low-rated client" in msg


def test_build_message_price_and_delivery_lines():
    msg = notifier.build_message("T", "u", 80.0, "p", suggested_price="$200", delivery_days=5)
    assert "$200" in msg
    assert "5" in msg


def test_markdown_special_characters_are_escaped():
    msg = notifier.build_message("Title with _italic_ and *bold*", "u", 80.0, "p")
    # The raw underscores/asterisks from the title must be escaped so they
    # don't break Telegram's Markdown parsing.
    assert "\\_italic\\_" in msg
    assert "\\*bold\\*" in msg


def test_proposal_code_block_strips_triple_backticks():
    msg = notifier.build_message("T", "u", 80.0, "Some ```proposal``` text")
    assert "```proposal```" not in msg  # inner backticks must be stripped to avoid breaking the fence


def test_build_inline_keyboard_without_project_id():
    kb = notifier.build_inline_keyboard("https://mostaql.com/x")
    assert len(kb["inline_keyboard"]) == 1
    assert kb["inline_keyboard"][0][0]["url"] == "https://mostaql.com/x"


def test_build_inline_keyboard_with_project_id_adds_feedback_buttons():
    kb = notifier.build_inline_keyboard("https://mostaql.com/x", project_id="12345")
    assert len(kb["inline_keyboard"]) == 2
    buttons = kb["inline_keyboard"][1]
    callback_data = [b["callback_data"] for b in buttons]
    assert "won:12345" in callback_data
    assert "lost:12345" in callback_data
