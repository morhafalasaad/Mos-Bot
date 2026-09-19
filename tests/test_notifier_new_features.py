"""
Tests for notifier.py's additions: build_screening_section (screening
Q&A, one independent tap-to-copy block per pair), the budget/timeline
adjustment note in build_message, and build_reply_options_message (Reply
Assistant formatting). Kept in a separate file from test_notifier.py
rather than editing it directly, so the original file's coverage stays
untouched and independently re-runnable.
"""

import notifier


# ---- build_screening_section -------------------------------------------------

def test_empty_screening_answers_returns_empty_string():
    assert notifier.build_screening_section([]) == ""
    assert notifier.build_screening_section(None) == ""


def test_screening_section_includes_each_question_and_answer():
    answers = [
        {"question": "How do you propose to implement this?", "answer": "Using FastAPI and PostgreSQL."},
        {"question": "What architecture will you use?", "answer": "A layered architecture."},
    ]
    section = notifier.build_screening_section(answers)
    assert "How do you propose to implement this?" in section
    assert "Using FastAPI and PostgreSQL." in section
    assert "What architecture will you use?" in section
    assert "أسئلة الفحص" in section


def test_screening_section_gives_each_answer_its_own_code_block():
    """Each Q&A pair must be independently tap-to-copy — i.e. its own
    fenced code block, not one shared block for all answers."""
    answers = [
        {"question": "Q1?", "answer": "A1"},
        {"question": "Q2?", "answer": "A2"},
    ]
    section = notifier.build_screening_section(answers)
    # 2 questions -> 2 code blocks -> 4 triple-backtick fences total.
    assert section.count("```") == 4


def test_screening_section_questions_are_numbered():
    answers = [{"question": "First?", "answer": "A"}, {"question": "Second?", "answer": "B"}]
    section = notifier.build_screening_section(answers)
    assert "1. First?" in section
    assert "2. Second?" in section


def test_screening_section_skips_malformed_entries():
    answers = [
        {"question": "", "answer": "Should be skipped, no question"},
        {"question": "Valid?", "answer": ""},  # no answer -> skipped
        {"question": "Real one?", "answer": "Real answer."},
        "not even a dict",
    ]
    section = notifier.build_screening_section(answers)
    assert "Real one?" in section
    assert "Should be skipped" not in section
    assert "1. Real one?" in section  # renumbered to 1, not 3


def test_screening_section_all_malformed_returns_empty():
    answers = [{"question": "", "answer": ""}, {"question": None, "answer": None}]
    assert notifier.build_screening_section(answers) == ""


def test_screening_section_markdown_escapes_question_text():
    answers = [{"question": "What about _italic_ text?", "answer": "An answer."}]
    section = notifier.build_screening_section(answers)
    assert "\\_italic\\_" in section


def test_build_message_appends_screening_section():
    answers = [{"question": "How will you build this?", "answer": "With Python."}]
    msg = notifier.build_message(
        "Title", "https://x", 80.0, "proposal text", screening_answers=answers,
    )
    assert "أسئلة الفحص" in msg
    assert "How will you build this?" in msg


def test_build_message_no_screening_section_when_no_answers():
    msg = notifier.build_message("Title", "https://x", 80.0, "proposal text")
    assert "أسئلة الفحص" not in msg


# ---- budget/timeline adjustment note ----------------------------------------

def test_adjustment_note_shown_when_adjusted_and_note_present():
    msg = notifier.build_message(
        "T", "u", 80.0, "p",
        budget_timeline_adjusted=True,
        budget_timeline_note="الميزانية غير كافية لنطاق العمل.",
    )
    assert "تعديل عن طلب العميل" in msg
    assert "الميزانية غير كافية" in msg


def test_adjustment_note_hidden_when_not_adjusted():
    msg = notifier.build_message(
        "T", "u", 80.0, "p",
        budget_timeline_adjusted=False,
        budget_timeline_note="",
    )
    assert "تعديل عن طلب العميل" not in msg


def test_adjustment_note_hidden_when_adjusted_true_but_note_empty():
    """Defensive: adjusted=True with an empty note string must not render
    a blank/broken line — the note text is required for the line to show."""
    msg = notifier.build_message(
        "T", "u", 80.0, "p",
        budget_timeline_adjusted=True,
        budget_timeline_note="",
    )
    assert "تعديل عن طلب العميل" not in msg


def test_adjustment_note_default_is_hidden():
    """Confirms the defaults on build_message's new kwargs are safe —
    calling it exactly like every pre-existing caller (no new kwargs at
    all) must never show the adjustment line."""
    msg = notifier.build_message("T", "u", 80.0, "p")
    assert "تعديل عن طلب العميل" not in msg


def test_notify_matched_project_forwards_new_fields(monkeypatch):
    """End-to-end: notify_matched_project must actually pass the new
    kwargs through to build_message / send_telegram_message, not just
    accept them silently."""
    captured = {}

    def fake_send(text, reply_markup=None):
        captured["text"] = text
        return True

    monkeypatch.setattr(notifier, "send_telegram_message", fake_send)

    notifier.notify_matched_project(
        title="T", url="https://x", score=80.0, proposal_ar="p",
        budget_timeline_adjusted=True,
        budget_timeline_note="تم التعديل بسبب ميزانية غير واقعية.",
        screening_answers=[{"question": "Q?", "answer": "A"}],
    )

    assert "تم التعديل بسبب ميزانية غير واقعية" in captured["text"]
    assert "أسئلة الفحص" in captured["text"]


# ---- build_reply_options_message / send_reply_options -----------------------

def test_reply_options_message_includes_summary_and_all_options():
    options = [
        {"label": "مختصر ومباشر", "reply": "شكراً لرسالتكم."},
        {"label": "تفصيلي وتقني", "reply": "بخصوص الجدول الزمني..."},
    ]
    msg = notifier.build_reply_options_message("Pre-hire timeline question", options)
    assert "Pre-hire timeline question" in msg
    assert "مختصر ومباشر" in msg
    assert "شكراً لرسالتكم" in msg
    assert "تفصيلي وتقني" in msg


def test_reply_options_message_each_option_has_own_code_block():
    options = [
        {"label": "A", "reply": "Reply A"},
        {"label": "B", "reply": "Reply B"},
        {"label": "C", "reply": "Reply C"},
    ]
    msg = notifier.build_reply_options_message("summary", options)
    assert msg.count("```") == 6  # 3 options -> 3 blocks -> 6 fences


def test_reply_options_message_numbers_options():
    options = [{"label": "A", "reply": "x"}, {"label": "B", "reply": "y"}]
    msg = notifier.build_reply_options_message("s", options)
    assert "الخيار 1" in msg
    assert "الخيار 2" in msg


def test_reply_options_message_missing_label_falls_back_to_index():
    options = [{"label": "", "reply": "Some reply"}]
    msg = notifier.build_reply_options_message("s", options)
    assert "خيار 1" in msg  # falls back to "خيار {i}"


def test_reply_options_message_handles_no_summary():
    options = [{"label": "A", "reply": "x"}]
    msg = notifier.build_reply_options_message(None, options)
    assert "تحليل الموقف" not in msg  # summary line omitted entirely
    assert "الخيار 1" in msg


def test_reply_options_message_empty_options_shows_failure_notice():
    msg = notifier.build_reply_options_message(None, [])
    assert "تعذر توليد" in msg


def test_send_reply_options_calls_send_telegram_message(monkeypatch):
    captured = {}

    def fake_send(text, reply_markup=None):
        captured["text"] = text
        captured["reply_markup"] = reply_markup
        return True

    monkeypatch.setattr(notifier, "send_telegram_message", fake_send)

    result = notifier.send_reply_options("summary", [{"label": "A", "reply": "x"}])

    assert result is True
    assert "summary" in captured["text"]
    # Unlike matched-project notifications, no inline keyboard is attached.
    assert captured["reply_markup"] is None
