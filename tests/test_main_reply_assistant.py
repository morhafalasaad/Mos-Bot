"""
Tests for main.py's Reply Assistant integration:
  - _handle_text_message's routing (chat_id filter, short-message skip)
  - The single shared telegram_feedback_loop correctly dispatches BOTH
    callback_query (Won/Lost) and message (Reply Assistant) updates from
    ONE getUpdates connection — this is a deliberate design constraint
    (Telegram allows only one active long-poll connection per bot token),
    not an implementation detail, so it gets its own explicit test.
"""

import config
import main
import notifier
import reply_assistant


def test_handle_text_message_routes_correct_chat_id(monkeypatch):
    sent = []
    monkeypatch.setattr(reply_assistant, "get_reply_options", lambda *a, **kw: ("summary", [{"label": "A", "reply": "B"}], {}))
    monkeypatch.setattr(notifier, "send_reply_options", lambda summary, options: sent.append((summary, options)))

    main._handle_text_message({
        "text": "When can you start on this and what is your rate?",
        "chat": {"id": int(config.TELEGRAM_CHAT_ID)},
    })

    assert len(sent) == 1


def test_handle_text_message_ignores_wrong_chat_id(monkeypatch):
    sent = []
    monkeypatch.setattr(reply_assistant, "get_reply_options", lambda *a, **kw: ("summary", [{"label": "A", "reply": "B"}], {}))
    monkeypatch.setattr(notifier, "send_reply_options", lambda summary, options: sent.append((summary, options)))

    main._handle_text_message({
        "text": "When can you start on this and what is your rate?",
        "chat": {"id": int(config.TELEGRAM_CHAT_ID) + 1},
    })

    assert sent == []


def test_handle_text_message_ignores_short_message_without_calling_gemini(monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("must not call Gemini for a too-short message")

    monkeypatch.setattr(reply_assistant, "get_reply_options", fail_if_called)

    main._handle_text_message({"text": "ok", "chat": {"id": int(config.TELEGRAM_CHAT_ID)}})
    # No assertion error raised = success; nothing to send either.


def test_handle_text_message_ignores_bot_commands(monkeypatch):
    def fail_if_called(*a, **kw):
        raise AssertionError("must not treat a bot command as a client message")

    monkeypatch.setattr(reply_assistant, "get_reply_options", fail_if_called)

    main._handle_text_message({"text": "/start", "chat": {"id": int(config.TELEGRAM_CHAT_ID)}})


def test_handle_text_message_sends_fallback_notice_on_reply_assistant_exception(monkeypatch):
    sent_texts = []

    def broken_get_reply_options(*a, **kw):
        raise RuntimeError("simulated Gemini outage")

    monkeypatch.setattr(reply_assistant, "get_reply_options", broken_get_reply_options)
    monkeypatch.setattr(notifier, "send_telegram_message", lambda text, reply_markup=None: sent_texts.append(text))

    main._handle_text_message({
        "text": "A perfectly normal client message asking about pricing",
        "chat": {"id": int(config.TELEGRAM_CHAT_ID)},
    })

    assert len(sent_texts) == 1
    assert "خطأ" in sent_texts[0]  # error notice, not silently swallowed


def test_handle_text_message_missing_chat_id_does_not_crash(monkeypatch):
    """A message update with no 'chat' key at all (malformed/unexpected
    shape) must not raise — chat_id becomes None, which the comparison
    logic must handle gracefully rather than crashing the listener."""
    sent = []
    monkeypatch.setattr(reply_assistant, "get_reply_options", lambda *a, **kw: ("s", [{"label": "A", "reply": "B"}], {}))
    monkeypatch.setattr(notifier, "send_reply_options", lambda summary, options: sent.append(1))

    # chat_id is None here -> `if chat_id is not None and ...` short-circuits,
    # so a message with no chat info at all is processed rather than
    # silently dropped (this matches the code's explicit None-check).
    main._handle_text_message({"text": "A message with no chat info at all here"})
    assert len(sent) == 1


# ---- Shared listener dispatch (callback_query vs message) -------------------

def test_feedback_loop_dispatches_callback_and_message_updates_separately(monkeypatch):
    """End-to-end: one getUpdates response containing BOTH a callback_query
    update and a message update must route each to its own handler —
    proof that the single shared listener (see telegram_feedback_loop's
    docstring on why there's only one) doesn't drop or misroute either
    kind of update."""
    callback_handled = []
    message_handled = []

    monkeypatch.setattr(main, "_handle_feedback_callback", lambda cb: callback_handled.append(cb))
    monkeypatch.setattr(main, "_handle_text_message", lambda msg: message_handled.append(msg))
    monkeypatch.setattr(config, "REPLY_ASSISTANT_ENABLED", True)
    monkeypatch.setattr(main, "_load_telegram_offset", lambda: 0)
    monkeypatch.setattr(main, "_save_telegram_offset", lambda offset: None)

    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"result": [
                {"update_id": 1, "callback_query": {"id": "cb1", "data": "won:proj1"}},
                {"update_id": 2, "message": {"text": "A client message here", "chat": {"id": 1}}},
            ]}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt("stop after one iteration")
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)

    try:
        main.telegram_feedback_loop()
    except KeyboardInterrupt:
        pass

    assert len(callback_handled) == 1
    assert callback_handled[0]["data"] == "won:proj1"
    assert len(message_handled) == 1
    assert message_handled[0]["text"] == "A client message here"


def test_feedback_loop_skips_message_dispatch_when_reply_assistant_disabled(monkeypatch):
    message_handled = []
    monkeypatch.setattr(main, "_handle_feedback_callback", lambda cb: None)
    monkeypatch.setattr(main, "_handle_text_message", lambda msg: message_handled.append(msg))
    monkeypatch.setattr(config, "REPLY_ASSISTANT_ENABLED", False)
    monkeypatch.setattr(main, "_load_telegram_offset", lambda: 0)
    monkeypatch.setattr(main, "_save_telegram_offset", lambda offset: None)

    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"result": [
                {"update_id": 1, "message": {"text": "A client message here", "chat": {"id": 1}}},
            ]}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt("stop after one iteration")
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)

    try:
        main.telegram_feedback_loop()
    except KeyboardInterrupt:
        pass

    assert message_handled == []  # disabled -> message updates never dispatched


def test_feedback_loop_allowed_updates_includes_message_when_enabled(monkeypatch):
    """Confirms the getUpdates request itself asks Telegram for 'message'
    updates when the feature is on — otherwise Telegram would never even
    deliver them regardless of the dispatch logic above."""
    monkeypatch.setattr(config, "REPLY_ASSISTANT_ENABLED", True)
    monkeypatch.setattr(main, "_load_telegram_offset", lambda: 0)
    monkeypatch.setattr(main, "_save_telegram_offset", lambda offset: None)

    seen_params = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"result": []}

    def fake_get(*args, **kwargs):
        seen_params.update(kwargs.get("params", {}))
        raise KeyboardInterrupt("stop after first call")

    monkeypatch.setattr(main.requests, "get", fake_get)

    try:
        main.telegram_feedback_loop()
    except KeyboardInterrupt:
        pass

    assert "message" in seen_params["allowed_updates"]
    assert "callback_query" in seen_params["allowed_updates"]


def test_feedback_loop_allowed_updates_excludes_message_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "REPLY_ASSISTANT_ENABLED", False)
    monkeypatch.setattr(main, "_load_telegram_offset", lambda: 0)
    monkeypatch.setattr(main, "_save_telegram_offset", lambda offset: None)

    seen_params = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"result": []}

    def fake_get(*args, **kwargs):
        seen_params.update(kwargs.get("params", {}))
        raise KeyboardInterrupt("stop after first call")

    monkeypatch.setattr(main.requests, "get", fake_get)

    try:
        main.telegram_feedback_loop()
    except KeyboardInterrupt:
        pass

    assert "message" not in seen_params["allowed_updates"]
    assert "callback_query" in seen_params["allowed_updates"]
