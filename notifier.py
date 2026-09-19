"""
notifier.py
-----------
Lightweight Telegram notifier using plain `requests` calls to the Telegram
Bot HTTP API (no heavy SDK needed). Sends the project title, match score,
an advisory client-profile warning (if any), price/delivery estimate, and
the AI-drafted Arabic proposal — with a tap-to-open inline button for the
project page and the proposal formatted as a tap-to-copy code block — so
the human can review and submit the proposal manually on Mostaql
(human-in-the-loop by design).

Matched-project notifications also include two feedback buttons ("✅ فاز
بالمشروع" / "❌ لم يفز") when a project_id is available — main.py's
telegram_feedback_loop listens for taps on these and records the outcome
via outcome_tracker.py. This is the only feedback loop in the whole
pipeline that learns whether a sent proposal actually won real work.

SCREENING QUESTIONS (new)
-------------------------------------------------------------------
When a project has explicit client screening questions (see
scraper.parse_screening_questions / ai_agent.draft_screening_answers),
build_message() appends a dedicated "أسئلة الفحص" section BELOW the main
proposal, with each question/answer pair in its OWN separate tap-to-copy
code block — never merged into one shared block — so a single tap always
copies exactly one answer, matching however Mostaql's screening-question
UI actually presents separate text boxes per question.

BUDGET/TIMELINE ADJUSTMENT NOTE (new)
-------------------------------------------------------------------
ai_agent.py's scoring call now follows a strict "use the client's own
stated budget/timeline as-is unless it's severely unrealistic" rule (see
ProjectScoreSchema). When it DID deviate, evaluation.budget_timeline_note
carries a short justification — build_message() shows that ONLY when a
deviation actually happened, right next to the price/delivery lines, so
the human reviewer immediately understands why the shown figure differs
from whatever the client asked for, instead of silently mismatching.

REPLY ASSISTANT (new)
-------------------------------------------------------------------
build_reply_options_message() formats the output of
reply_assistant.get_reply_options() for Telegram — one tap-to-copy code
block PER option, each preceded by its own short tone label, so picking
and copying a specific option is unambiguous. Sent by main.py's
telegram_reply_assistant_loop in direct response to a pasted client
message (a normal text message, not a button tap), separate from every
other notification flow in this file.
"""

import json
import logging

import requests

import config

logger = logging.getLogger("notifier")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"


def _escape_markdown(text: str) -> str:
    """Escape characters that break Telegram's legacy Markdown parser."""
    if not text:
        return ""
    for ch in ["_", "*", "`", "["]:
        text = text.replace(ch, f"\\{ch}")
    return text


def _to_code_block(text: str) -> str:
    """
    Wraps text in a Telegram Markdown fenced code block (```...```), which
    renders as tap-to-copy monospace text on Telegram mobile clients — the
    whole point being one tap to copy the proposal, no manual text
    selection, for maximum speed when submitting on Mostaql.

    Content inside a Telegram code block is NOT re-parsed for other
    Markdown entities (asterisks/underscores in the proposal text won't be
    misread as bold/italic) — but a literal backtick or triple-backtick
    sequence WITHIN the text could still break the fence itself, so those
    are stripped defensively. AI-drafted Arabic prose is extremely unlikely
    to contain them, but this costs nothing and closes the edge case.
    """
    if not text:
        return "```\n(لا يوجد نص عرض)\n```"
    safe = text.replace("```", "").replace("`", "'")
    return f"```\n{safe}\n```"


def build_screening_section(screening_answers: list) -> str:
    """
    Builds the "أسئلة الفحص" (screening questions) section appended below
    the main proposal in build_message() — one clearly separated,
    INDEPENDENTLY tap-to-copy code block per question/answer pair, so a
    single tap on Telegram always copies exactly one answer and nothing
    else. Returns "" (nothing appended) if screening_answers is empty,
    which is the common case (most projects have no screening questions).

    screening_answers is the list of {"question", "answer"} dicts on
    ai_agent.Evaluation.screening_answers (see
    ai_agent.draft_screening_answers). Defensive against malformed entries
    — never raises, silently skips anything that doesn't look right rather
    than breaking the whole Telegram message over one bad item.
    """
    if not screening_answers:
        return ""

    blocks = []
    for item in screening_answers:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        blocks.append(
            f"*{len(blocks) + 1}. {_escape_markdown(question)}*\n{_to_code_block(answer)}"
        )

    if not blocks:
        return ""

    return (
        "\n\n📋 *أسئلة الفحص (اضغط على كل إجابة للنسخ بشكل منفصل):*\n\n"
        + "\n\n".join(blocks)
    )


def build_message(
    title: str,
    url: str,
    score: float,
    proposal_ar: str,
    budget: str = None,
    suggested_price: str = None,
    delivery_days: int = None,
    client_warning: str = None,
    matched_skills: list = None,
    missing_skills: list = None,
    repost_warning: str = None,
    budget_timeline_adjusted: bool = False,
    budget_timeline_note: str = None,
    screening_answers: list = None,
) -> str:
    # Placed right under the score, above price/delivery, so it's one of
    # the first things visible — a warning the user has to scroll past
    # defeats the point of "still able to apply, but aware."
    warning_line = f"\n{client_warning}" if client_warning else ""
    repost_line = f"\n{repost_warning}" if repost_warning else ""
    price_line = f"\n💵 *السعر المقترح:* {_escape_markdown(str(suggested_price))}" if suggested_price else ""
    days_line = f"\n⏱ *مدة التسليم المتوقعة:* {delivery_days} يوم" if delivery_days else ""
    budget_line = f"\n💰 *ميزانية العميل:* {_escape_markdown(budget)}" if budget else ""
    # Shown ONLY when ai_agent.py's strict budget/timeline-adherence rule
    # actually deviated from what the client stated (see
    # ProjectScoreSchema.budget_timeline_adjusted) — the common case is
    # this stays empty because the bot committed to the client's own
    # figures as-is, which needs no extra explanation.
    adjustment_line = ""
    if budget_timeline_adjusted and budget_timeline_note:
        adjustment_line = f"\n⚖️ *تعديل عن طلب العميل:* {_escape_markdown(budget_timeline_note)}"
    # Fast-scan skill breakdown right under the score — lets you gut-check
    # a match without reading the full proposal first. Each list is
    # already capped to a handful of items by the schema (see
    # ai_agent.ProjectScoreSchema), so no further truncation needed here.
    skills_line = ""
    if matched_skills:
        skills_line += f"\n✅ *مهارات متطابقة:* {_escape_markdown(', '.join(matched_skills))}"
    if missing_skills:
        skills_line += f"\n➕ *مهارات غير متوفرة لديك:* {_escape_markdown(', '.join(missing_skills))}"
    screening_section = build_screening_section(screening_answers or [])
    return (
        f"🆕 *مشروع جديد مطابق*\n\n"
        f"📌 *العنوان:* {_escape_markdown(title)}\n"
        f"📊 *نسبة التطابق:* {score:.0f}%"
        f"{repost_line}"
        f"{skills_line}"
        f"{warning_line}"
        f"{price_line}"
        f"{days_line}"
        f"{budget_line}"
        f"{adjustment_line}\n\n"
        f"✍️ *مسودة العرض المقترح (اضغط للنسخ):*\n{_to_code_block(proposal_ar)}"
        f"{screening_section}\n\n"
        f"_راجع العرض ثم استخدم الزر أدناه لفتح المشروع وإرسال العرض يدوياً._"
    )


def build_inline_keyboard(url: str, project_id: str = None) -> dict:
    """
    A button that opens the project page directly on Mostaql — Telegram
    inline keyboard 'url' buttons require a valid absolute http(s) URL,
    which project.url always is (see scraper.py).

    When project_id is given, a second row of outcome-tracking buttons is
    added: "✅ فاز بالمشروع" / "❌ لم يفز", with callback_data
    "won:<id>" / "lost:<id>" that main.py's telegram_feedback_loop listens
    for (see outcome_tracker.py). Kept as a SEPARATE row from the "open
    project" button so the two purposes (act now vs. record later) stay
    visually distinct rather than crowding one row.
    """
    keyboard = [[{"text": "🔗 فتح المشروع على مستقل", "url": url}]]
    if project_id:
        keyboard.append([
            {"text": "✅ فاز بالمشروع", "callback_data": f"won:{project_id}"},
            {"text": "❌ لم يفز", "callback_data": f"lost:{project_id}"},
        ])
    return {"inline_keyboard": keyboard}


def build_pending_message(
    title: str,
    budget: str = None,
    duration: str = None,
    description: str = None,
) -> str:
    """
    For the 'AI evaluation paused' case (Gemini quota/rate-limit exhausted
    on every configured key) — sent INSTANTLY when this happens, with the
    raw scraped fields Mostaql provides (no AI score/proposal exist yet,
    since that's exactly what failed). Deliberately mirrors
    build_message()'s visual structure (same emoji-labeled fields, same
    inline-button pattern via build_inline_keyboard) so it reads as the
    same family of notification, not a different, unfamiliar format.
    """
    budget_line = f"\n💰 *الميزانية / السعر المحدد من العميل:* {_escape_markdown(str(budget))}" if budget else "\n💰 *الميزانية / السعر المحدد من العميل:* غير محددة"
    duration_line = f"\n⏳ *مدة التسليم المطلوبة:* {_escape_markdown(str(duration))}" if duration else "\n⏳ *مدة التسليم المطلوبة:* غير محددة"
    desc = description or "غير متوفر"
    return (
        f"⏸️ *تم إيقاف تقييم المشروع مؤقتاً — تجاوز حد Gemini API*\n\n"
        f"📌 *اسم المشروع:* {_escape_markdown(title)}"
        f"{budget_line}"
        f"{duration_line}\n\n"
        f"📝 *تفاصيل المشروع الكاملة:*\n{_to_code_block(desc)}\n\n"
        f"_تم حفظ المشروع تلقائياً على GitHub وسيُعاد تقييمه بالذكاء الاصطناعي "
        f"تلقائياً بمجرد تجدد الحصة — لا حاجة لأي إجراء الآن، أو يمكنك فتح "
        f"المشروع وكتابة عرض يدوياً باستخدام الزر أدناه._"
    )


def notify_pending_project(
    title: str,
    url: str,
    budget: str = None,
    duration: str = None,
    description: str = None,
):
    """Sends the instant 'AI evaluation paused' alert — same inline-keyboard
    button as a successful match (build_inline_keyboard), so the tap-to-open
    behavior is identical between the two notification types."""
    message = build_pending_message(title, budget, duration, description)
    send_telegram_message(message, reply_markup=build_inline_keyboard(url))


def send_telegram_message(text: str, reply_markup: dict = None) -> bool:
    """Send a message to the configured chat, optionally with an inline
    keyboard. Returns True on success, never raises."""
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        # True removes Telegram's link-preview card (the blue box with the
        # site logo/title) that otherwise renders under the message for the
        # Mostaql URL — keeps the notification compact.
        "disable_web_page_preview": True,
    }
    if reply_markup:
        # Telegram's Bot API requires reply_markup to be a JSON-serialized
        # string when sent as form-encoded data (as opposed to a raw JSON
        # request body) — passing the dict directly would be wrongly
        # stringified by requests and silently rejected by Telegram.
        payload["reply_markup"] = json.dumps(reply_markup)

    try:
        resp = requests.post(TELEGRAM_API_URL, data=payload, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code == 200:
            logger.info("Telegram notification sent successfully")
            return True

        logger.error("Telegram API error %s: %s", resp.status_code, resp.text[:300])
        # Retry once as plain text if Markdown parsing was the problem.
        # reply_markup (already JSON-stringified above) is preserved in the
        # retry since we only remove parse_mode from the same payload dict.
        if resp.status_code == 400:
            payload.pop("parse_mode", None)
            retry_resp = requests.post(TELEGRAM_API_URL, data=payload, timeout=config.REQUEST_TIMEOUT)
            if retry_resp.status_code == 200:
                logger.info("Telegram notification sent successfully (plain-text fallback)")
                return True
            logger.error("Telegram retry also failed: %s", retry_resp.text[:300])
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Telegram request failed: %s", exc)
        return False


def notify_matched_project(
    title: str,
    url: str,
    score: float,
    proposal_ar: str,
    budget: str = None,
    suggested_price: str = None,
    delivery_days: int = None,
    client_warning: str = None,
    project_id: str = None,
    matched_skills: list = None,
    missing_skills: list = None,
    repost_warning: str = None,
    budget_timeline_adjusted: bool = False,
    budget_timeline_note: str = None,
    screening_answers: list = None,
):
    message = build_message(
        title, url, score, proposal_ar, budget, suggested_price, delivery_days, client_warning,
        matched_skills, missing_skills, repost_warning,
        budget_timeline_adjusted=budget_timeline_adjusted,
        budget_timeline_note=budget_timeline_note,
        screening_answers=screening_answers,
    )
    send_telegram_message(message, reply_markup=build_inline_keyboard(url, project_id))


def notify_error(context: str, error_message: str):
    """Optional: ping yourself on Telegram if the worker hits repeated failures."""
    text = f"⚠️ *تنبيه خطأ في النظام*\n\nالسياق: {_escape_markdown(context)}\n{_escape_markdown(error_message[:500])}"
    send_telegram_message(text)


# ---------------------------------------------------------------------------
# Reply Assistant formatting (new)
# ---------------------------------------------------------------------------

def build_reply_options_message(situation_summary: str, options: list) -> str:
    """
    Formats reply_assistant.get_reply_options()'s output for Telegram —
    one clearly labeled, INDEPENDENTLY tap-to-copy code block per option
    (same "each answer its own block" principle as
    build_screening_section), so picking and copying exactly one drafted
    reply is unambiguous even with 3 options in the same message.

    options is a list of {"label", "reply"} dicts. Returns a message
    explaining no usable draft could be produced if `options` is empty —
    callers should still send this (rather than nothing) so the human
    knows the paste was received but drafting failed, instead of silently
    getting no response at all.
    """
    if not options:
        return (
            "⚠️ *تعذر توليد ردود مقترحة لهذه الرسالة*\n\n"
            "حاول لصق نص الرسالة مرة أخرى، أو تأكد أنها رسالة عميل فعلية "
            "وليست نصاً قصيراً جداً."
        )

    summary_line = f"🧭 *تحليل الموقف:* {_escape_markdown(situation_summary)}\n\n" if situation_summary else ""

    blocks = []
    for i, opt in enumerate(options, start=1):
        label = opt.get("label") or f"خيار {i}"
        reply = opt.get("reply") or ""
        blocks.append(f"*الخيار {i} — {_escape_markdown(label)}:*\n{_to_code_block(reply)}")

    return (
        f"💬 *مساعد الرد على العميل*\n\n"
        f"{summary_line}"
        + "\n\n".join(blocks)
        + "\n\n_اختر الرد الأنسب واضغط عليه للنسخ، ثم أرسله يدوياً على مستقل._"
    )


def send_reply_options(situation_summary: str, options: list) -> bool:
    """Sends the formatted Reply Assistant message. No inline keyboard —
    unlike matched-project notifications, there's no per-message follow-up
    action (open project / record outcome) that applies here; the human's
    only next step is copying whichever option they pick."""
    message = build_reply_options_message(situation_summary, options)
    return send_telegram_message(message)
