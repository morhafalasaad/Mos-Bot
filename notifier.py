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


def build_message(
    title: str,
    url: str,
    score: float,
    proposal_ar: str,
    budget: str = None,
    suggested_price: str = None,
    delivery_days: int = None,
    client_warning: str = None,
) -> str:
    # Placed right under the score, above price/delivery, so it's one of
    # the first things visible — a warning the user has to scroll past
    # defeats the point of "still able to apply, but aware."
    warning_line = f"\n{client_warning}" if client_warning else ""
    price_line = f"\n💵 *السعر المقترح:* {_escape_markdown(str(suggested_price))}" if suggested_price else ""
    days_line = f"\n⏱ *مدة التسليم المتوقعة:* {delivery_days} يوم" if delivery_days else ""
    budget_line = f"\n💰 *ميزانية العميل:* {_escape_markdown(budget)}" if budget else ""
    return (
        f"🆕 *مشروع جديد مطابق*\n\n"
        f"📌 *العنوان:* {_escape_markdown(title)}\n"
        f"📊 *نسبة التطابق:* {score:.0f}%"
        f"{warning_line}"
        f"{price_line}"
        f"{days_line}"
        f"{budget_line}\n\n"
        f"✍️ *مسودة العرض المقترح (اضغط للنسخ):*\n{_to_code_block(proposal_ar)}\n\n"
        f"_راجع العرض ثم استخدم الزر أدناه لفتح المشروع وإرسال العرض يدوياً._"
    )


def build_inline_keyboard(url: str) -> dict:
    """A single button that opens the project page directly on Mostaql —
    Telegram inline keyboard 'url' buttons require a valid absolute
    http(s) URL, which project.url always is (see scraper.py)."""
    return {
        "inline_keyboard": [
            [{"text": "🔗 فتح المشروع على مستقل", "url": url}],
        ]
    }


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
):
    message = build_message(
        title, url, score, proposal_ar, budget, suggested_price, delivery_days, client_warning,
    )
    send_telegram_message(message, reply_markup=build_inline_keyboard(url))


def notify_error(context: str, error_message: str):
    """Optional: ping yourself on Telegram if the worker hits repeated failures."""
    text = f"⚠️ *تنبيه خطأ في النظام*\n\nالسياق: {_escape_markdown(context)}\n{_escape_markdown(error_message[:500])}"
    send_telegram_message(text)
