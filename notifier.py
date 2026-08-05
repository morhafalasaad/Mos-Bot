"""
notifier.py
-----------
Lightweight Telegram notifier using plain `requests` calls to the Telegram
Bot HTTP API (no heavy SDK needed). Sends the project title, match score,
link, and AI-drafted Arabic proposal so the human can review and submit
the proposal manually on Mostaql (human-in-the-loop by design).
"""

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


def build_message(
    title: str,
    url: str,
    score: float,
    proposal_ar: str,
    budget: str = None,
    suggested_price: str = None,
    delivery_days: int = None,
) -> str:
    price_line = f"\n💵 *السعر المقترح:* {_escape_markdown(str(suggested_price))}" if suggested_price else ""
    days_line = f"\n⏱ *مدة التسليم المتوقعة:* {delivery_days} يوم" if delivery_days else ""
    budget_line = f"\n💰 *ميزانية العميل:* {_escape_markdown(budget)}" if budget else ""
    return (
        f"🆕 *مشروع جديد مطابق*\n\n"
        f"📌 *العنوان:* {_escape_markdown(title)}\n"
        f"📊 *نسبة التطابق:* {score:.0f}%"
        f"{price_line}"
        f"{days_line}"
        f"{budget_line}\n"
        f"🔗 *الرابط:* {url}\n\n"
        f"✍️ *مسودة العرض المقترح:*\n{proposal_ar}\n\n"
        f"_راجع العرض ثم قم بإرساله يدوياً على منصة مستقل._"
    )


def send_telegram_message(text: str) -> bool:
    """Send a message to the configured chat. Returns True on success, never raises."""
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        # True removes Telegram's link-preview card (the blue box with the
        # site logo/title) that otherwise renders under the message for the
        # Mostaql URL — keeps the notification compact.
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(TELEGRAM_API_URL, data=payload, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code == 200:
            logger.info("Telegram notification sent successfully")
            return True

        logger.error("Telegram API error %s: %s", resp.status_code, resp.text[:300])
        # Retry once as plain text if Markdown parsing was the problem.
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
):
    message = build_message(title, url, score, proposal_ar, budget, suggested_price, delivery_days)
    send_telegram_message(message)


def notify_error(context: str, error_message: str):
    """Optional: ping yourself on Telegram if the worker hits repeated failures."""
    text = f"⚠️ *تنبيه خطأ في النظام*\n\nالسياق: {_escape_markdown(context)}\n{_escape_markdown(error_message[:500])}"
    send_telegram_message(text)
