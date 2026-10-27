"""
secure_logging.py
-----------------
Keeps secrets out of log output (Render's log dashboard retains and displays
everything the process prints).

Why this exists: Telegram's Bot API puts the bot token in the URL PATH
(https://api.telegram.org/bot<TOKEN>/sendMessage), and `requests` copies the
full URL into the message of every ConnectionError / SSLError / timeout it
raises. Any `logger.error("...failed: %s", exc)` therefore wrote the token
to the logs exactly when a TLS handshake or network call failed. Likewise,
GEMINI_PROXY_URL (http://user:pass@host:port) and MONGODB_URI carry
credentials inline.

RedactingFormatter masks both known secret VALUES (read from the
environment) and well-known secret SHAPES, on the final formatted text —
including tracebacks and third-party loggers such as urllib3 at DEBUG.
"""

import logging
import os
import re
from urllib.parse import urlsplit

_REDACTED = "<redacted>"

# Env vars whose values are secrets. Comma-separated lists are split.
_SECRET_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "GEMINI_API_KEYS",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "MONGODB_URI",
    "GEMINI_PROXY_URL",
)

# Shapes that are secrets regardless of where they came from.
_PATTERNS = [
    # Telegram bot token inside a URL path: /bot123456789:AAH...
    (re.compile(r"(?<=/bot)\d{5,}:[A-Za-z0-9_-]{20,}"), _REDACTED),
    # Google API keys
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), _REDACTED),
    # GitHub tokens
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|github_pat_[A-Za-z0-9_]{20,}"), _REDACTED),
    # Authorization: Bearer xxx
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + _REDACTED),
    # user:password@ inside any URL (mongodb+srv://, http://proxy, ...)
    (re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+:[^@\s/]+@"), r"\1" + _REDACTED + "@"),
]

_MIN_SECRET_LEN = 6


def _secret_values() -> list:
    values = set()
    for name in _SECRET_ENV_VARS:
        raw = os.getenv(name, "")
        if not raw:
            continue
        parts = raw.split(",") if name == "GEMINI_API_KEYS" else [raw]
        for part in parts:
            part = part.strip().strip('"').strip("'").strip()
            if len(part) >= _MIN_SECRET_LEN:
                values.add(part)
            # Also mask the bare password of a credentialed URL.
            if "://" in part:
                try:
                    pw = urlsplit(part).password
                except ValueError:
                    pw = None
                if pw and len(pw) >= _MIN_SECRET_LEN:
                    values.add(pw)
    # Longest first so a full URI is masked before any substring of it.
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """Returns `text` with known secret values and secret-shaped strings masked."""
    if not text:
        return text
    for secret in _secret_values():
        if secret in text:
            text = text.replace(secret, _REDACTED)
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def mask_url_credentials(url: str) -> str:
    """'http://user:pass@host:8080' -> 'http://host:8080' (safe to log)."""
    if not url:
        return url
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return f"{parts.scheme}://{host}"
    except ValueError:
        return _REDACTED
    return url


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the final rendered record (message + traceback)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))
