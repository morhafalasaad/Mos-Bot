"""
repost_detector.py
-------------------
Flags when a newly-matched project is very likely a repost or
resubmission of a project already notified about recently. Mostaql
assigns a NEW project id to a repost, so seen_projects.json's ID-based
dedup (see scraper.py) can't catch this on its own — and even a slightly
reworded repost defeats ScoreCache's exact-content-hash matching (see
ai_agent.ScoreCache), since that requires byte-for-byte identical text.

Uses stdlib difflib.SequenceMatcher for a lightweight, dependency-free
text-similarity check against a small, recency-bounded store of recently-
notified projects' text — no ML/embeddings, consistent with this
codebase's "no heavy SDK" philosophy (see notifier.py's docstring).

Deliberately ADVISORY, not blocking: a likely-repost match still gets
notified normally, exactly as before — NEVER silently suppressed. A false
positive here (two different clients independently posting similarly-
worded but genuinely unrelated projects — common with generic/templated
descriptions) would silently cost a lead with zero visibility if this
blocked anything, which is worse than one avoidable Telegram ping. Instead
it just prepends a "⚠️ يبدو أنه معاد نشره" warning line, mirroring
scraper.build_client_warning's "advisory, never blocking" pattern exactly
— the human makes the final call, same as everywhere else in this
pipeline.

Only called for projects that are ABOUT to be notified (score cleared the
threshold) — see main.py's _handle_evaluation_result — so the comparison
set stays small and relevant rather than cluttered with every scraped
project regardless of relevance.
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional

import config

logger = logging.getLogger("repost_detector")

# Guards read-modify-write access — the consumer thread is the only one
# that calls this today, but the lock costs nothing and protects against
# future callers on a different thread.
_lock = threading.Lock()


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _load_history() -> list:
    try:
        with open(config.REPOST_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []  # missing file, corrupt JSON, anything else -> empty history


def _save_history(history: list) -> None:
    try:
        with open(config.REPOST_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass  # silent/fail-safe, matching every other tracker in this codebase


def check_and_record(project_id: str, title: str, description: str) -> Optional[str]:
    """
    Checks whether (title, description) looks like a near-duplicate of a
    recently-notified project, THEN records this one for future checks
    regardless of the result — so a repost of THIS project can also be
    caught later, even if this one wasn't itself flagged as a repost of
    something earlier.

    Returns a short Arabic warning string (ready to prepend to a Telegram
    notification) if similarity against some prior entry is >=
    config.REPOST_SIMILARITY_THRESHOLD, else None. Never raises — any
    failure degrades to "no repost detected" (returns None) rather than
    blocking a notification.

    Bounded by config.REPOST_HISTORY_MAX_ENTRIES and
    config.REPOST_HISTORY_MAX_AGE_DAYS so the comparison set can't grow
    unbounded, and old entries can't keep producing matches indefinitely.
    """
    if not config.REPOST_DETECTION_ENABLED:
        return None

    normalized = _normalize(f"{title} {description}")
    if not normalized:
        return None

    try:
        with _lock:
            history = _load_history()

            cutoff = datetime.now(timezone.utc) - timedelta(days=config.REPOST_HISTORY_MAX_AGE_DAYS)
            fresh_history = []
            for h in history:
                try:
                    if datetime.fromisoformat(h.get("recorded_at", "")) > cutoff:
                        fresh_history.append(h)
                except (ValueError, TypeError):
                    continue  # malformed timestamp -> drop this entry rather than crash
            history = fresh_history

            best_ratio = 0.0
            best_title = None
            for h in history:
                if h.get("project_id") == project_id:
                    continue  # don't compare a project against a prior record of itself
                ratio = SequenceMatcher(None, normalized, h.get("normalized_text", "")).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_title = h.get("title")

            warning = None
            if best_ratio >= config.REPOST_SIMILARITY_THRESHOLD:
                warning = (
                    f"⚠️ *يبدو أن هذا المشروع معاد نشره أو مكرر* "
                    f"(تشابه {best_ratio * 100:.0f}% مع: {best_title})"
                )
                logger.info(
                    "Likely repost: '%s' is %.0f%% similar to previously-notified '%s'",
                    title, best_ratio * 100, best_title,
                )

            history.append({
                "project_id": project_id,
                "title": title,
                "normalized_text": normalized,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            if len(history) > config.REPOST_HISTORY_MAX_ENTRIES:
                history = history[-config.REPOST_HISTORY_MAX_ENTRIES:]

            _save_history(history)
            return warning
    except Exception:
        logger.error("Repost detection failed unexpectedly for '%s'", title, exc_info=True)
        return None
