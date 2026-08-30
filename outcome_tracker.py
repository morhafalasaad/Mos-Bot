"""
outcome_tracker.py
-------------------
Records win/loss outcomes for projects the bot sent a proposal for, via
Telegram inline-button taps ("✅ فاز بالمشروع" / "❌ لم يفز" — see
notifier.build_inline_keyboard, wired up by main.py's
telegram_feedback_loop). This closes the one feedback loop nothing else in
this codebase provides: whether an AI-drafted proposal that got sent
actually won the job. Nothing currently *uses* this data to change
behavior automatically — it's captured so you can look at outcomes.json
periodically and decide whether to adjust MY_SKILLS, MATCH_THRESHOLD, or
the proposal prompt based on real results instead of guesswork.

Storage is a simple local JSON file, same fail-safe philosophy as
TokenUsageTracker/ScoreCache: any read/write error degrades to "no
history" rather than raising, since a broken file must never interrupt
the bot. Like seen_projects.json, this resets on a Render redeploy (see
README's ephemeral-local-disk caveat) — acceptable for the same reason
token usage stats are: it's an analytics/insight file, not something the
bot's own decisions depend on at runtime.
"""

import json
import logging
import threading
from datetime import datetime, timezone

import config

logger = logging.getLogger("outcome_tracker")

# Guards read-modify-write access to the outcomes file — the Telegram
# feedback listener runs on its own thread, separate from producer/
# consumer, so a tap arriving while something else touches the file could
# otherwise race.
_lock = threading.Lock()


def record_outcome(project_id: str, title: str = None, outcome: str = None) -> bool:
    """
    Records (or updates) the outcome for one project_id. `outcome` must be
    "won" or "lost". Retapping the same or the other button for a project
    already recorded simply overwrites it (tracked via `updated_at`) rather
    than rejecting the correction — a human tapping the wrong button by
    mistake should be able to just tap the right one.

    Never raises; returns False on any failure (invalid outcome, file I/O
    error) so the caller (main.py's callback handler) can tell the user
    the tap wasn't actually saved rather than showing a false confirmation.
    """
    if not project_id or outcome not in ("won", "lost"):
        logger.error(
            "record_outcome called with invalid arguments (project_id=%r, outcome=%r) — ignoring",
            project_id, outcome,
        )
        return False

    try:
        with _lock:
            try:
                with open(config.OUTCOME_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                data = {}

            now = datetime.now(timezone.utc).isoformat()
            existing = data.get(project_id) or {}
            data[project_id] = {
                "title": title or existing.get("title"),
                "outcome": outcome,
                "recorded_at": existing.get("recorded_at", now),
                "updated_at": now,
            }

            with open(config.OUTCOME_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Outcome recorded for project %s ('%s'): %s", project_id, title, outcome)
        return True
    except Exception:
        logger.error("Failed to record outcome for project %s", project_id, exc_info=True)
        return False


def get_stats() -> dict:
    """
    Returns {"won": N, "lost": N, "total": N} across everything recorded
    so far. Never raises — returns all-zero counts on any failure or if
    the file doesn't exist yet (nothing recorded).
    """
    try:
        with _lock:
            with open(config.OUTCOME_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        if not isinstance(data, dict):
            return {"won": 0, "lost": 0, "total": 0}
        won = sum(1 for v in data.values() if isinstance(v, dict) and v.get("outcome") == "won")
        lost = sum(1 for v in data.values() if isinstance(v, dict) and v.get("outcome") == "lost")
        return {"won": won, "lost": lost, "total": won + lost}
    except Exception:
        return {"won": 0, "lost": 0, "total": 0}
