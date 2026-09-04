"""
outcome_tracker.py
-------------------
Records win/loss outcomes for projects the bot sent a proposal for, via
Telegram inline-button taps ("✅ فاز بالمشروع" / "❌ لم يفز" — see
notifier.build_inline_keyboard, wired up by main.py's
telegram_feedback_loop). This closes the one feedback loop nothing else in
this codebase provides: whether an AI-drafted proposal that got sent
actually won the job. Nothing currently *uses* this data to change
behavior automatically — it's captured so you can look at MongoDB's
`outcomes` collection periodically and decide whether to adjust
MY_SKILLS, MATCH_THRESHOLD, or the proposal prompt based on real results
instead of guesswork.

Storage is MongoDB Atlas (see db.py) as of the Sept 2026 migration off a
local JSON file — same fail-safe philosophy as before: any read/write
error degrades to "no history" rather than raising, since a broken
connection must never interrupt the bot. Unlike the old local file, this
now genuinely survives a Render redeploy instead of resetting every time.
"""

import logging

import db

logger = logging.getLogger("outcome_tracker")


def record_outcome(project_id: str, title: str = None, outcome: str = None) -> bool:
    """
    Records (or updates) the outcome for one project_id. `outcome` must be
    "won" or "lost". Retapping the same or the other button for a project
    already recorded simply overwrites it (tracked via `updated_at`) rather
    than rejecting the correction — a human tapping the wrong button by
    mistake should be able to just tap the right one.

    Never raises; returns False on any failure (invalid outcome, MongoDB
    error) so the caller (main.py's callback handler) can tell the user
    the tap wasn't actually saved rather than showing a false confirmation.
    """
    if not project_id or outcome not in ("won", "lost"):
        logger.error(
            "record_outcome called with invalid arguments (project_id=%r, outcome=%r) — ignoring",
            project_id, outcome,
        )
        return False

    ok = db.record_outcome(project_id, title, outcome)
    if ok:
        logger.info("Outcome recorded for project %s ('%s'): %s", project_id, title, outcome)
    else:
        logger.error("Failed to record outcome for project %s", project_id)
    return ok


def get_stats() -> dict:
    """
    Returns {"won": N, "lost": N, "total": N} across everything recorded
    so far. Never raises — returns all-zero counts on any failure or if
    nothing has been recorded yet.
    """
    return db.get_outcome_stats()
