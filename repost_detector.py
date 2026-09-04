"""
repost_detector.py
-------------------
Flags when a newly-matched project is very likely a repost or
resubmission of a project already notified about recently. Mostaql
assigns a NEW project id to a repost, so seen_projects dedup (see
scraper.py) can't catch this on its own — and even a slightly reworded
repost defeats ScoreCache's exact-content-hash matching (see
ai_agent.ScoreCache), since that requires byte-for-byte identical text.

Uses stdlib difflib.SequenceMatcher for a lightweight, dependency-free
text-similarity check against a small, recency-bounded store of recently-
notified projects' text — no ML/embeddings, consistent with this
codebase's "no heavy SDK" philosophy (see notifier.py's docstring).

Storage is MongoDB Atlas's `repost_history` collection (see db.py) as of
the Sept 2026 migration off a local JSON file — a TTL index there expires
entries older than REPOST_HISTORY_MAX_AGE_DAYS automatically (see
db._ensure_indexes), replacing the old manual prune-on-every-call logic.

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

import logging
from typing import Optional
from difflib import SequenceMatcher

import config
import db

logger = logging.getLogger("repost_detector")


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


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

    History is bounded by config.REPOST_HISTORY_MAX_AGE_DAYS — MongoDB's
    TTL index prunes entries older than that automatically, and
    get_repost_history() additionally filters by age at read time as a
    belt-and-suspenders measure (a TTL sweep runs roughly once a minute
    on Atlas, not instantly, so a query moments after expiry could
    otherwise still see a stale entry).
    """
    if not config.REPOST_DETECTION_ENABLED:
        return None

    normalized = _normalize(f"{title} {description}")
    if not normalized:
        return None

    try:
        history = db.get_repost_history(config.REPOST_HISTORY_MAX_AGE_DAYS)

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

        db.add_repost_entry(project_id, title, normalized)
        return warning
    except Exception:
        logger.error("Repost detection failed unexpectedly for '%s'", title, exc_info=True)
        return None
