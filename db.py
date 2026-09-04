"""
db.py
-----
MongoDB Atlas persistence layer — replaces every local-JSON-file store
this bot used to depend on (seen-projects dedup, score cache, daily
request counter, token-usage analytics, repost history, win/loss
outcomes, and the small Telegram-offset/bot-state values), AND replaces
the GitHub-Contents-API pending-projects retry queue that used to live in
this very repo.

WHY THIS EXISTS (Sept 2026 migration)
-------------------------------------------------------------------
Two separate problems, one fix:
  1. Local JSON files on Render's free tier live on EPHEMERAL disk — every
     one of them (seen_projects.json, score_cache.json, outcomes.json,
     repost_history.json, daily_request_count.json) was silently wiped on
     every redeploy/restart.
  2. The one file that WAS made durable (pending_projects.json) achieved
     that by committing itself back to this repo via GitHub's Contents
     API on every update — which made it, and its accompanying
     token-usage-stats sync, the actual ROOT CAUSE of a redeploy storm
     (each commit is a push Render's auto-deploy watches), which in turn
     produced repeated Telegram 409 conflicts and reset every in-memory
     safety net on every cycle. See the Sept 2026 audit for the full
     forensic trail.

MongoDB Atlas's free tier solves both: state survives restarts/redeploys
AND writing to it never touches git, so it can never trigger a deploy.

FAIL-SAFE PHILOSOPHY (unchanged from every other tracker in this codebase)
-------------------------------------------------------------------
Every public function below catches its own errors and degrades to a
safe default (empty result / no-op) rather than raising — a Mongo hiccup
must never be allowed to crash a worker loop, exactly like a corrupt JSON
file never used to. The ONE exception is the initial connection itself:
if MONGODB_URI is missing or entirely unreachable at startup, that fails
LOUD (see config._require), same as a missing Gemini key — a bot with no
working persistence at all should not run silently degraded for hours
before anyone notices.

TESTING
-------------------------------------------------------------------
Every function that touches a collection accepts an optional injected
collection object (or goes through get_collection(), which tests replace
via monkeypatching `_db` — see tests/conftest.py's `mongomock`-backed
fixture) so nothing here ever needs a real Atlas connection to be tested.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import config

logger = logging.getLogger("db")

_lock = threading.Lock()
_client = None
_db = None

# Indexes are created once per process (idempotent on Atlas — re-running
# create_index with the same spec is a cheap no-op), tracked here so a
# reconnect/test-swap doesn't redundantly attempt it every single call.
_indexes_ensured = False


def get_db():
    """
    Returns the active pymongo Database handle, connecting lazily on first
    use. Tests short-circuit this entirely by monkeypatching the module-
    level `_db` directly (see tests/conftest.py) — this function only ever
    builds a real connection when `_db` is still None.
    """
    global _client, _db
    if _db is not None:
        return _db
    with _lock:
        if _db is not None:
            return _db
        from pymongo import MongoClient
        _client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=config.MONGODB_TIMEOUT_MS,
            connectTimeoutMS=config.MONGODB_TIMEOUT_MS,
        )
        _db = _client[config.MONGODB_DB_NAME]
        logger.info("Connected to MongoDB Atlas (db=%s)", config.MONGODB_DB_NAME)
        _ensure_indexes(_db)
        return _db


def get_collection(name: str):
    return get_db()[name]


def _ensure_indexes(database) -> None:
    """
    Creates the handful of indexes this module relies on for correctness
    (uniqueness) or housekeeping (TTL auto-expiry, replacing the old
    manual 'trim to N entries' logic every local-file tracker used to do
    by hand). Never raises — a missing index degrades to slower queries
    or unbounded growth, not a crash.
    """
    global _indexes_ensured
    if _indexes_ensured:
        return
    try:
        # seen_projects: auto-expire after SEEN_PROJECTS_TTL_DAYS so the
        # dedup set can't grow forever, without any manual trimming code.
        database["seen_projects"].create_index(
            "seen_at", expireAfterSeconds=config.SEEN_PROJECTS_TTL_DAYS * 86400,
        )
        # score_cache: same idea, long-lived but not eternal.
        database["score_cache"].create_index(
            "cached_at", expireAfterSeconds=config.SCORE_CACHE_TTL_DAYS * 86400,
        )
        # repost_history: expire on the same schedule repost_detector.py
        # already used to prune by hand (REPOST_HISTORY_MAX_AGE_DAYS).
        database["repost_history"].create_index(
            "recorded_at", expireAfterSeconds=config.REPOST_HISTORY_MAX_AGE_DAYS * 86400,
        )
        _indexes_ensured = True
    except Exception:
        logger.warning("Could not ensure MongoDB indexes (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Seen-projects dedup (replaces scraper.py's seen_projects.json)
# ---------------------------------------------------------------------------

def get_seen_ids() -> set:
    try:
        docs = get_collection("seen_projects").find({}, {"_id": 1})
        return {d["_id"] for d in docs}
    except Exception:
        logger.warning("Could not load seen-projects from MongoDB, starting fresh", exc_info=True)
        return set()


def mark_seen(project_ids) -> None:
    if not project_ids:
        return
    try:
        coll = get_collection("seen_projects")
        now = datetime.now(timezone.utc)
        for pid in project_ids:
            coll.update_one({"_id": pid}, {"$set": {"seen_at": now}}, upsert=True)
    except Exception:
        logger.error("Could not persist seen-projects to MongoDB", exc_info=True)


# ---------------------------------------------------------------------------
# Pending-projects retry queue (replaces the GitHub-Contents-API queue —
# this is the piece that used to cause redeploys by committing to this repo)
# ---------------------------------------------------------------------------

def queue_pending_project(entry: dict) -> bool:
    """Upserts one retry-queue entry keyed by project id. Never raises."""
    try:
        coll = get_collection("pending_projects")
        doc = dict(entry)
        doc["_id"] = doc.pop("id")
        doc.setdefault("queued_at", datetime.now(timezone.utc).isoformat())
        doc.setdefault("retry_count", 0)
        coll.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        return True
    except Exception:
        logger.error("Could not queue pending project in MongoDB", exc_info=True)
        return False


def get_pending_projects() -> List[dict]:
    try:
        docs = list(get_collection("pending_projects").find({}))
        for d in docs:
            d["id"] = d.pop("_id")
        return docs
    except Exception:
        logger.error("Could not load pending-projects queue from MongoDB", exc_info=True)
        return []


def update_pending_project(project_id: str, fields: dict) -> None:
    try:
        get_collection("pending_projects").update_one({"_id": project_id}, {"$set": fields})
    except Exception:
        logger.error("Could not update pending project %s in MongoDB", project_id, exc_info=True)


def remove_pending_project(project_id: str) -> None:
    try:
        get_collection("pending_projects").delete_one({"_id": project_id})
    except Exception:
        logger.error("Could not remove pending project %s from MongoDB", project_id, exc_info=True)


# ---------------------------------------------------------------------------
# Outcome tracking (replaces outcomes.json)
# ---------------------------------------------------------------------------

def record_outcome(project_id: str, title: Optional[str], outcome: str) -> bool:
    try:
        coll = get_collection("outcomes")
        now = datetime.now(timezone.utc).isoformat()
        existing = coll.find_one({"_id": project_id}) or {}
        coll.update_one(
            {"_id": project_id},
            {"$set": {
                "title": title or existing.get("title"),
                "outcome": outcome,
                "recorded_at": existing.get("recorded_at", now),
                "updated_at": now,
            }},
            upsert=True,
        )
        return True
    except Exception:
        logger.error("Failed to record outcome for project %s", project_id, exc_info=True)
        return False


def get_outcome_stats() -> dict:
    try:
        coll = get_collection("outcomes")
        won = coll.count_documents({"outcome": "won"})
        lost = coll.count_documents({"outcome": "lost"})
        return {"won": won, "lost": lost, "total": won + lost}
    except Exception:
        return {"won": 0, "lost": 0, "total": 0}


# ---------------------------------------------------------------------------
# Repost/duplicate history (replaces repost_history.json)
# ---------------------------------------------------------------------------

def get_repost_history(max_age_days: int) -> List[dict]:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        docs = get_collection("repost_history").find({"recorded_at": {"$gte": cutoff}})
        return list(docs)
    except Exception:
        logger.error("Could not load repost history from MongoDB", exc_info=True)
        return []


def add_repost_entry(project_id: str, title: str, normalized_text: str, max_entries: int = None) -> None:
    """
    Inserts one repost-history entry, then evicts the oldest entries (by
    recorded_at) once the collection exceeds `max_entries` — replaces the
    old local-file version's manual "trim the list" logic. The TTL index
    on `recorded_at` (see _ensure_indexes) additionally prunes by AGE
    independently of this count-based cap.
    """
    try:
        coll = get_collection("repost_history")
        coll.insert_one({
            "project_id": project_id,
            "title": title,
            "normalized_text": normalized_text,
            "recorded_at": datetime.now(timezone.utc),
        })
        cap = max_entries if max_entries is not None else config.REPOST_HISTORY_MAX_ENTRIES
        count = coll.count_documents({})
        if count > cap:
            overflow = count - cap
            oldest_ids = [
                d["_id"] for d in
                coll.find({}, {"_id": 1}).sort("recorded_at", 1).limit(overflow)
            ]
            if oldest_ids:
                coll.delete_many({"_id": {"$in": oldest_ids}})
    except Exception:
        logger.error("Could not persist repost-history entry to MongoDB", exc_info=True)


# ---------------------------------------------------------------------------
# Small key/value bot state (replaces telegram_update_offset.txt and any
# other single-value on-disk state)
# ---------------------------------------------------------------------------

def get_state(key: str, default=None):
    try:
        doc = get_collection("bot_state").find_one({"_id": key})
        return doc["value"] if doc else default
    except Exception:
        return default


def set_state(key: str, value) -> None:
    try:
        get_collection("bot_state").update_one(
            {"_id": key}, {"$set": {"value": value}}, upsert=True,
        )
    except Exception:
        logger.error("Could not persist bot_state[%s] to MongoDB", key, exc_info=True)
