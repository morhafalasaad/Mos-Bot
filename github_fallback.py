"""
github_fallback.py
-------------------
Fallback path for when the Gemini AI evaluation call itself fails — most
commonly 429 RESOURCE_EXHAUSTED after every key in GEMINI_API_KEYS has been
tried and exhausted (see ai_agent._generate()).

STRICT REQUIREMENT: no local storage. When AI evaluation is unavailable,
the raw scraped project data (title, link, budget, full description) is
sent directly to GitHub — either as a new Issue or as an uploaded Markdown
file in a designated repo — instead of being silently dropped or written to
local/ephemeral disk (which wouldn't even survive a Render redeploy anyway).
This is a genuine "nothing gets lost" guarantee: it either lands on GitHub
or the failure is logged loudly, never a silent local write.

Uses only `requests` against the GitHub REST API — no PyGithub/octokit
dependency needed for two simple calls (create an Issue / PUT a file).
"""

import base64
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import requests

import config

logger = logging.getLogger("github_fallback")

GITHUB_API_BASE = "https://api.github.com"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _format_project_markdown(project, reason: str) -> str:
    """Raw project details as Markdown — used as both the GitHub Issue body
    and the uploaded .md file's content. Includes everything scraped so a
    manual proposal can be written from this alone, without going back to
    Mostaql first."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    budget = project.budget or "غير محدد"
    tags = ", ".join(getattr(project, "tags", None) or []) or "غير متوفرة"

    return f"""# {project.title}

**السبب:** {reason}
**التاريخ (UTC):** {timestamp}
**الرابط:** {project.url}
**الميزانية المعلنة:** {budget}
**المهارات المطلوبة (إن توفرت):** {tags}

---

## الوصف الكامل

{project.description}
"""


def create_github_issue(project, reason: str) -> bool:
    """Creates a new GitHub Issue containing the full raw project details.
    Returns True on success. Never raises — logs and returns False on any
    failure, so a GitHub-side problem can't crash the caller either."""
    if not config.GITHUB_FALLBACK_ENABLED:
        logger.warning(
            "GitHub fallback not configured (GITHUB_TOKEN/GITHUB_REPO missing) "
            "— cannot save unevaluated project '%s'. It will be logged only.",
            project.title,
        )
        return False

    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/issues"
    payload = {
        "title": f"[مشروع بدون تقييم AI] {project.title}",
        "body": _format_project_markdown(project, reason),
        "labels": ["needs-manual-review", "ai-unavailable"],
    }

    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code == 201:
            issue_url = resp.json().get("html_url", "")
            logger.info("Saved unevaluated project '%s' to GitHub issue: %s", project.title, issue_url)
            return True
        logger.error(
            "GitHub issue creation failed for '%s' (HTTP %s): %s",
            project.title, resp.status_code, resp.text[:300],
        )
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("GitHub issue creation request failed for '%s': %s", project.title, exc)
        return False


def upload_github_markdown(project, reason: str) -> bool:
    """Uploads (creates) a Markdown file with the full raw project details
    under config.GITHUB_FALLBACK_DIR in the designated repo. Returns True
    on success. Never raises."""
    if not config.GITHUB_FALLBACK_ENABLED:
        logger.warning(
            "GitHub fallback not configured (GITHUB_TOKEN/GITHUB_REPO missing) "
            "— cannot save unevaluated project '%s'. It will be logged only.",
            project.title,
        )
        return False

    safe_id = getattr(project, "id", None) or "unknown"
    path = f"{config.GITHUB_FALLBACK_DIR}/{safe_id}.md"
    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/contents/{path}"

    content_str = _format_project_markdown(project, reason)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("ascii")

    payload = {
        "message": f"Add unevaluated project: {project.title}",
        "content": content_b64,
        "branch": config.GITHUB_FALLBACK_BRANCH,
    }

    try:
        resp = requests.put(url, json=payload, headers=_headers(), timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code in (200, 201):
            file_url = resp.json().get("content", {}).get("html_url", "")
            logger.info("Saved unevaluated project '%s' to GitHub file: %s", project.title, file_url)
            return True
        logger.error(
            "GitHub file upload failed for '%s' (HTTP %s): %s",
            project.title, resp.status_code, resp.text[:300],
        )
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("GitHub file upload request failed for '%s': %s", project.title, exc)
        return False


def save_project_to_github(project, reason: str) -> bool:
    """
    Dispatches to the configured fallback mode (config.GITHUB_FALLBACK_MODE:
    "issue" or "file"). This is a human-readable permanent record — it is
    NOT what gets read back for auto re-evaluation (see the queue functions
    below for that); this is purely for visibility/audit trail.
    Never raises — always returns a bool so the caller can log the outcome
    but a GitHub-side failure can never crash the main loop.
    """
    if config.GITHUB_FALLBACK_MODE == "file":
        return upload_github_markdown(project, reason)
    return create_github_issue(project, reason)


# ---------------------------------------------------------------------------
# Structured retry queue (pending_projects.json in the repo)
# ---------------------------------------------------------------------------
# This is the actual mechanism main.py reads back from each cycle to
# automatically re-evaluate projects once Gemini quota/rate limits allow.
# A JSON file (via GitHub's Contents API) is used instead of Issues because
# it's straightforward to read back as structured data, update in place,
# and remove entries from — none of which the Issues API gives you for free
# without significant extra complexity (searching by label, parsing a
# formatted body back into fields, closing issues after reprocessing, etc).
# ---------------------------------------------------------------------------

def _get_json_file(path: str):
    """GETs a JSON file's parsed content and its `sha` (required by the
    Contents API for updates — without the current sha, a PUT would be
    rejected as a conflict). Returns (None, None) if the file doesn't exist
    yet (a brand-new repo before the first queue write) or on any error —
    never raises."""
    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/contents/{path}"
    params = {"ref": config.GITHUB_FALLBACK_BRANCH}
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code == 404:
            return None, None
        if resp.status_code != 200:
            logger.error("GitHub GET %s failed (HTTP %s): %s", path, resp.status_code, resp.text[:300])
            return None, None
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content), data["sha"]
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        logger.error("Could not read/parse GitHub file %s: %s", path, exc)
        return None, None


def _put_json_file(path: str, obj, sha: Optional[str], message: str) -> bool:
    """PUTs (creates or updates, based on whether `sha` is provided) a JSON
    file's content. Never raises."""
    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/contents/{path}"
    content_b64 = base64.b64encode(
        json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    ).decode("ascii")
    payload = {
        "message": message,
        "content": content_b64,
        "branch": config.GITHUB_FALLBACK_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(url, json=payload, headers=_headers(), timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code in (200, 201):
            return True
        logger.error("GitHub PUT %s failed (HTTP %s): %s", path, resp.status_code, resp.text[:300])
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("GitHub PUT %s request failed: %s", path, exc)
        return False


def queue_project(project, reason: str) -> bool:
    """
    Appends (or refreshes, if already present) a project entry in the
    GitHub-hosted pending-projects queue file. This is what
    load_pending_queue()/main.py's retry loop reads back from. Never
    raises; returns False (and logs) if GitHub fallback isn't configured or
    the write fails — the project is still preserved via
    save_project_to_github()'s Issue/file record either way.
    """
    if not config.GITHUB_FALLBACK_ENABLED:
        logger.warning(
            "GitHub fallback not configured — cannot queue '%s' for auto-retry.",
            project.title,
        )
        return False

    queue, sha = _get_json_file(config.GITHUB_QUEUE_FILE)
    if queue is None:
        queue = []

    # Replace any existing entry for the same project id rather than
    # duplicating it (e.g. if it fails again on a later cycle before this
    # entry is ever successfully reprocessed).
    queue = [e for e in queue if e.get("id") != project.id]
    queue.append({
        "id": project.id,
        "title": project.title,
        "description": project.description,
        "url": project.url,
        "budget": project.budget,
        "duration": getattr(project, "duration", None),
        "tags": getattr(project, "tags", None) or [],
        "client_warning": getattr(project, "client_warning", None),
        "reason": reason,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "retry_count": 0,
    })

    ok = _put_json_file(
        config.GITHUB_QUEUE_FILE, queue, sha,
        message=f"Queue pending project for auto-retry: {project.title}",
    )
    if ok:
        logger.info("Queued '%s' on GitHub for automatic re-evaluation (%s pending total)", project.title, len(queue))
    return ok


def load_pending_queue() -> List[dict]:
    """Reads the full pending-projects queue from GitHub. Never raises;
    returns [] if not configured, the file doesn't exist yet, or on any
    read error — an empty queue is always a safe default (nothing to
    retry)."""
    if not config.GITHUB_FALLBACK_ENABLED:
        return []
    queue, _sha = _get_json_file(config.GITHUB_QUEUE_FILE)
    return queue or []


def save_pending_queue(queue: List[dict], message: str) -> bool:
    """
    Overwrites the queue file with the given (presumably modified) list —
    used after removing successfully-reprocessed entries or dropping ones
    that exceeded GITHUB_QUEUE_MAX_RETRIES. Re-fetches the current `sha`
    immediately before writing to minimize (though, given a single-worker-
    thread bot, not eliminate entirely) the chance of a stale-sha conflict.
    Never raises.
    """
    if not config.GITHUB_FALLBACK_ENABLED:
        return False
    _current, sha = _get_json_file(config.GITHUB_QUEUE_FILE)
    return _put_json_file(config.GITHUB_QUEUE_FILE, queue, sha, message=message)
