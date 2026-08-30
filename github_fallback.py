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
import os
import re
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


def _log_request_exception(context: str, exc: Exception) -> None:
    """
    Logs a GitHub API request failure — with a SPECIFIC, actionable message
    when it's an SSL/certificate error, since that's a distinct root cause
    (typically a stale/incomplete CA bundle in the container image — see
    the certifi fix at the top of main.py) from an ordinary network drop or
    GitHub-side outage. This is diagnostic only: it does NOT change control
    flow or weaken verification. Every call site already catches
    requests.exceptions.RequestException (which SSLError is a subclass
    of — requests.exceptions.SSLError -> ConnectionError -> RequestException),
    so this never needs its own except clause; call it from inside the
    existing except block instead.
    """
    if isinstance(exc, requests.exceptions.SSLError):
        logger.error(
            "GitHub API SSL/certificate verification failed during %s: %s. "
            "This is almost always a stale/incomplete CA bundle in the "
            "container image, not an actual network attack — confirm "
            "main.py's certifi CA-bundle fix (REQUESTS_CA_BUNDLE/"
            "SSL_CERT_FILE, set at the very top of the file) is active. "
            "This request was intentionally NOT retried with verification "
            "disabled — see main.py's module docstring for why.",
            context, exc,
        )
    else:
        logger.error("GitHub API request failed during %s: %s", context, exc)


ISSUE_TITLE_PREFIX = "[مشروع بدون تقييم AI] "
FALLBACK_ISSUE_LABEL = "ai-unavailable"


def _format_project_markdown(project, reason: str) -> str:
    """Raw project details as Markdown — used as both the GitHub Issue body
    and the uploaded .md file's content. Includes everything scraped so a
    manual proposal can be written from this alone, without going back to
    Mostaql first. Field labels here are matched exactly by parse_issue_body()
    below when reading an issue back for auto re-evaluation — keep them in
    sync if this format ever changes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    budget = project.budget or "غير محدد"
    duration = getattr(project, "duration", None) or "غير محددة"
    tags = ", ".join(getattr(project, "tags", None) or []) or "غير متوفرة"

    return f"""# {project.title}

**السبب:** {reason}
**التاريخ (UTC):** {timestamp}
**الرابط:** {project.url}
**الميزانية المعلنة:** {budget}
**مدة التسليم المطلوبة:** {duration}
**المهارات المطلوبة (إن توفرت):** {tags}

---

## الوصف الكامل

{project.description}
"""


def create_github_issue(project, reason: str) -> Optional[int]:
    """Creates a new GitHub Issue containing the full raw project details.
    Returns the issue NUMBER on success (used to link/track it — see
    queue_project's issue_number param — and to close it later), or None
    on failure. Never raises."""
    if not config.GITHUB_FALLBACK_ENABLED:
        logger.warning(
            "GitHub fallback not configured (GITHUB_TOKEN/GITHUB_REPO missing) "
            "— cannot save unevaluated project '%s'. It will be logged only.",
            project.title,
        )
        return None

    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/issues"
    payload = {
        "title": f"{ISSUE_TITLE_PREFIX}{project.title}",
        "body": _format_project_markdown(project, reason),
        "labels": ["needs-manual-review", FALLBACK_ISSUE_LABEL],
    }

    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code == 201:
            data = resp.json()
            logger.info("Saved unevaluated project '%s' to GitHub issue: %s", project.title, data.get("html_url", ""))
            return data.get("number")
        logger.error(
            "GitHub issue creation failed for '%s' (HTTP %s): %s",
            project.title, resp.status_code, resp.text[:300],
        )
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("GitHub issue creation request failed for '%s': %s", project.title, exc)
        return None


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


def save_project_to_github(project, reason: str) -> tuple:
    """
    Dispatches to the configured fallback mode (config.GITHUB_FALLBACK_MODE:
    "issue" or "file"). This is a human-readable permanent record.
    Returns (success: bool, issue_number: Optional[int]) — issue_number is
    only populated in "issue" mode, and is used to link a queue entry to
    its issue (see queue_project) so it can be auto-closed once the project
    is successfully re-evaluated, whichever mechanism resolves it first.
    Never raises — always returns cleanly so a GitHub-side failure can
    never crash the main loop.
    """
    if config.GITHUB_FALLBACK_MODE == "file":
        ok = upload_github_markdown(project, reason)
        return ok, None
    issue_number = create_github_issue(project, reason)
    return issue_number is not None, issue_number


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
        if isinstance(exc, requests.exceptions.RequestException):
            _log_request_exception(f"GET {path}", exc)
        else:
            logger.error("Could not parse GitHub file %s: %s", path, exc)
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
        _log_request_exception(f"PUT {path}", exc)
        return False


def queue_project(project, reason: str, issue_number: Optional[int] = None) -> bool:
    """
    Appends (or refreshes, if already present) a project entry in the
    GitHub-hosted pending-projects queue file. This is what
    load_pending_queue()/main.py's retry loop reads back from. Never
    raises; returns False (and logs) if GitHub fallback isn't configured or
    the write fails — the project is still preserved via
    save_project_to_github()'s Issue/file record either way.

    issue_number (optional): if this project was also saved as a GitHub
    Issue, pass its number here so that when this queue entry is
    successfully re-evaluated, the linked issue can be auto-closed too
    (see main.py's retry_pending_github_queue) — guaranteeing each project
    is only ever actually re-evaluated once, even though both the queue
    and the issue independently reference it.
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
        "client_info": getattr(project, "client_info", None),
        "reason": reason,
        "issue_number": issue_number,
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


# ---------------------------------------------------------------------------
# Open-Issues re-evaluation worker
# ---------------------------------------------------------------------------
# Reads GitHub Issues directly (rather than the queue file above) as an
# independent, explicit "check open issues, parse, re-evaluate, close"
# capability. main.py's retry_open_github_issues() skips any issue number
# already tracked by an active queue entry, so a project referenced by both
# mechanisms is still only ever actually re-evaluated once.
# ---------------------------------------------------------------------------

_ISSUE_URL_RE = re.compile(r"\*\*الرابط:\*\*\s*(\S+)")
_ISSUE_BUDGET_RE = re.compile(r"\*\*الميزانية المعلنة:\*\*\s*(.+)")
_ISSUE_DURATION_RE = re.compile(r"\*\*مدة التسليم المطلوبة:\*\*\s*(.+)")
_ISSUE_TAGS_RE = re.compile(r"\*\*المهارات المطلوبة \(إن توفرت\):\*\*\s*(.+)")
_ISSUE_DESC_RE = re.compile(r"## الوصف الكامل\s*\n+(.*)", re.DOTALL)


def list_open_fallback_issues() -> List[dict]:
    """
    Fetches open GitHub issues labeled FALLBACK_ISSUE_LABEL (i.e. only
    issues this bot itself created via create_github_issue — never touches
    unrelated issues in the repo). Never raises; returns [] on any failure.
    """
    if not config.GITHUB_FALLBACK_ENABLED:
        return []

    url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/issues"
    params = {"state": "open", "labels": FALLBACK_ISSUE_LABEL, "per_page": 100}
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=config.GITHUB_API_TIMEOUT)
        if resp.status_code != 200:
            logger.error("Listing open GitHub issues failed (HTTP %s): %s", resp.status_code, resp.text[:300])
            return []
        issues = resp.json()
        # The Issues endpoint also returns pull requests; filter those out.
        return [i for i in issues if "pull_request" not in i]
    except requests.exceptions.RequestException as exc:
        logger.error("Listing open GitHub issues request failed: %s", exc)
        return []


def parse_issue_body(body: str) -> dict:
    """
    Parses the raw project details back out of an Issue body formatted by
    _format_project_markdown(). Field labels here must match that function
    exactly. Never raises — any field that doesn't match returns None (or
    [] for tags), which callers must treat as 'unknown', not 'empty'.
    """
    body = body or ""

    def _extract(pattern: re.Pattern, placeholder: str = None) -> Optional[str]:
        match = pattern.search(body)
        if not match:
            return None
        value = match.group(1).strip()
        return None if placeholder and value == placeholder else value

    url = _extract(_ISSUE_URL_RE)
    budget = _extract(_ISSUE_BUDGET_RE, placeholder="غير محدد")
    duration = _extract(_ISSUE_DURATION_RE, placeholder="غير محددة")
    tags_raw = _extract(_ISSUE_TAGS_RE, placeholder="غير متوفرة")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    desc_match = _ISSUE_DESC_RE.search(body)
    description = desc_match.group(1).strip() if desc_match else ""

    return {"url": url, "budget": budget, "duration": duration, "tags": tags, "description": description}


def close_issue(issue_number: int, comment: Optional[str] = None) -> bool:
    """
    Closes a GitHub issue (optionally posting a comment first, e.g. the
    re-evaluation result). Never raises; returns False on any failure or if
    issue_number is falsy/not configured."""
    if not config.GITHUB_FALLBACK_ENABLED or not issue_number:
        return False

    base_url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/issues/{issue_number}"
    try:
        if comment:
            requests.post(
                f"{base_url}/comments", json={"body": comment},
                headers=_headers(), timeout=config.GITHUB_API_TIMEOUT,
            )
        resp = requests.patch(
            base_url, json={"state": "closed"},
            headers=_headers(), timeout=config.GITHUB_API_TIMEOUT,
        )
        if resp.status_code == 200:
            logger.info("Closed GitHub issue #%s", issue_number)
            return True
        logger.error("Failed to close GitHub issue #%s (HTTP %s): %s", issue_number, resp.status_code, resp.text[:300])
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Request to close GitHub issue #%s failed: %s", issue_number, exc)
        return False


# ---------------------------------------------------------------------------
# Token-usage stats sync (analytics — completely SEPARATE from the
# fallback-queue mechanism above, and deliberately SILENT per requirement:
# no logger.* calls anywhere in this function, unlike everything else in
# this module. A missing token, network error, or GitHub-side failure here
# must never surface anywhere, not even in logs, and must never raise.)
# ---------------------------------------------------------------------------

def sync_token_stats_to_github() -> bool:
    """
    Reads the local config.TOKEN_USAGE_STATS_FILE and pushes its full
    current content to GitHub via the Contents API
    (PUT /repos/{owner}/{repo}/contents/{path}), fetching the existing
    file's `sha` first (GET) so this correctly UPDATES it in place instead
    of conflicting or creating a duplicate.

    Completely silent and fail-safe by explicit requirement: a missing
    GITHUB_TOKEN, no local file yet, a network error, or a GitHub API
    failure (including rate limiting) all simply return False with zero
    logging and zero exceptions — this must never interrupt the main
    worker loop. The boolean return value is purely a courtesy for a
    caller that wants to check; nothing in this codebase is required to.
    """
    try:
        if not config.GITHUB_FALLBACK_ENABLED:
            return False

        local_path = config.TOKEN_USAGE_STATS_FILE
        if not os.path.exists(local_path):
            return False

        with open(local_path, "r", encoding="utf-8") as f:
            local_content = f.read()

        url = f"{GITHUB_API_BASE}/repos/{config.GITHUB_REPO}/contents/{local_path}"

        # Step 1: fetch current file metadata (if it exists on GitHub yet)
        # to get its sha — required by the Contents API to update rather
        # than reject the write as a conflict.
        sha = None
        try:
            get_resp = requests.get(
                url, headers=_headers(), params={"ref": config.GITHUB_BRANCH},
                timeout=config.GITHUB_API_TIMEOUT,
            )
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")
            # Any other status (404 = doesn't exist yet, or an error) is
            # fine to proceed without a sha — the PUT below will create the
            # file if needed, or simply fail silently if something's wrong.
        except requests.exceptions.RequestException:
            pass

        # Step 2: base64-encode the local file's current content.
        content_b64 = base64.b64encode(local_content.encode("utf-8")).decode("ascii")

        # Step 3: PUT the update.
        payload = {
            "message": "chore: update token usage stats [silent]",
            "content": content_b64,
            "branch": config.GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(url, json=payload, headers=_headers(), timeout=config.GITHUB_API_TIMEOUT)
        return put_resp.status_code in (200, 201)
    except Exception:
        return False
