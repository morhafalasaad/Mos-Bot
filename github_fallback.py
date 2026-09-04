"""
github_fallback.py
-------------------
OPTIONAL, SECONDARY fallback for when the Gemini AI evaluation call
itself fails — most commonly 429 RESOURCE_EXHAUSTED after every key in
GEMINI_API_KEYS has been tried and exhausted (see ai_agent._generate()).

As of the Sept 2026 MongoDB migration, this module is NO LONGER the
retry-queue-of-record — that's MongoDB's `pending_projects` collection
now (see db.py / main.py's retry_pending_queue()), which is what
main.py actually reads back from each cycle. What's left here is purely
an OPTIONAL human-readable audit trail: when configured (GITHUB_TOKEN +
GITHUB_REPO both set), a raw project that Gemini couldn't evaluate is
also posted to GitHub as an Issue or an uploaded Markdown file, so a
human has something to read/search/react to without needing direct
MongoDB access. If GitHub isn't configured, the bot's actual retry
mechanism is entirely unaffected — only this secondary human notice is
skipped (logged as a warning).

Uses only `requests` against the GitHub REST API — no PyGithub/octokit
dependency needed for the two simple calls this now makes (create an
Issue / PUT a file).
"""

import base64
import logging
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
# Open-Issues re-evaluation worker
# ---------------------------------------------------------------------------
# Reads GitHub Issues directly as an independent, explicit "check open
# issues, parse, re-evaluate, close" capability — a safety net for issues
# whose linked MongoDB queue entry is missing for any reason. main.py's
# retry_open_github_issues() skips any issue number already tracked by an
# active MongoDB queue entry (see db.get_pending_projects), so a project
# referenced by both mechanisms is still only ever actually re-evaluated
# once.
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

