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
import logging
from datetime import datetime, timezone

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
    "issue" or "file"). This is the single entry point main.py should call.
    Never raises — always returns a bool so the caller can log the outcome
    but a GitHub-side failure can never crash the main loop.
    """
    if config.GITHUB_FALLBACK_MODE == "file":
        return upload_github_markdown(project, reason)
    return create_github_issue(project, reason)
