"""
scraper.py
----------
Monitors Mostaql's public project-listing page for new projects.

WHY THE PREVIOUS VERSION RETURNED 0 PROJECTS EVERY CYCLE
----------------------------------------------------------
Two likely causes, both addressed below:

1. The CSS selectors (`div.project-card`, `.project-title a`, etc.) were
   reasonable guesses but did not match Mostaql's actual markup, so
   `soup.select(...)` silently matched zero elements every time — no error,
   just an empty list. Fixed by switching the PRIMARY parsing strategy to
   something structural and far less likely to break: Mostaql's project
   detail links always follow the pattern `/project/<numeric_id>-<slug>`.
   We find every anchor matching that pattern instead of depending on
   specific class names, and derive the title/description from the anchors
   themselves. The old class-based selectors are kept as a secondary
   fallback attempt in case Mostaql's markup includes them after all.
2. Possible Cloudflare / bot-protection blocking of Render's IP ranges
   (403, 503, or a JS-challenge "Just a moment..." interstitial page
   instead of real content). Fixed by trying `cloudscraper` first (which
   solves basic Cloudflare JS/IUAM challenges automatically) and falling
   back to plain `requests` with realistic browser headers if
   `cloudscraper` isn't installed. Debug logging now prints the exact
   status code and flags known block/challenge signatures so this is
   directly visible in the Render logs instead of silently returning "0
   new projects".

IMPORTANT: Before deploying, check https://mostaql.com/robots.txt and
Mostaql's Terms of Service to confirm automated polling of this page is
permitted, and keep your polling interval conservative.
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Optional cloudscraper support. cloudscraper wraps requests and automatically
# solves Cloudflare's basic JS/IUAM ("I'm Under Attack Mode") challenges,
# which plain `requests` cannot do no matter how good the headers are. If
# it's not installed, we transparently fall back to plain requests.
# ---------------------------------------------------------------------------
try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False
    logger.warning(
        "cloudscraper is not installed — falling back to plain requests. "
        "If Mostaql is Cloudflare-protected, add 'cloudscraper' to "
        "requirements.txt for a much higher success rate."
    )

# Rotate between a handful of realistic desktop User-Agent strings.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

# Legacy/fallback CSS selectors — tried first in case Mostaql's markup uses
# them; if they match nothing, parse_projects() falls back to the
# link-pattern strategy below, which is the primary, more robust method.
SELECTORS = {
    "project_card": "div.project-card, li.project-item, article.project",
    "title": "h2 a, h3 a, .project-title a",
    "description": ".project-description, .project-brief, p",
    "budget": ".budget, .project-budget",
}

# Mostaql project detail URLs always look like:
#   https://mostaql.com/project/1265468-<arabic-or-latin-slug>
# "similar project" links look like /project/create?template=1265468 and are
# deliberately excluded by requiring a "-" right after the numeric id.
PROJECT_LINK_RE = re.compile(r"^/project/(\d+)-")

# Signatures that indicate a block / bot-challenge page rather than the
# real listing, so we can log it clearly instead of just "0 projects".
BLOCK_SIGNATURES = [
    "just a moment",            # Cloudflare JS challenge interstitial
    "attention required",       # Cloudflare block page
    "cf-browser-verification",
    "checking your browser",
    "captcha",
    "access denied",
    "sorry, you have been blocked",
    "ddos protection by",
]


@dataclass
class Project:
    id: str
    title: str
    description: str
    url: str
    budget: Optional[str] = None


def _build_session():
    """
    Returns a cloudscraper session if available (handles Cloudflare JS
    challenges), otherwise a plain requests.Session with browser-like
    defaults. Either way the returned object supports .get() and .close()
    like a normal requests.Session.
    """
    if _HAS_CLOUDSCRAPER:
        logger.info("Using cloudscraper session (Cloudflare-aware)")
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        return scraper

    logger.info("Using plain requests.Session (cloudscraper not installed)")
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)  # we handle retries manually
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _random_headers() -> dict:
    """Realistic full browser header set — helps with basic bot heuristics
    even though it cannot solve a real Cloudflare JS challenge on its own
    (that's what cloudscraper is for)."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ar,en-US;q=0.8,en;q=0.6",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://mostaql.com/",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
    }


def _polite_delay():
    """Random human-like delay to avoid pattern-based rate limiting."""
    time.sleep(random.uniform(2, 6))


def _detect_block(status_code: int, html: str) -> Optional[str]:
    """Returns a human-readable reason string if the response looks like a
    block/challenge page rather than real content, else None."""
    if status_code in (403, 503):
        return f"HTTP {status_code} (commonly returned by Cloudflare/WAF blocks)"

    lowered = (html or "").lower()
    for signature in BLOCK_SIGNATURES:
        if signature in lowered:
            return f"page content matched block signature: '{signature}'"

    return None


def fetch_page(session, url: str) -> Optional[str]:
    """Fetch a page with retries + exponential backoff. Never raises.
    Logs the HTTP status code and any detected block/challenge every time,
    so the exact failure mode is always visible in the Render logs."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            _polite_delay()
            resp = session.get(
                url,
                headers=_random_headers(),
                timeout=(10, config.REQUEST_TIMEOUT),
            )

            logger.info(
                "GET %s -> status %s (attempt %s/%s, %s bytes)",
                url, resp.status_code, attempt, config.MAX_RETRIES, len(resp.content or b""),
            )

            block_reason = _detect_block(resp.status_code, resp.text)
            if block_reason:
                logger.warning(
                    "Response looks BLOCKED, not a real listing page: %s. "
                    "First 300 chars: %r",
                    block_reason, resp.text[:300] if resp.text else "",
                )
                # A block is still worth retrying (Cloudflare challenges can
                # be intermittent) but there's no point hammering it fast.
                if attempt < config.MAX_RETRIES:
                    time.sleep(15 * attempt)
                continue

            if resp.status_code == 200:
                return resp.text

            if resp.status_code == 429:
                wait = 30 * attempt
                logger.warning("Rate limited (429). Backing off %ss", wait)
                time.sleep(wait)
                continue

            logger.warning("Unexpected status %s on attempt %s", resp.status_code, attempt)

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed (attempt %s/%s): %s: %s",
                attempt, config.MAX_RETRIES, type(exc).__name__, exc,
            )
            time.sleep(5 * attempt)  # exponential-ish backoff
        except Exception as exc:
            # Catch-all so a truly unexpected error (e.g. a bug in a
            # dependency) still can't escape and kill the calling thread.
            logger.error(
                "Unexpected error fetching %s (attempt %s/%s): %s",
                url, attempt, config.MAX_RETRIES, exc, exc_info=True,
            )
            time.sleep(5 * attempt)

    logger.error("All retries exhausted for %s — treat this cycle as a failed fetch", url)
    return None


def _parse_via_css_selectors(soup: BeautifulSoup) -> List[Project]:
    """Legacy/fallback strategy — only used if it actually matches something."""
    projects: List[Project] = []
    cards = soup.select(SELECTORS["project_card"])
    for card in cards:
        try:
            title_el = card.select_one(SELECTORS["title"])
            if not title_el or not title_el.get("href"):
                continue

            url = title_el["href"]
            if url.startswith("/"):
                url = "https://mostaql.com" + url

            title = title_el.get_text(strip=True)
            desc_el = card.select_one(SELECTORS["description"])
            description = desc_el.get_text(strip=True) if desc_el else ""
            budget_el = card.select_one(SELECTORS["budget"])
            budget = budget_el.get_text(strip=True) if budget_el else None

            match = PROJECT_LINK_RE.match(title_el["href"]) if title_el["href"].startswith("/") else None
            project_id = match.group(1) if match else url.rstrip("/").split("/")[-1]

            projects.append(Project(id=project_id, title=title, description=description, url=url, budget=budget))
        except Exception as exc:
            logger.debug("Skipping unparsable card (CSS strategy): %s", exc)
            continue
    return projects


def _parse_via_link_pattern(soup: BeautifulSoup) -> List[Project]:
    """
    PRIMARY strategy. Finds every anchor whose href matches
    /project/<id>-<slug> (Mostaql's stable project-detail URL pattern)
    instead of relying on CSS class names that can change at any time.

    Mostaql typically renders both the project title AND its description
    excerpt as separate <a> tags pointing to the same project URL. When we
    see multiple anchors for the same project id, we treat the SHORTEST
    text as the title and the LONGEST as the description (the excerpt is
    always longer than the title in practice).
    """
    groups: dict = {}  # project_id -> {"url": ..., "texts": [str, ...]}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        match = PROJECT_LINK_RE.match(href)
        if not match:
            continue

        project_id = match.group(1)
        text = a.get_text(strip=True)
        if not text:
            continue

        full_url = "https://mostaql.com" + href if href.startswith("/") else href
        entry = groups.setdefault(project_id, {"url": full_url, "texts": []})
        entry["texts"].append(text)

    projects: List[Project] = []
    for project_id, data in groups.items():
        texts = sorted(set(data["texts"]), key=len)
        title = texts[0]
        description = texts[-1] if len(texts) > 1 else texts[0]
        projects.append(
            Project(id=project_id, title=title, description=description, url=data["url"], budget=None)
        )

    return projects


def parse_projects(html: str) -> List[Project]:
    """Parse the listing HTML into Project objects using the CSS-selector
    strategy first, falling back to the link-pattern strategy if that
    yields nothing. Logs which strategy succeeded and, if BOTH fail, dumps
    a snippet of the HTML so the real cause is visible in the logs instead
    of a bare '0 new projects'."""
    if not html:
        logger.warning("parse_projects called with empty HTML (fetch must have failed upstream)")
        return []

    soup = BeautifulSoup(html, "html.parser")

    css_results = _parse_via_css_selectors(soup)
    if css_results:
        logger.info("Parsed %s project(s) via CSS-selector strategy", len(css_results))
        return css_results

    link_results = _parse_via_link_pattern(soup)
    if link_results:
        logger.info("Parsed %s project(s) via link-pattern strategy", len(link_results))
        return link_results

    # Both strategies found nothing — this is the case that used to
    # silently log "0 new projects" with no way to diagnose why. Now we
    # dump enough context to tell block-page vs. genuinely-changed-markup
    # vs. genuinely-empty-listing apart.
    any_project_style_links = bool(re.search(r"/project/\d+-", html))
    logger.warning(
        "Both parsing strategies found 0 projects. "
        "Any '/project/<id>-' links present in raw HTML at all? %s. "
        "HTML length: %s chars. First 500 chars: %r",
        any_project_style_links, len(html), html[:500],
    )
    return []


# ---------------------------------------------------------------------------
# Deduplication so we only notify about genuinely NEW projects each cycle.
# ---------------------------------------------------------------------------

def _load_seen() -> set:
    if not os.path.exists(config.SEEN_PROJECTS_FILE):
        return set()
    try:
        with open(config.SEEN_PROJECTS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Could not read seen-projects file, starting fresh: %s", exc)
        return set()


def _save_seen(seen: set):
    try:
        # Cap file size: keep only the most recent 500 IDs.
        trimmed = list(seen)[-500:]
        with open(config.SEEN_PROJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f)
    except IOError as exc:
        logger.error("Could not persist seen-projects file: %s", exc)


def get_new_projects() -> List[Project]:
    """
    Main entry point for main.py. Fetches the listing, parses it, and returns
    only projects not seen in previous cycles. Fully defensive: returns an
    empty list rather than raising, so the worker loop never crashes here.
    """
    session = _build_session()
    try:
        html = fetch_page(session, config.MOSTAQL_PROJECTS_URL)
        all_projects = parse_projects(html)
        logger.info("Total projects found on page this cycle: %s", len(all_projects))

        seen = _load_seen()
        new_projects = [p for p in all_projects if p.id not in seen]

        if new_projects:
            seen.update(p.id for p in new_projects)
            _save_seen(seen)
            logger.info("Found %s new project(s)", len(new_projects))
        else:
            logger.info("No new projects this cycle")

        return new_projects
    except Exception as exc:
        # Absolute last line of defense for this module.
        logger.exception("Unexpected error in get_new_projects: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def project_to_dict(p: Project) -> dict:
    return asdict(p)
