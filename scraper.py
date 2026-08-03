"""
scraper.py
----------
Monitors Mostaql's public project-listing page for new projects.

Design notes:
- Mostaql does not publish a reliable public RSS feed for the general project
  list, so this module scrapes the public HTML listing page (no login,
  no private data). Because site markup changes over time, ALL CSS selectors
  live in one place (SELECTORS dict below) so you can update them in one spot
  if Mostaql changes its HTML.
- Anti-ban measures: rotating User-Agents, randomized delay between requests,
  a requests.Session with retry/backoff, and short timeouts so a hung
  connection never blocks the worker forever.
- IMPORTANT: Before deploying, check https://mostaql.com/robots.txt and
  Mostaql's Terms of Service to confirm automated polling of this page is
  permitted, and keep your polling interval conservative (this project
  defaults to a 5-10 minute random interval, not rapid-fire requests).
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger("scraper")

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

# Centralized CSS selectors — update here if Mostaql changes its markup.
SELECTORS = {
    "project_card": "div.project-card, li.project-item",   # container per project
    "title": "h2 a, h3 a, .project-title a",
    "description": ".project-description, .project-brief, p",
    "budget": ".budget, .project-budget",
}


@dataclass
class Project:
    id: str
    title: str
    description: str
    url: str
    budget: Optional[str] = None


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)  # we handle retries manually
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _random_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ar,en-US;q=0.8,en;q=0.6",
        "Connection": "keep-alive",
        "Referer": "https://mostaql.com/",
    }


def _polite_delay():
    """Random human-like delay to avoid pattern-based rate limiting."""
    delay = random.uniform(2, 6)
    time.sleep(delay)


def fetch_page(session: requests.Session, url: str) -> Optional[str]:
    """Fetch a page with retries + exponential backoff. Never raises."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            _polite_delay()
            resp = session.get(
                url, headers=_random_headers(), timeout=config.REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                wait = 30 * attempt
                logger.warning("Rate limited (429). Backing off %ss", wait)
                time.sleep(wait)
                continue
            logger.warning("Unexpected status %s on attempt %s", resp.status_code, attempt)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request failed (attempt %s/%s): %s", attempt, config.MAX_RETRIES, exc)
            time.sleep(5 * attempt)  # exponential-ish backoff
    logger.error("All retries exhausted for %s", url)
    return None


def parse_projects(html: str) -> List[Project]:
    """Parse the listing HTML into Project objects. Skips cards it can't parse."""
    projects: List[Project] = []
    if not html:
        return projects

    soup = BeautifulSoup(html, "html.parser")
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

            project_id = url.rstrip("/").split("/")[-1]

            projects.append(
                Project(
                    id=project_id,
                    title=title,
                    description=description,
                    url=url,
                    budget=budget,
                )
            )
        except Exception as exc:  # never let one bad card kill the whole run
            logger.debug("Skipping unparsable card: %s", exc)
            continue

    return projects


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
        session.close()


def project_to_dict(p: Project) -> dict:
    return asdict(p)
