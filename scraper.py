"""
scraper.py
----------
Monitors Mostaql's public project-listing page for new projects.

WHY THIS KEPT FAILING (root causes, in the order we found them)
-------------------------------------------------------------------
1. Guessed CSS classes (`div.project-card`, etc.) never matched Mostaql's
   real markup — silently returned 0 elements, no error.
2. The link-pattern regex was anchored (`^/project/...`), so it only
   matched RELATIVE hrefs. Mostaql actually renders ABSOLUTE hrefs
   (`https://mostaql.com/project/...`), so it also silently matched 0.
3. Even after fixing the regex, extraction still failed. The cause: this
   module was parsing HTML with `BeautifulSoup(html, "html.parser")`.
   Python's built-in `html.parser` is strict and can silently mis-nest or
   drop tags when it encounters real-world malformed HTML (unclosed tags,
   stray attributes, etc.) — so `soup.find_all("a", href=True)` can miss
   anchors that a plain string search would still find. That's exactly why
   the diagnostic log showed "links present in raw HTML: True" while
   BeautifulSoup extraction returned 0: the *string* contains the links,
   but the *parsed tree* BeautifulSoup built from them was broken.

THE FIX — three independent extraction strategies, most-robust-first
-------------------------------------------------------------------
1. CSS-selector strategy (legacy) — only used if it happens to match.
2. BeautifulSoup link-pattern strategy — now parsed with `lxml` if
   available (a lenient, industry-standard HTML parser that handles
   malformed markup far better than `html.parser`), with automatic
   fallback to `html.parser` if `lxml` isn't installed.
3. RAW REGEX fallback strategy (new, and now the guaranteed last resort) —
   operates directly on the raw HTML string with a regex, completely
   bypassing BeautifulSoup's tree-building. This cannot be defeated by
   malformed markup, unknown class names, or DOM-nesting quirks: it finds
   every `<a ... href="...project/<id>-...">...</a>` occurrence directly
   in the text. Title + link are ALWAYS captured this way; description is
   best-effort (nearby anchor text); budget is optional (only populated by
   strategy 1, since it has no unambiguous text signature to regex for).

IMPORTANT: Before deploying, check https://mostaql.com/robots.txt and
Mostaql's Terms of Service to confirm automated polling of this page is
permitted, and keep your polling interval conservative.
"""

import html as html_module
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
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

# ---------------------------------------------------------------------------
# Optional lxml support. lxml is a much more forgiving HTML parser than the
# stdlib html.parser and handles real-world malformed markup correctly —
# this was the actual root cause of extraction silently returning 0 results
# even though the target links were confirmed present in the raw HTML.
# ---------------------------------------------------------------------------
try:
    import lxml  # noqa: F401
    _BS4_PARSER = "lxml"
except ImportError:
    _BS4_PARSER = "html.parser"
    logger.warning(
        "lxml is not installed — falling back to html.parser, which is "
        "stricter and can mis-parse malformed real-world HTML. Add 'lxml' "
        "to requirements.txt for more reliable extraction."
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
# them; if they match nothing, parse_projects() falls back to the next
# strategy. Kept purely as a bonus for budget extraction if it ever matches.
SELECTORS = {
    "project_card": "div.project-card, li.project-item, article.project",
    "title": "h2 a, h3 a, .project-title a",
    "description": ".project-description, .project-brief, p",
    "budget": ".budget, .project-budget",
}

# Mostaql project detail URLs always look like:
#   https://mostaql.com/project/1265468-<arabic-or-latin-slug>
# Unanchored so it matches both absolute and relative hrefs. "similar
# project" links look like /project/create?template=1265468 and are
# deliberately excluded by requiring a "-" right after the numeric id.
PROJECT_LINK_RE = re.compile(r"/project/(\d+)-")

# Raw-regex fallback: matches a full <a ...href="...">inner html</a> block
# directly against the HTML string, independent of BeautifulSoup entirely.
# Non-greedy + DOTALL so it correctly spans anchors whose inner content
# includes nested tags or newlines.
RAW_ANCHOR_RE = re.compile(
    r'<a\b[^>]*?href=["\']([^"\']*?/project/\d+-[^"\']*)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

# Mostaql skill/tag pages follow /projects/skill/<slug> — project detail
# pages link each required-skill tag ("المهارات المطلوبة") to its own
# archive page using this exact pattern, so it's used the same way
# PROJECT_LINK_RE is: to find tags independent of any CSS class name.
TAG_LINK_RE = re.compile(r"/projects/skill/([^\"'/?#]+)")

# Raw-regex fallback for tags, mirroring RAW_ANCHOR_RE's approach.
RAW_TAG_ANCHOR_RE = re.compile(
    r'<a\b[^>]*?href=["\']([^"\']*?/projects/skill/[^"\']*)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
# real listing, so we can log it clearly instead of just "0 projects".
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
    # Official "المهارات المطلوبة" (required skills) tags from the project's
    # own detail page. Only the LISTING page is scraped by default — tags
    # live on each project's individual page, so this stays empty until
    # fetch_project_tags() is called for that project (see get_new_projects).
    tags: List[str] = field(default_factory=list)


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
            time.sleep(5 * attempt)
        except Exception as exc:
            logger.error(
                "Unexpected error fetching %s (attempt %s/%s): %s",
                url, attempt, config.MAX_RETRIES, exc, exc_info=True,
            )
            time.sleep(5 * attempt)

    logger.error("All retries exhausted for %s — treat this cycle as a failed fetch", url)
    return None


def _make_soup(html: str) -> BeautifulSoup:
    """Builds a BeautifulSoup tree using lxml if available (much more
    forgiving of malformed real-world HTML), falling back to html.parser."""
    try:
        return BeautifulSoup(html, _BS4_PARSER)
    except Exception as exc:
        if _BS4_PARSER != "html.parser":
            logger.warning("lxml parsing failed (%s), retrying with html.parser", exc)
            return BeautifulSoup(html, "html.parser")
        raise


def _parse_via_css_selectors(soup: BeautifulSoup) -> List[Project]:
    """Legacy/fallback strategy — only used if it actually matches something.
    This is currently the ONLY strategy that can populate `budget`."""
    projects: List[Project] = []
    cards = soup.select(SELECTORS["project_card"])
    for card in cards:
        try:
            title_el = card.select_one(SELECTORS["title"])
            if not title_el or not title_el.get("href"):
                continue

            href = title_el["href"]
            url = "https://mostaql.com" + href if href.startswith("/") else href

            title = title_el.get_text(strip=True)
            desc_el = card.select_one(SELECTORS["description"])
            description = desc_el.get_text(strip=True) if desc_el else ""
            budget_el = card.select_one(SELECTORS["budget"])
            budget = budget_el.get_text(strip=True) if budget_el else None

            match = PROJECT_LINK_RE.search(href)
            project_id = match.group(1) if match else url.rstrip("/").split("/")[-1]

            projects.append(Project(id=project_id, title=title, description=description, url=url, budget=budget))
        except Exception as exc:
            logger.debug("Skipping unparsable card (CSS strategy): %s", exc)
            continue
    return projects


def _parse_via_link_pattern(soup: BeautifulSoup) -> List[Project]:
    """
    BeautifulSoup-based strategy. Finds every anchor whose href matches
    /project/<id>-<slug> (Mostaql's stable project-detail URL pattern)
    instead of relying on CSS class names that can change at any time.

    When multiple anchors point to the same project id (Mostaql typically
    renders both the title and a description excerpt as separate <a> tags
    to the same URL), the SHORTEST text is treated as the title and the
    LONGEST as the description.
    """
    groups: dict = {}  # project_id -> {"url": ..., "texts": [str, ...]}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        match = PROJECT_LINK_RE.search(href)
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


def _strip_inner_tags(inner_html: str) -> str:
    """Strips any nested HTML tags out of an anchor's inner content and
    decodes HTML entities, for use by the raw-regex fallback strategy."""
    text = re.sub(r"<[^>]+>", " ", inner_html)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_via_raw_regex(html: str) -> List[Project]:
    """
    LAST-RESORT, GUARANTEED strategy. Operates directly on the raw HTML
    string with a regex — completely bypasses BeautifulSoup's tree-building,
    so it cannot be defeated by malformed markup or DOM-nesting quirks that
    make html.parser (or even lxml, in rare cases) miss real anchors.

    Title and link are always captured here as long as the anchor exists in
    the raw text at all. Description is best-effort (longest distinct anchor
    text seen for that project id). Budget is not populated by this
    strategy — it has no reliable, class-independent text signature to
    regex for; only the CSS-selector strategy can supply it.
    """
    groups: dict = {}  # project_id -> {"url": ..., "texts": [str, ...]}

    for href, inner_html in RAW_ANCHOR_RE.findall(html):
        match = PROJECT_LINK_RE.search(href)
        if not match:
            continue

        project_id = match.group(1)
        text = _strip_inner_tags(inner_html)
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


def _parse_tags_via_bs4(soup: BeautifulSoup) -> List[str]:
    """Finds every anchor linking to /projects/skill/<slug> on a project's
    detail page — these are the official required-skill tags. Order-preserving,
    de-duplicated."""
    tags: List[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        if not TAG_LINK_RE.search(a["href"]):
            continue
        text = a.get_text(strip=True)
        if text and text not in seen:
            seen.add(text)
            tags.append(text)
    return tags


def _parse_tags_via_raw_regex(html: str) -> List[str]:
    """Guaranteed fallback — same rationale as _parse_via_raw_regex above:
    bypasses BeautifulSoup's tree-building entirely."""
    tags: List[str] = []
    seen = set()
    for _href, inner_html in RAW_TAG_ANCHOR_RE.findall(html):
        text = _strip_inner_tags(inner_html)
        if text and text not in seen:
            seen.add(text)
            tags.append(text)
    return tags


def parse_project_tags(html: str) -> List[str]:
    """Parses a project DETAIL page's HTML for its official required-skill
    tags. Tries BeautifulSoup first, falls back to raw regex. Returns an
    empty list (never raises) if none are found — callers must treat an
    empty list as 'unknown', not 'no skills required', since local
    pre-filtering fails OPEN (doesn't block) when tags are unavailable."""
    if not html:
        return []
    soup = _make_soup(html)
    tags = _parse_tags_via_bs4(soup)
    if tags:
        return tags
    return _parse_tags_via_raw_regex(html)


def fetch_project_tags(session, project: Project) -> List[str]:
    """
    Fetches a single project's own detail page and extracts its required-
    skill tags. This is an EXTRA Mostaql request per newly-seen project
    (gated by config.FETCH_PROJECT_TAGS), separate from the one listing-page
    request per cycle. The trade-off is intentional: it only runs for
    projects that already passed dedup and would otherwise cost a Gemini
    API call — spending one cheap Mostaql request to potentially save a
    Gemini call is the whole point of local pre-filtering.

    Never raises: returns [] on any failure (block, timeout, parse miss),
    which is the correct "unknown" signal for the fail-open pre-filter.
    """
    try:
        html = fetch_page(session, project.url)
        tags = parse_project_tags(html)
        logger.info("Fetched %s tag(s) for project '%s': %s", len(tags), project.title, tags)
        return tags
    except Exception as exc:
        logger.warning("Could not fetch/parse tags for %s: %s", project.url, exc)
        return []


def parse_projects(html: str) -> List[Project]:
    """
    Parse the listing HTML into Project objects, trying three strategies in
    order of preference (richest data first, most-robust-and-guaranteed
    last):
      1. CSS-selector strategy   — only useful if class names happen to match;
                                    the only one that can populate `budget`.
      2. BeautifulSoup link-pattern strategy (lxml-backed if available).
      3. Raw-regex fallback strategy — bypasses BeautifulSoup entirely;
         title + link are ALWAYS captured here if the link exists in the
         HTML string at all.
    Each strategy's result count is logged so it's obvious which one
    succeeded. If ALL THREE fail, dumps enough HTML context to diagnose why.
    """
    if not html:
        logger.warning("parse_projects called with empty HTML (fetch must have failed upstream)")
        return []

    soup = _make_soup(html)

    css_results = _parse_via_css_selectors(soup)
    if css_results:
        logger.info("Parsed %s project(s) via CSS-selector strategy", len(css_results))
        return css_results

    link_results = _parse_via_link_pattern(soup)
    if link_results:
        logger.info(
            "Parsed %s project(s) via BeautifulSoup link-pattern strategy (parser=%s)",
            len(link_results), _BS4_PARSER,
        )
        return link_results

    regex_results = _parse_via_raw_regex(html)
    if regex_results:
        logger.info(
            "Parsed %s project(s) via raw-regex fallback strategy "
            "(BeautifulSoup missed them — likely malformed markup)",
            len(regex_results),
        )
        return regex_results

    # All three strategies found nothing.
    any_project_style_links = bool(re.search(r"/project/\d+-", html))
    logger.warning(
        "ALL THREE parsing strategies found 0 projects. "
        "Any '/project/<id>-' links present in raw HTML at all? %s. "
        "HTML length: %s chars. First 800 chars: %r",
        any_project_style_links, len(html), html[:800],
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

            if config.FETCH_PROJECT_TAGS:
                for project in new_projects:
                    project.tags = fetch_project_tags(session, project)
            else:
                logger.info("FETCH_PROJECT_TAGS is disabled — skipping per-project tag fetch")
        else:
            logger.info("No new projects this cycle")

        return new_projects
    except Exception as exc:
        logger.exception("Unexpected error in get_new_projects: %s", exc)
        return []
    finally:
        try:
            session.close()
        except Exception:
            pass


def project_to_dict(p: Project) -> dict:
    return asdict(p)
