"""
ai_agent.py
-----------
Uses Google's Gen AI SDK (`from google import genai`) to:
  1. Score how well a project matches the freelancer's skill set (0-100),
     and estimate a suggested bid price and delivery time.
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's specific details.

Two cost-control features on top of that:

LOCAL TAG PRE-FILTERING (zero API cost for irrelevant projects)
-------------------------------------------------------------------
Before any Gemini call is made, `local_skill_prefilter()` compares the
project's official Mostaql skill tags (scraped from its detail page —
see scraper.fetch_project_tags) against config.MY_SKILLS. If there's no
overlap at all, `evaluate_project()` returns immediately with
match_score=0.0 and makes ZERO Gemini API calls for that project.

IMPORTANT — fail-open by design: if tags weren't fetched (empty list —
either FETCH_PROJECT_TAGS is off, or the detail-page fetch/parse failed),
the pre-filter does NOT block the project; it falls through to the normal
Gemini evaluation. We would rather spend an API call on an uncertain
project than silently drop a good one because of a scraping gap.

API KEY ROTATION (survive per-key free-tier quota limits)
-------------------------------------------------------------------
`config.GEMINI_API_KEYS` is a list. `_generate()` tries the current key;
if the call fails with a 429 / RESOURCE_EXHAUSTED quota error, it rotates
to the next key in the list, rebuilds the client, and retries — up to once
per configured key. Any other kind of error (bad prompt, network issue,
etc.) is NOT treated as a rotation trigger and is raised immediately, so we
don't burn through every key on an unrelated failure.

SDK NOTE: single stable model (config.GEMINI_MODEL, default
"gemini-3.5-flash"), no model fallback chain, per current requirements.

TIMEOUT CAVEAT: google-genai's http_options timeout has known upstream
reliability issues (requests can occasionally hang despite a timeout being
set — see googleapis/python-genai#1893, #911). We still set it below as a
first line of defense, but the real guarantee against a permanent hang is
main.py's CYCLE_TIMEOUT watchdog (ThreadPoolExecutor + future.result
timeout), which is untouched by this file and must stay in place.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from google.genai import types

import config

logger = logging.getLogger("ai_agent")


# ---------------------------------------------------------------------------
# Client + API key rotation
# ---------------------------------------------------------------------------
# Module-level state is safe without locking because main.py runs cycles one
# at a time on a single worker thread (ThreadPoolExecutor(max_workers=1)).
_current_key_index = 0


def _build_client(key_index: int) -> genai.Client:
    return genai.Client(
        api_key=config.GEMINI_API_KEYS[key_index],
        # http_options timeout is in MILLISECONDS for this SDK.
        http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT * 1000),
    )


_client = _build_client(_current_key_index)


def _is_quota_error(exc: Exception) -> bool:
    """Detects a 429 / RESOURCE_EXHAUSTED quota error across SDK versions,
    since relying on a single exception type/attribute is fragile."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


@dataclass
class Evaluation:
    match_score: float
    reasoning: str
    suggested_price: Optional[str] = None
    delivery_days: Optional[int] = None
    proposal_ar: Optional[str] = None


def _generate(prompt: str, json_mode: bool = False):
    """
    Calls the currently active Gemini API key/client. On a quota error
    (429 RESOURCE_EXHAUSTED), rotates to the next key in
    config.GEMINI_API_KEYS, rebuilds the client, and retries — up to once
    per configured key. Any non-quota error is raised immediately without
    rotating. Raises the last exception if every key is exhausted.
    """
    global _client, _current_key_index

    gen_config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    n = len(config.GEMINI_API_KEYS)

    last_exc: Optional[Exception] = None
    for attempt in range(n):
        try:
            return _client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config,
            )
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and n > 1:
                _current_key_index = (_current_key_index + 1) % n
                logger.warning(
                    "Gemini API key #%s hit quota (429/RESOURCE_EXHAUSTED) — "
                    "rotating to key #%s of %s and retrying",
                    attempt + 1, _current_key_index + 1, n,
                )
                _client = _build_client(_current_key_index)
                continue
            raise

    logger.error("All %s Gemini API key(s) exhausted their quota. Last error: %s", n, last_exc)
    raise last_exc if last_exc else RuntimeError("No Gemini API keys available")


def _extract_json(text: str) -> Optional[dict]:
    """Gemini sometimes wraps JSON in ```json fences — strip and parse safely."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.error("Could not parse JSON from Gemini response: %s", text[:300])
    return None


# ---------------------------------------------------------------------------
# Local tag pre-filter (zero API cost)
# ---------------------------------------------------------------------------

def _skill_tokens(skill: str) -> List[str]:
    """Expands a skill entry into extra matchable tokens, e.g.
    'Object-Oriented Programming (OOP)' -> also match plain 'OOP'."""
    tokens = [skill.strip()]
    paren_match = re.search(r"\(([^)]+)\)", skill)
    if paren_match:
        tokens.append(paren_match.group(1).strip())
    stripped = re.sub(r"\([^)]*\)", "", skill).strip()
    if stripped and stripped not in tokens:
        tokens.append(stripped)
    return [t for t in tokens if len(t) >= 2]


def local_skill_prefilter(tags: List[str]) -> bool:
    """
    Returns True if the project should proceed to Gemini evaluation, False
    if it should be skipped locally with zero API cost.

    Fail-open: an empty/missing tags list means "unknown, not confirmed
    irrelevant" and always returns True. Only an explicit, non-empty tag
    list with NO overlap against config.MY_SKILLS returns False.
    """
    if not tags:
        return True

    tag_texts = [t.lower() for t in tags if t]
    if not tag_texts:
        return True

    for skill in config.MY_SKILLS:
        for token in _skill_tokens(skill):
            token_l = token.lower()
            for tag in tag_texts:
                if token_l in tag or tag in token_l:
                    return True

    return False


def score_project(title: str, description: str) -> Optional[dict]:
    """
    Step 1: ask Gemini for a match score, reasoning, a suggested bid price,
    and an estimated delivery time. Returns None on failure.
    """
    skills_list = ", ".join(config.MY_SKILLS)

    prompt = f"""
You are an expert freelance-bidding assistant. Compare the project below
against the freelancer's skill set, estimate how good a match it is, and
recommend a realistic bid.

Freelancer skills: {skills_list}

Project title: {title}
Project description: {description}

Respond with ONLY a raw JSON object (no markdown, no extra text) in this
exact shape:
{{
  "match_score": <integer 0-100>,
  "reasoning": "<one short sentence in English explaining the score>",
  "suggested_price": "<a realistic recommended bid price/budget for this
                        project's scope, as a short string including
                        currency, e.g. '$150' or '$300-400'>",
  "delivery_days": <integer, realistic number of days to complete the
                     project based on its scope>
}}
"""
    try:
        response = _generate(prompt, json_mode=True)
        data = _extract_json(response.text)
        if data is None or "match_score" not in data:
            return None
        return data
    except Exception as exc:
        logger.error("Gemini scoring call failed: %s", exc, exc_info=True)
        return None


def draft_proposal(title: str, description: str, budget: Optional[str] = None) -> Optional[str]:
    """Step 2 (only called if score > threshold): draft an Arabic proposal."""
    skills_list = ", ".join(config.MY_SKILLS)
    budget_line = f"\nProject budget: {budget}" if budget else ""

    prompt = f"""
أنت مساعد كتابة عروض احترافي لمستقل يعمل على منصة مستقل (Mostaql).
مهارات المستقل: {skills_list}

عنوان المشروع: {title}
وصف المشروع: {description}{budget_line}

اكتب عرضاً احترافياً ومقنعاً باللغة العربية الفصحى لتقديمه على هذا المشروع، بحيث:
- يبدأ بجملة تُظهر فهماً دقيقاً لاحتياج العميل المحدد في وصف المشروع.
- يبرز الخبرات ذات الصلة المباشرة بالمشروع فقط من بين مهارات المستقل.
- يقترح خطوات عمل أو منهجية مختصرة لتنفيذ المشروع.
- يكون بأسلوب واثق ومهني، دون مبالغة أو عبارات عامة جاهزة.
- طوله لا يتجاوز 150 كلمة.
- لا تضع أي عناوين أو تنسيق ماركداون، فقط نص العرض جاهزاً للنسخ.
"""
    try:
        response = _generate(prompt)
        text = response.text.strip()
        return text if text else None
    except Exception as exc:
        logger.error("Gemini proposal drafting failed: %s", exc, exc_info=True)
        return None


def evaluate_project(
    title: str,
    description: str,
    budget: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Evaluation:
    """
    Full pipeline for one project:
      0. Local tag pre-filter — zero-cost skip if tags exist and don't
         overlap with config.MY_SKILLS at all.
      1. Score it via Gemini (including price/duration estimates).
      2. If it clears the threshold, draft a proposal too.
    Always returns an Evaluation object — never raises — so main.py's loop
    can rely on it unconditionally.
    """
    if tags and not local_skill_prefilter(tags):
        logger.info(
            "Local pre-filter: no overlap between project tags %s and "
            "MY_SKILLS — skipping Gemini entirely (zero API cost)",
            tags,
        )
        return Evaluation(
            match_score=0.0,
            reasoning="No matching skill tags (filtered locally, zero API cost).",
        )

    score_data = score_project(title, description)
    if score_data is None:
        return Evaluation(match_score=0.0, reasoning="AI scoring unavailable (error).")

    score = float(score_data.get("match_score", 0))
    reasoning = score_data.get("reasoning", "")
    suggested_price = score_data.get("suggested_price")
    delivery_days = score_data.get("delivery_days")
    try:
        delivery_days = int(delivery_days) if delivery_days is not None else None
    except (TypeError, ValueError):
        delivery_days = None

    proposal = None
    if score > config.MATCH_THRESHOLD:
        proposal = draft_proposal(title, description, budget)

    return Evaluation(
        match_score=score,
        reasoning=reasoning,
        suggested_price=suggested_price,
        delivery_days=delivery_days,
        proposal_ar=proposal,
    )
