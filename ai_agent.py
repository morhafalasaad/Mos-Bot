"""
ai_agent.py
-----------
Uses Google's new unified Gen AI SDK (`google-genai`, imported as
`from google import genai`) to:
  1. Score how well a project matches the freelancer's skill set (0-100),
     and estimate a suggested bid price and delivery time.
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's specific details.

SDK NOTE: this replaces the old, now end-of-life `google.generativeai`
package with the current `google.genai` package. Per your request, this
uses a single stable model (config.GEMINI_MODEL, default "gemini-3.5-flash")
directly with no fallback chain.

TIMEOUT CAVEAT (important — please read): `google-genai`'s http_options
timeout is known to be unreliable in some versions of the SDK — there are
open upstream bugs where requests can hang indefinitely even with a timeout
configured (e.g. googleapis/python-genai#1893, #911). We still set it
below as a first line of defense, but the actual guarantee that this bot
can never hang forever comes from main.py's CYCLE_TIMEOUT watchdog
(ThreadPoolExecutor + future.result(timeout=...)), which is untouched by
this change. Don't remove that watchdog even if this SDK's own timeout
seems to be working — it's the real safety net.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

import config

logger = logging.getLogger("ai_agent")

# http_options timeout is in MILLISECONDS for this SDK.
_client = genai.Client(
    api_key=config.GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT * 1000),
)


@dataclass
class Evaluation:
    match_score: float
    reasoning: str
    suggested_price: Optional[str] = None
    delivery_days: Optional[int] = None
    proposal_ar: Optional[str] = None


def _generate(prompt: str, json_mode: bool = False):
    """Thin wrapper around client.models.generate_content with the single
    configured model. json_mode=True asks Gemini to emit raw JSON directly
    (in addition to the prompt's own instructions), which reduces — but
    doesn't eliminate — the chance of markdown-fenced output."""
    gen_config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    return _client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=gen_config,
    )


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


def evaluate_project(title: str, description: str, budget: Optional[str] = None) -> Evaluation:
    """
    Full pipeline for one project: score it (including price/duration
    estimates), and if it clears the threshold, draft a proposal too.
    Always returns an Evaluation object — never raises — so main.py's loop
    can rely on it unconditionally.
    """
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
