"""
ai_agent.py
-----------
Uses Google Gemini (google-generativeai) to:
  1. Score how well a project matches the freelancer's skill set (0-100).
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's specific details.

Both calls ask Gemini to return strict JSON so the rest of the pipeline can
consume the result programmatically without brittle regex parsing.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

import config

logger = logging.getLogger("ai_agent")

genai.configure(api_key=config.GEMINI_API_KEY)


@dataclass
class Evaluation:
    match_score: float
    reasoning: str
    proposal_ar: Optional[str] = None


def _get_model():
    return genai.GenerativeModel(config.GEMINI_MODEL)


def _extract_json(text: str) -> Optional[dict]:
    """Gemini sometimes wraps JSON in ```json fences — strip and parse safely."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to find the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.error("Could not parse JSON from Gemini response: %s", text[:300])
    return None


def score_project(title: str, description: str) -> Optional[dict]:
    """Step 1: ask Gemini for a match score + short reasoning. Returns None on failure."""
    skills_list = ", ".join(config.MY_SKILLS)

    prompt = f"""
You are an expert freelance-bidding assistant. Compare the project below
against the freelancer's skill set and estimate how good a match it is.

Freelancer skills: {skills_list}

Project title: {title}
Project description: {description}

Respond with ONLY a raw JSON object (no markdown, no extra text) in this
exact shape:
{{
  "match_score": <integer 0-100>,
  "reasoning": "<one short sentence in English explaining the score>"
}}
"""
    try:
        model = _get_model()
        # CRITICAL: request_options timeout — without this the SDK call can
        # hang indefinitely on a stalled connection with no error raised.
        response = model.generate_content(
            prompt,
            request_options={"timeout": config.GEMINI_TIMEOUT},
        )
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
        model = _get_model()
        response = model.generate_content(
            prompt,
            request_options={"timeout": config.GEMINI_TIMEOUT},
        )
        text = response.text.strip()
        return text if text else None
    except Exception as exc:
        logger.error("Gemini proposal drafting failed: %s", exc, exc_info=True)
        return None


def evaluate_project(title: str, description: str, budget: Optional[str] = None) -> Evaluation:
    """
    Full pipeline for one project: score it, and if it clears the threshold,
    draft a proposal too. Always returns an Evaluation object — never raises —
    so main.py's loop can rely on it unconditionally.
    """
    score_data = score_project(title, description)
    if score_data is None:
        return Evaluation(match_score=0.0, reasoning="AI scoring unavailable (error).")

    score = float(score_data.get("match_score", 0))
    reasoning = score_data.get("reasoning", "")

    proposal = None
    if score > config.MATCH_THRESHOLD:
        proposal = draft_proposal(title, description, budget)

    return Evaluation(match_score=score, reasoning=reasoning, proposal_ar=proposal)
