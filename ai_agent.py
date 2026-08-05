import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from google.genai import types

import config

logger = logging.getLogger("ai_agent")

# ---------------------------------------------------------------------------
# Client + API key rotation
# ---------------------------------------------------------------------------
_current_key_index = 0

def _build_client(key_index: int) -> genai.Client:
    return genai.Client(
        api_key=config.GEMINI_API_KEYS[key_index],
        http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT * 1000),
    )

_client = _build_client(_current_key_index)

def _is_quota_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()

def _is_transient_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (500, 502, 503, 504):
        return True
    text = str(exc)
    markers = (
        "504", "DEADLINE_EXCEEDED", "Gateway Timeout",
        "503", "UNAVAILABLE", "Service Unavailable",
        "500", "Internal error", "Server disconnected",
    )
    return any(marker.lower() in text.lower() for marker in markers)

@dataclass
class Evaluation:
    match_score: float
    reasoning: str
    suggested_price: Optional[str] = None
    delivery_days: Optional[int] = None
    proposal_ar: Optional[str] = None

def _generate(prompt: str, json_mode: bool = False):
    global _client, _current_key_index

    gen_config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    n_keys = len(config.GEMINI_API_KEYS)

    last_exc: Optional[Exception] = None
    keys_tried = 0

    while keys_tried < n_keys:
        for transient_attempt in range(config.GEMINI_MAX_TRANSIENT_RETRIES + 1):
            try:
                return _client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=gen_config,
                )
            except Exception as exc:
                last_exc = exc

                if _is_quota_error(exc):
                    logger.warning(
                        "Gemini API key #%s hit quota (429/RESOURCE_EXHAUSTED) — rotating key",
                        _current_key_index + 1,
                    )
                    break

                if _is_transient_error(exc):
                    if transient_attempt < config.GEMINI_MAX_TRANSIENT_RETRIES:
                        wait = config.GEMINI_RETRY_BACKOFF_BASE * (transient_attempt + 1)
                        logger.warning(
                            "Transient Gemini error (%s: %s) on key #%s — retrying same "
                            "key in %ss (attempt %s/%s)",
                            type(exc).__name__, exc, _current_key_index + 1,
                            wait, transient_attempt + 1, config.GEMINI_MAX_TRANSIENT_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(
                            "Transient Gemini error persisted after %s retries on key #%s "
                            "— rotating key as a last resort",
                            config.GEMINI_MAX_TRANSIENT_RETRIES, _current_key_index + 1,
                        )
                        break

                logger.error("Non-retryable Gemini error: %s", exc, exc_info=True)
                raise

        keys_tried += 1
        if keys_tried >= n_keys:
            break
        _current_key_index = (_current_key_index + 1) % n_keys
        logger.warning(
            "Rotating to Gemini API key #%s of %s after repeated failures on the previous key",
            _current_key_index + 1, n_keys,
        )
        _client = _build_client(_current_key_index)

    logger.error(
        "All %s Gemini API key(s) exhausted (quota and/or repeated transient errors). "
        "Last error: %s", n_keys, last_exc,
    )
    raise last_exc if last_exc else RuntimeError("No Gemini API keys available")

def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None

def _extract_json(text: str) -> Optional[dict]:
    stripped = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", stripped, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    balanced = _extract_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

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
    tokens = [skill.strip()]
    paren_match = re.search(r"\(([^)]+)\)", skill)
    if paren_match:
        tokens.append(paren_match.group(1).strip())
    stripped = re.sub(r"\([^)]*\)", "", skill).strip()
    if stripped and stripped not in tokens:
        tokens.append(stripped)
    return [t for t in tokens if len(t) >= 2]

def local_skill_prefilter(tags: List[str]) -> bool:
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
  "suggested_price": "<a realistic recommended bid price/budget for this project's scope, as a short string including currency, e.g. '$150' or '$300-400'>",
  "delivery_days": <integer, realistic number of days to complete the project based on its scope>
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
