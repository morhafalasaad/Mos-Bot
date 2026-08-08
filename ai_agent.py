"""
ai_agent.py
-----------
Uses Google's Gen AI SDK (`from google import genai`) to:
  1. Score how well a project matches the freelancer's skill set (0-100),
     and estimate a suggested bid price and delivery time.
  2. If the score clears the threshold, draft a persuasive, customized
     proposal in Arabic based on the project's details.

Three reliability features on top of that:

LOCAL TAG PRE-FILTERING (zero API cost for irrelevant projects)
-------------------------------------------------------------------
Before any Gemini call is made, `local_skill_prefilter()` compares the
project's official Mostaql skill tags (scraped from its detail page —
see scraper.fetch_project_tags) against config.MY_SKILLS — which can mix
English and Arabic entries freely; matching is plain case-insensitive
substring matching, language-agnostic (Python's str.lower() is a safe
no-op on Arabic script, so mixed-language lists just work). If there's no
overlap at all, `evaluate_project()` returns immediately with
match_score=0.0 and makes ZERO Gemini API calls for that project.

IMPORTANT — fail-open by design: if tags weren't fetched (empty list —
either FETCH_PROJECT_TAGS is off, or the detail-page fetch/parse failed),
the pre-filter does NOT block the project; it falls through to the normal
Gemini evaluation. We would rather spend an API call on an uncertain
project than silently drop a good one because of a scraping gap.

API KEY ROTATION (survive per-key free-tier quota limits)
-------------------------------------------------------------------
`config.GEMINI_API_KEYS` is a list. On a 429 / RESOURCE_EXHAUSTED quota
error, `_generate()` immediately rotates to the next key, rebuilds the
client, and retries — no backoff wait, since a different key's quota is
unrelated to how long we wait on this one.

TRANSIENT-ERROR RETRY (504 Gateway Timeout / DEADLINE_EXCEEDED / 503)
-------------------------------------------------------------------
These are different from quota errors: switching keys doesn't fix a
timed-out gateway, so `_generate()` instead retries the SAME key with
short exponential backoff, up to config.GEMINI_MAX_TRANSIENT_RETRIES times,
before giving up on that key and (only then) also rotating to the next key
as a last resort. Any error that is neither a quota error nor a recognized
transient error is raised immediately without retrying or rotating, so a
genuinely broken prompt/request doesn't waste time or keys.

Both retry paths are bounded (at most n_keys * (1 + max_transient_retries)
attempts total), so `_generate()` always eventually returns or raises —
it cannot loop forever. score_project()/draft_proposal() catch whatever it
raises and return None, and evaluate_project() turns that into a safe
fallback Evaluation (match_score=0.0, suggested_price=None,
delivery_days=None) rather than letting the exception propagate — so one
bad project can never take down main.py's loop.

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
import time
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from google.genai import types

import config

# Optional final-fallback for genuinely malformed JSON (e.g. a stray/garbled
# token where a value should be — corruption beyond what regex cleanup can
# reliably fix). json_repair is purpose-built for exactly this: repairing
# broken JSON from LLM outputs. If it's not installed, _extract_json still
# works via the sanitizer + balanced-brace scan below; json_repair is only
# the last line of defense for the messiest cases.
try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

logger = logging.getLogger("ai_agent")


# ---------------------------------------------------------------------------
# Client + API key rotation
# ---------------------------------------------------------------------------
# Module-level state is safe without locking because main.py runs cycles one
# at a time on a single worker thread (ThreadPoolExecutor(max_workers=1)).
_current_key_index = 0

if len(config.GEMINI_API_KEYS) > 1:
    logger.info(
        "Gemini: %s API key(s) configured for rotation on quota exhaustion.",
        len(config.GEMINI_API_KEYS),
    )
else:
    logger.warning(
        "Gemini: only 1 API key configured — GEMINI_API_KEYS is not set (or "
        "only contains one entry), so there is nothing to rotate to on a 429. "
        "Set GEMINI_API_KEYS as a comma-separated list of keys FROM DIFFERENT "
        "GOOGLE CLOUD PROJECTS to actually get separate quota pools — keys "
        "created under the same project can share the same underlying quota, "
        "in which case rotating between them will not avoid 429s either."
    )


def _build_client(key_index: int) -> genai.Client:
    return genai.Client(
        api_key=config.GEMINI_API_KEYS[key_index],
        http_options=types.HttpOptions(
            # timeout is in MILLISECONDS for this SDK.
            timeout=config.GEMINI_TIMEOUT * 1000,
            # CRITICAL: the SDK's own default retry behavior is up to 5
            # attempts with exponential backoff (up to ~60s), and 429 is in
            # its default retryable status list. Left at the default, a
            # single generate_content() call would silently retry the SAME
            # already-exhausted key up to 5 times internally — taking up to
            # ~a minute — before our exception handler in _generate() ever
            # sees it and gets a chance to rotate to the next key. Setting
            # attempts=1 disables the SDK's internal retry entirely, so a
            # 429 raises immediately and OUR rotation logic (which is what
            # actually knows about the other keys) takes over right away.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _build_client_safe(key_index: int) -> Optional[genai.Client]:
    """Same as _build_client, but never raises — used when rotating to a
    new key mid-retry-loop, since a single malformed/invalid key in the
    list must not abort rotation to the REST of the list."""
    try:
        return _build_client(key_index)
    except Exception as exc:
        logger.error(
            "Failed to build Gemini client for key #%s (is it malformed?): %s",
            key_index + 1, exc,
        )
        return None


_client = _build_client(_current_key_index)


def _is_quota_error(exc: Exception) -> bool:
    """Detects a 429 / RESOURCE_EXHAUSTED quota error across SDK versions,
    since relying on a single exception type/attribute is fragile."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


def _is_transient_error(exc: Exception) -> bool:
    """
    Detects gateway/server-side transient errors — 504 Gateway Timeout,
    DEADLINE_EXCEEDED, 503 Service Unavailable, and generic 500s — which are
    worth retrying on the SAME key after a short backoff (unlike quota
    errors, a different key doesn't fix a timed-out gateway).
    """
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
    # True specifically when the Gemini call itself failed (e.g. every key
    # in GEMINI_API_KEYS hit 429/RESOURCE_EXHAUSTED, or another API error) —
    # as opposed to a successful call that simply scored the project low.
    # main.py uses this to route to the GitHub fallback instead of the
    # normal "below threshold" path, since match_score=0.0 alone can't
    # distinguish "genuinely irrelevant project" from "we never actually
    # found out."
    ai_failed: bool = False


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    """
    Reads Google's own suggested wait time out of a 429 error's response
    body (a google.rpc.RetryInfo detail with a retryDelay like '13s'), when
    present, so _generate() can respect the server's actual guidance
    instead of guessing blindly with pure exponential backoff. Returns None
    if not present or not parseable — never raises.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    try:
        error_body = details.get("error", details)
        for item in error_body.get("details", []) or []:
            if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
                match = re.match(r"([\d.]+)\s*s", str(item.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    except (AttributeError, TypeError):
        pass
    return None


def _generate(prompt: str, json_mode: bool = False):
    """
    Calls the currently active Gemini API key/client. Three retry
    strategies, each scoped to attempts on a SINGLE key (counters reset for
    every new key, so none of them can leak into or interfere with each
    other across a rotation):

      - Quota error (429/RESOURCE_EXHAUSTED) WITH a server-provided
        retryDelay: wait exactly that (capped at GEMINI_QUOTA_BACKOFF_MAX),
        retry the SAME key, up to GEMINI_MAX_QUOTA_SAME_KEY_RETRIES times.
        A short server-suggested delay usually means a transient per-minute
        limit worth waiting out rather than immediately burning a rotation.
      - Quota error WITHOUT a usable retryDelay (or same-key retries used
        up): this key is now considered exhausted. Wait with our own
        exponential backoff (GEMINI_QUOTA_BACKOFF_BASE * 2^n, capped at
        GEMINI_QUOTA_BACKOFF_MAX) before rotating — avoids bursting through
        every remaining key within the same second and tripping each one's
        RPM limit in turn.
      - Transient error (504/DEADLINE_EXCEEDED/503/500): retry the SAME key
        with its own short exponential backoff, up to
        config.GEMINI_MAX_TRANSIENT_RETRIES times, before rotating.
      - Anything else: raised immediately, no retry/rotation wasted on a
        genuinely broken request.

    Bounded (each key gets at most 1 + GEMINI_MAX_QUOTA_SAME_KEY_RETRIES +
    GEMINI_MAX_TRANSIENT_RETRIES attempts, across at most n_keys keys), so
    this always eventually returns or raises. Only once EVERY key has been
    fully exhausted does this raise the last exception — callers
    (score_project/draft_proposal) catch that and return None, which
    evaluate_project() turns into Evaluation(ai_failed=True). THAT is the
    one and only point where main.py's GitHub fallback (Telegram alert +
    Issue/queue) gets triggered — never on an individual key's first 429,
    and never before rotation has had a full chance to work.
    """
    global _client, _current_key_index

    gen_config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    n_keys = len(config.GEMINI_API_KEYS)

    last_exc: Optional[Exception] = None
    keys_tried = 0

    while keys_tried < n_keys:
        transient_attempt = 0
        quota_same_key_attempts = 0

        while True:  # attempts on the CURRENT key only
            try:
                return _client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=gen_config,
                )
            except Exception as exc:
                last_exc = exc

                if _is_quota_error(exc):
                    retry_delay = _extract_retry_delay_seconds(exc)
                    if retry_delay is not None and quota_same_key_attempts < config.GEMINI_MAX_QUOTA_SAME_KEY_RETRIES:
                        quota_same_key_attempts += 1
                        wait = min(retry_delay, config.GEMINI_QUOTA_BACKOFF_MAX)
                        logger.warning(
                            "Gemini key #%s hit quota (429) with a server-suggested "
                            "retryDelay=%.1fs — waiting and retrying the SAME key "
                            "(attempt %s/%s) before treating it as exhausted",
                            _current_key_index + 1, wait,
                            quota_same_key_attempts, config.GEMINI_MAX_QUOTA_SAME_KEY_RETRIES,
                        )
                        time.sleep(wait)
                        continue  # retry same key

                    # No usable server-provided delay, or same-key retries
                    # already used up — this key is done. Back off (avoids
                    # tripping the NEXT key's RPM limit too), then rotate.
                    wait = min(
                        config.GEMINI_QUOTA_BACKOFF_BASE * (2 ** keys_tried),
                        config.GEMINI_QUOTA_BACKOFF_MAX,
                    )
                    logger.warning(
                        "Gemini API key #%s exhausted (429 RESOURCE_EXHAUSTED, no "
                        "further server-suggested delay to respect) — waiting "
                        "%.1fs before rotating to the next key",
                        _current_key_index + 1, wait,
                    )
                    time.sleep(wait)
                    break  # stop attempts on this key; rotate below

                if _is_transient_error(exc):
                    if transient_attempt < config.GEMINI_MAX_TRANSIENT_RETRIES:
                        wait = config.GEMINI_RETRY_BACKOFF_BASE * (transient_attempt + 1)
                        transient_attempt += 1
                        logger.warning(
                            "Transient Gemini error (%s: %s) on key #%s — retrying "
                            "same key in %ss (attempt %s/%s)",
                            type(exc).__name__, exc, _current_key_index + 1,
                            wait, transient_attempt, config.GEMINI_MAX_TRANSIENT_RETRIES,
                        )
                        time.sleep(wait)
                        continue  # retry same key
                    logger.warning(
                        "Transient Gemini error persisted after %s retries on key "
                        "#%s — rotating key as a last resort",
                        config.GEMINI_MAX_TRANSIENT_RETRIES, _current_key_index + 1,
                    )
                    break  # exhausted transient retries on this key; rotate below

                # Not a quota error, and not a (recognized) transient error
                # at all — not worth retrying or rotating for. Raise immediately.
                logger.error("Non-retryable Gemini error: %s", exc, exc_info=True)
                raise

        # Reached only via one of the `break`s above — this key is fully
        # done (quota-exhausted or transient-exhausted).
        keys_tried += 1

        # Find the next USABLE key, skipping any that fail to even build a
        # client (e.g. malformed key), without wasting a generate_content
        # attempt on a stale/unset client. Each skipped bad key also counts
        # toward keys_tried, so this stays bounded by n_keys.
        found_usable = False
        while keys_tried < n_keys:
            _current_key_index = (_current_key_index + 1) % n_keys
            candidate_client = _build_client_safe(_current_key_index)
            if candidate_client is not None:
                _client = candidate_client
                logger.warning(
                    "Rotating to Gemini API key #%s of %s after repeated failures on the previous key",
                    _current_key_index + 1, n_keys,
                )
                found_usable = True
                break
            logger.warning("Key #%s could not be initialized — skipping to next key", _current_key_index + 1)
            keys_tried += 1

        if not found_usable:
            break  # no usable key left to rotate to

    logger.error(
        "All %s Gemini API key(s) fully exhausted (quota and/or repeated "
        "transient errors, or failed to initialize) — THIS is the only "
        "point where the GitHub fallback (Telegram alert + Issue/queue) "
        "gets triggered, never on any single key's first failure. Last "
        "error: %s", n_keys, last_exc,
    )
    raise last_exc if last_exc else RuntimeError("No Gemini API keys available")


def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Scans for the first top-level {...} object using string-aware
    brace-depth tracking, instead of relying on regex alone. While
    scanning, any RAW (unescaped) control character found INSIDE a string
    literal — most commonly a literal newline, because Gemini sometimes
    line-wraps a free-text field like "reasoning" without escaping it — is
    rewritten to its valid escaped form (\\n, \\r, \\t). This is invisible
    in a terminal/log either way, which is exactly why this class of bug
    is so easy to miss just by reading the logged text: a raw newline
    inside a JSON string and the whitespace between key-value pairs look
    identical when printed, but only one of them is valid JSON.

    Why brace-depth tracking instead of regex: a plain greedy regex like
    r'\\{.*\\}' matches from the FIRST '{' to the VERY LAST '}' anywhere in
    the text, which breaks on trailing prose, a stray extra closing brace,
    or a second brace-like fragment further in the response. Tracking
    depth character-by-character (while staying string-aware) finds the
    exact end of the real JSON object and ignores everything after it.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    out = []
    control_escapes = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                out.append(ch)
                continue
            if ch == "\\":
                escape = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch in control_escapes:
                out.append(control_escapes[ch])  # fix: escape the raw control char
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)

    return None  # unbalanced (truncated response) — no complete object found


# Smart/curly quotes and other invisible Unicode characters that Gemini
# occasionally emits and that break json.loads while being visually
# indistinguishable (or literally invisible) from valid JSON in a log.
_SMART_QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"',   # “ ”  -> "
    "\u2018": "'", "\u2019": "'",   # ‘ ’  -> '
    "\u00a0": " ",                  # non-breaking space -> regular space
    "\u200b": "", "\u200c": "", "\u200d": "",  # zero-width chars -> removed
}


def _sanitize_text(text: str) -> str:
    """Strips a BOM and normalizes smart quotes / invisible Unicode
    whitespace that break json.loads but render as normal-looking
    characters (or nothing at all) wherever this text gets logged."""
    text = text.lstrip("\ufeff")
    for bad, good in _SMART_QUOTE_MAP.items():
        text = text.replace(bad, good)
    return text


def _strip_trailing_commas(text: str) -> str:
    """Removes a trailing comma right before a closing '}' or ']' — a very
    common small mistake in LLM-generated JSON that json.loads rejects
    outright (e.g. '{"a": 1,}')."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extracts and parses a JSON object from Gemini's response,
    tolerating the anomalies Gemini actually produces in practice:
    markdown fences, leading/trailing prose, trailing garbage or a stray
    extra closing brace, smart/curly quotes, a leading BOM, invisible
    Unicode whitespace, trailing commas, raw unescaped control characters
    (typically a literal newline) inside a string value, and — as a final
    layer — genuinely garbled/corrupted key-value pairs (e.g. a stray
    extra ':'/'"' where a value should be) via json_repair.

    Strategy, most-common-case first, most-permissive last:
      1. Sanitize invisible/smart characters, strip ```json fences,
         attempt a direct parse.
      2. Balanced-brace scan (string-aware, also fixes raw control chars
         found inside strings) for the first complete {...} object.
      3. Same balanced-brace result with trailing commas stripped, in case
         that was the (additional) problem.
      4. Greedy regex (\\{.*\\}), also with trailing commas stripped, in
         case the balanced scan found nothing at all (e.g. a genuinely
         truncated response).
      5. json_repair (if installed) as an absolute last resort for
         corruption too irregular for regex-based cleanup to fix
         deterministically — it may occasionally guess a value differently
         than intended for truly ambiguous input, but it reliably avoids
         raising, which is the actual goal: a formatting slip should never
         crash the pipeline.
    """
    sanitized = _sanitize_text(text.strip())
    cleaned = re.sub(r"^```(?:json)?|```$", "", sanitized, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    balanced = _extract_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            try:
                return json.loads(_strip_trailing_commas(balanced))
            except json.JSONDecodeError:
                pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_strip_trailing_commas(candidate))
            except json.JSONDecodeError:
                pass

    # Absolute last resort: json_repair is a heuristic JSON repair library
    # built for exactly this — genuinely garbled/corrupted LLM output that
    # regex-based cleanup can't reliably fix (e.g. a stray extra ':'/'"'
    # where a value should be, a missing comma, an unquoted key, etc.).
    # It may occasionally guess a value differently than intended for truly
    # ambiguous corruption, but it reliably avoids raising — which is the
    # actual goal here: never let a formatting slip crash the pipeline.
    if _HAS_JSON_REPAIR:
        try:
            repaired = json_repair.loads(cleaned)
            if isinstance(repaired, dict) and repaired:
                logger.warning(
                    "Standard JSON parsing failed — json_repair recovered a "
                    "result, but please verify it looks sane: %s", repaired,
                )
                return repaired
        except Exception as exc:
            logger.debug("json_repair also failed: %s", exc)

    logger.error("Could not parse JSON from Gemini response: %s", text[:500])
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


def draft_proposal(
    title: str,
    description: str,
    budget: Optional[str] = None,
) -> Optional[str]:
    """
    Step 2 (only called if score >= threshold): draft an Arabic proposal
    following Mostaql's professional-proposal standards.

    IMPORTANT — price/delivery time are DELIBERATELY NOT passed to this
    prompt and DELIBERATELY NOT mentioned anywhere in the proposal text.
    suggested_price/delivery_days (computed by score_project) still exist
    and are still sent to Telegram as their own dedicated fields — this
    function's prompt just never asks Gemini to restate them inside the
    proposal body, and explicitly forbids it from doing so on its own
    initiative, per Mostaql's rules against quoting price/duration inside
    proposal text (that information belongs only in the platform's
    dedicated bid fields, not embedded in free text).
    """
    skills_list = ", ".join(config.MY_SKILLS)
    budget_line = f"\n(للسياق فقط، لا تذكره: ميزانية العميل المعلنة هي {budget})" if budget else ""

    prompt = f"""
أنت مستقل خبير تكتب عرضك الشخصي لتقديمه على مشروع في منصة مستقل (Mostaql).
مهاراتك الفعلية (استخدم منها فقط ما يخدم هذا المشروع تحديداً، وتجاهل الباقي
تماماً): {skills_list}

عنوان المشروع: {title}
وصف المشروع: {description}{budget_line}

اكتب عرضاً شخصياً بصوت مستقل بشري حقيقي وخبير، باللغة العربية الفصحى،
يغطي هذه العناصر بشكل متدفق وطبيعي (بدون كتابة عناوين الأقسام، وبدون أن
يبدو كقالب جامد مكرر):

- تحية ومقدمة موجزة تعرّف بك كمستقل مختص.
- إثبات فهم دقيق ومحدد لما يحتاجه هذا العميل تحديداً كما ورد في وصف
  المشروع فعلياً — وليس فهماً عاماً ينطبق على أي مشروع مشابه.
- خطة عمل مختصرة (2-4 خطوات) تُظهر منهجية واضحة تبني الثقة.
- لماذا أنت الخيار المناسب: اذكر فقط الخبرات المرتبطة مباشرة بما يطلبه
  هذا المشروع تحديداً. لا تسرد كل مهاراتك، ولا تذكر أي تقنية (بايثون،
  OOP، فلاتر، الخ) إلا إذا كانت مطلوبة صراحة في وصف المشروع أو ضرورية
  تقنياً وبشكل مباشر لحل المشكلة المطروحة — ذِكر تقنيات غير ذات صلة
  "لحشو" العرض ممنوع تماماً.
- خاتمة احترافية تدعو العميل للتواصل أو طرح الأسئلة.

قواعد صارمة وإلزامية:
1. ممنوع منعاً باتاً ذكر أي رقم أو إشارة تخص السعر، التكلفة، الميزانية،
   أو مدة التسليم/عدد الأيام في أي مكان من نص العرض — تحت أي ظرف ولأي
   سبب. هذه المعلومات موجودة في حقول منفصلة خارج نص العرض على المنصة،
   وذكرها داخل النص يخالف قواعد مستقل. لا تكتب حتى عبارات عامة تلمّح
   لذلك مثل "سعر مناسب" أو "خلال مدة قصيرة" — تجنب الموضوع كلياً.
2. ممنوع حشو المهارات أو ذكرها بشكل روتيني في كل عرض — فقط ما يرتبط
   تحديداً بهذا المشروع كما هو موضح أعلاه.
3. اكتب بأسلوب إنساني طبيعي ومرن كما يكتب مستقل محترف حقيقي، وليس بأسلوب
   جامد أو نمطي يبدو آلياً أو مولداً تلقائياً. تجنب الجمل الجاهزة
   المكررة والعبارات الفضفاضة.
4. لا تبالغ ولا تعد بما لا يمكنك تنفيذه بدقة — كن شفافاً وواقعياً.
5. طوله لا يتجاوز 180 كلمة.
6. لا تضع أي عناوين أقسام أو تنسيق ماركداون، فقط نص العرض جاهزاً للنسخ
   مباشرة.
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
        return Evaluation(match_score=0.0, reasoning="AI scoring unavailable (error).", ai_failed=True)

    # Round to a whole number ONCE, immediately, before any comparison or
    # logging happens anywhere downstream (here, and in main.py). This is
    # what guarantees the score used in the threshold check, the one shown
    # in logs, and the one sent to Telegram are always the exact same
    # number — e.g. a raw 59.6 becomes 60 here and stays 60 everywhere,
    # instead of comparing 59.6 against the threshold while a log
    # elsewhere displays a separately-rounded "60%" that looks like it
    # should have passed.
    raw_score = float(score_data.get("match_score", 0))
    score = float(round(raw_score))

    reasoning = score_data.get("reasoning", "")
    suggested_price = score_data.get("suggested_price")
    delivery_days = score_data.get("delivery_days")
    try:
        delivery_days = int(delivery_days) if delivery_days is not None else None
    except (TypeError, ValueError):
        delivery_days = None

    proposal = None
    # >= : a score exactly equal to the threshold must be treated as a match,
    # not skipped. This must match main.py's notification-gate comparison
    # exactly, since proposal_ar only gets set here — if this gate were
    # stricter than main.py's, a boundary-score project would clear the
    # notification check but still have no proposal to send.
    if score >= config.MATCH_THRESHOLD:
        # Deliberately NOT passing suggested_price/delivery_days here — see
        # draft_proposal()'s docstring. They still flow to Telegram via the
        # Evaluation object below, just never into the proposal text itself.
        proposal = draft_proposal(title, description, budget)

    return Evaluation(
        match_score=score,
        reasoning=reasoning,
        suggested_price=suggested_price,
        delivery_days=delivery_days,
        proposal_ar=proposal,
    )
