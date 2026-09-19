"""
reply_assistant.py
-------------------
"Reply Assistant" mode: given a client message pasted into Telegram (as a
normal text message, not a button tap — see main.py's
telegram_reply_assistant_loop), drafts several DISTINCT reply options so
the human can pick whichever tone fits the situation, then send it
manually on Mostaql (same human-in-the-loop principle as every other
notification in this bot — nothing here ever auto-sends anything).

SCOPE
-------------------------------------------------------------------
Two broad situations, both handled by ONE Gemini call (the prompt itself
asks the model to first classify, then draft accordingly), since asking
Gemini to classify-then-draft in one pass is both cheaper and more
context-aware than a separate classification call followed by a second
drafting call:

  PRE-HIRE  — a prospective client is asking questions about your
              proposal, negotiating price/scope, or deciding whether to
              hire you. Replies aim to build trust, resolve technical
              doubts, and move toward being hired — without over-
              promising or being pushy.

  POST-HIRE — an already-hired client is messaging during an active
              project. Three common sub-cases the prompt explicitly
              names (the model infers which applies from the pasted
              text, if any is obviously that): delivering a milestone
              and asking for feedback, sending a routine progress
              update, and politely handling scope creep (new/extra asks
              beyond what was agreed). The model isn't forced to pick
              exactly one label — an ordinary reply just falls under a
              general "ongoing project communication" framing.

Always returns exactly config.REPLY_ASSISTANT_OPTION_COUNT (default 3)
DISTINCT options, each carrying a short tone label (e.g. "مختصر ومباشر" /
"تفصيلي وتقني" / "دافئ ومطمئن") so the human can tell them apart at a
glance in Telegram, mirroring notifier.py's existing "advisory, human
picks" pattern used everywhere else in this codebase.

FAIL-SAFE PHILOSOPHY (unchanged from every other module here)
-------------------------------------------------------------------
Never raises. On any failure, returns an empty list — the caller
(main.py's telegram_reply_assistant_loop) is responsible for telling the
human the draft couldn't be generated, rather than this module trying to
paper over a failure with a fake/generic reply that could be sent by
mistake.

This module reuses ai_agent's existing Gemini plumbing (_generate, key
rotation, rate limiting, retry) rather than reimplementing any of it —
see the imports below. It does NOT touch score_project/draft_proposal or
any scoring state; a Reply Assistant call is entirely independent of
whether/how a project was ever scored.
"""

import logging
import time
from typing import List, Optional

from pydantic import BaseModel, Field

import config

logger = logging.getLogger("reply_assistant")


class _ReplyOption(BaseModel):
    label: str = Field(
        description="A short (2-4 word) Arabic label naming this option's tone/approach, e.g. "
                     "'مختصر ومباشر', 'تفصيلي وتقني', 'دافئ ومطمئن'. Must be genuinely different "
                     "from the other options' labels — not just a rewording of the same idea.",
    )
    reply: str = Field(
        description="The full drafted reply message in Arabic, ready to paste directly into the chat/message "
                     "box as-is. Plain text, no markdown formatting, no placeholder brackets like [name] — "
                     "write it as a complete, sendable message.",
    )


class ReplyOptionsSchema(BaseModel):
    situation_summary: str = Field(
        description="One short English sentence classifying the situation, e.g. 'Pre-hire client asking "
                     "about timeline before committing' or 'Post-hire client requesting scope addition not "
                     "in original agreement'. Shown to the human above the options, not sent anywhere.",
    )
    options: List[_ReplyOption] = Field(
        default_factory=list,
        description="Multiple genuinely distinct reply drafts, each with its own label.",
    )


def _build_prompt(client_message: str, project_context: Optional[str], option_count: int) -> str:
    context_block = f"\nProject context (for background only, do not repeat it verbatim in the reply):\n{project_context}\n" if project_context else ""

    return f"""
You are an expert freelancer's communication assistant on a freelance
platform (Mostaql). The freelancer just received the message below from a
client and needs to reply. Your job is to (1) briefly classify what kind
of situation this is, and (2) draft {option_count} genuinely DIFFERENT
reply options so the freelancer can choose the one that best fits their
judgment of the situation.

Classify the situation as one of, or a natural blend of:
- PRE-HIRE: the client hasn't hired yet — they're asking clarifying
  questions, negotiating price/scope, or deciding whether to proceed.
  Aim to build trust, answer technical doubts specifically and honestly,
  and move toward being hired, without sounding desperate or overselling.
- POST-HIRE / milestone delivery: the freelancer is delivering completed
  work and asking for feedback.
- POST-HIRE / progress update: a routine "here's where things stand"
  update, not tied to a specific deliverable being handed over.
- POST-HIRE / scope creep: the client is asking for something beyond what
  was originally agreed. Handle this politely and professionally — never
  flatly refuse, but don't silently agree to unpaid extra work either;
  acknowledge the request, note it's outside the current agreed scope,
  and suggest a clear next step (discussing it as an addition, a separate
  follow-up project, or an adjusted timeline/price) without being
  confrontational.
- General ongoing project communication, if none of the above fits well.

Client's message (as pasted by the freelancer):
\"\"\"{client_message}\"\"\"
{context_block}
Now draft {option_count} distinct reply options. Each option must:
- Be a complete, ready-to-send message in Arabic (unless the client's
  message was clearly in English, in which case reply in English instead
  — match the client's own language).
- Actually differ in approach/tone from the other options, not just
  reworded restatements of the same message. Reasonable spreads to use
  across the options: brief-and-direct vs. detailed-and-technical vs.
  warm-and-reassuring; or firm-on-scope vs. flexible-with-conditions, when
  the situation calls for a scope/price negotiation instead.
- Never invent commitments, prices, or dates not already implied by the
  conversation context given above.
- Sound like a real, professional human freelancer wrote it — not
  robotic, not overly formal, not overly casual.
"""


def get_reply_options(
    client_message: str,
    project_context: Optional[str] = None,
    option_count: Optional[int] = None,
) -> tuple:
    """
    Main entry point. Returns (situation_summary, options, call_stats)
    where options is a list of {"label": str, "reply": str} dicts — always
    length 0 (on failure) or up to `option_count` (default
    config.REPLY_ASSISTANT_OPTION_COUNT). Never raises.

    project_context is OPTIONAL free text (e.g. the matched project's
    title + a short excerpt of its description, if the human is replying
    about a specific tracked project) — purely for grounding the model's
    understanding of what's being discussed; entirely optional since a
    client message is often self-contained enough on its own.
    """
    # Imported lazily (not at module load time) to avoid a hard import-
    # order dependency between reply_assistant and ai_agent — both are
    # leaf modules imported by main.py, and this keeps reply_assistant
    # usable/testable without requiring ai_agent's Gemini client to have
    # already been constructed first.
    import ai_agent

    option_count = option_count or config.REPLY_ASSISTANT_OPTION_COUNT
    prompt = _build_prompt(client_message, project_context, option_count)

    start = time.time()
    try:
        response = ai_agent._generate(
            prompt,
            response_schema=ReplyOptionsSchema,
            max_output_tokens=config.GEMINI_REPLY_ASSISTANT_MAX_OUTPUT_TOKENS,
        )
        stats = ai_agent._extract_call_stats(response, time.time() - start)

        data = None
        try:
            parsed = response.parsed
            if parsed is not None:
                data = parsed.model_dump()
        except Exception as parsed_exc:
            logger.warning("Reply-assistant response.parsed access failed, falling back to text parsing: %s", parsed_exc)

        if data is None:
            try:
                data = ai_agent.parse_gemini_json(response.text)
            except ValueError as parse_exc:
                logger.error("%s", parse_exc)
                return None, [], stats

        summary = str(data.get("situation_summary") or "").strip()
        raw_options = data.get("options")
        options: List[dict] = []
        if isinstance(raw_options, list):
            for item in raw_options:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                reply = str(item.get("reply") or "").strip()
                if label and reply:
                    options.append({"label": label, "reply": reply})

        return summary or None, options, stats
    except Exception as exc:
        logger.error("Gemini reply-assistant call failed: %s", exc, exc_info=True)
        stats = {
            "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "response_time_sec": time.time() - start, "key_alias": None,
        }
        return None, [], stats


def looks_like_client_message(text: str) -> bool:
    """
    Cheap, local (zero-Gemini-cost) heuristic used by main.py's listener
    to decide whether a pasted Telegram text message is worth spending a
    Reply Assistant Gemini call on, vs. being an ordinary short chat
    message/typo/command that isn't actually a client message to respond
    to. Deliberately permissive (fails open toward "yes, try it") — the
    real judgment call belongs to the human pasting the text, this is
    only a trivial guard against generating 3 replies to "hi" or "test".
    Never raises.
    """
    try:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if stripped.startswith("/"):
            return False  # looks like a bot command, not a pasted message
        return len(stripped) >= config.REPLY_ASSISTANT_MIN_MESSAGE_CHARS
    except Exception:
        return False
