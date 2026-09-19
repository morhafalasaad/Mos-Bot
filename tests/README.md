# Tests

Unit tests for the main features built into this bot. All tests are
fully offline — no real Gemini, Telegram, or Mostaql network calls are
ever made; every external call is mocked via `monkeypatch`.

## Running

```bash
pip install -r requirements-dev.txt
pytest
```

Or target a single file/test while working on something specific:

```bash
pytest tests/test_score_cache.py -v
pytest tests/test_batch_scoring.py::test_batch_scoring_reduces_call_count -v
```

## What's covered

| File | Feature |
|---|---|
| `test_score_cache.py` | `ai_agent.ScoreCache` — get/set, `MY_SKILLS` invalidation, max-entries eviction, corrupt-file handling, budget/timeline field round-trip |
| `test_batch_scoring.py` | Batch scoring call-count reduction, dropped-index handling, total batch failure, cache interaction |
| `test_local_prefilter.py` | `local_skill_prefilter`'s tag-based and title/description-based checks |
| `test_adaptive_threshold.py` | `DailyRequestTracker` persistence/day-rollover, `get_effective_match_threshold`'s ramp |
| `test_client_aware_proposals.py` | `draft_proposal`'s client-info tone adaptation and the no-mention safety rule |
| `test_richer_reasoning.py` | `matched_skills`/`missing_skills` parsing, including malformed-input defensiveness |
| `test_budget_timeline_adherence.py` | `ProjectScoreSchema`'s budget/timeline strict-adherence fields, and their flow through `_finalize_score_result` (fresh score, cache hit, missing-field defaults) |
| `test_screening_questions.py` | `ai_agent.draft_screening_answers` and its wiring into `_finalize_score_result`/`evaluate_project`/`evaluate_projects_batch` — only called above threshold, only when questions exist, never blocks the proposal on failure |
| `test_scraper_screening_questions.py` | `scraper.parse_screening_questions`'s three extraction strategies (CSS selector, heading-anchored sibling walk, raw numbered-question regex) against synthetic HTML |
| `test_reply_assistant.py` | `reply_assistant.get_reply_options` (prompt construction, multi-option parsing, failure handling) and `looks_like_client_message`'s local heuristic |
| `test_main_reply_assistant.py` | `main.py`'s `_handle_text_message` routing (chat-ID filter, short-message skip) and the single shared Telegram listener's dispatch between `callback_query` and `message` updates |
| `test_notifier_new_features.py` | `notifier.py`'s screening-questions section, budget/timeline adjustment note, and Reply Assistant message formatting |
| `test_outcome_tracker.py` | Win/loss recording, correction/overwrite, validation |
| `test_repost_detector.py` | Exact/fuzzy repost detection, expiry, size cap, self-exclusion |
| `test_notifier.py` | Telegram message and inline-keyboard construction |
| `test_scraper_categories.py` | `MOSTAQL_CATEGORIES` → request URL building |
| `test_my_skills_config.py` | `MY_SKILLS` env var override behavior |
| `test_main_telegram_offset.py` | Telegram feedback offset persistence, fail-safe against any exception type |
| `test_health_server.py` | `/status` snapshot, heartbeat staleness detection across all three worker threads |
| `test_incident_regressions.py` | Regression coverage for the retired `gemini-2.5-flash-lite` model and the Telegram 409 Conflict handling fix |

## How isolation works

`conftest.py` does two things before any test module can import the app
code:

1. Adds the project root to `sys.path` (the app modules live at the repo
   root, not an installed package).
2. Sets placeholder `GEMINI_API_KEYS`/`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/
   `MONGODB_URI` env vars, since `config.py` raises immediately at import
   time if these are missing.

An autouse fixture then swaps `db.py`'s live database handle for a fresh
in-memory `mongomock` database on every single test, and rebuilds
`ai_agent`'s module-level singletons (`ScoreCache`, `DailyRequestTracker`)
so they resolve against that fresh mock rather than a stale reference —
nothing a test does ever touches a real database, and tests can't leak
state into each other regardless of execution order.

## A note on interacting features

Several tests explicitly disable `TITLE_PREFILTER_ENABLED` even though
they're not testing the pre-filter itself. This is intentional: the local
pre-filter (see `test_local_prefilter.py`) checks a project's title and
description for skill-keyword overlap when no tags are given, and several
other tests use short placeholder text like `"T"` / `"D"` that
legitimately contains no overlap with the test `MY_SKILLS` — which would
otherwise cause the pre-filter to correctly, but distractingly, intercept
the project before the actual feature under test ever runs. Disabling it
in those specific tests isolates the feature being tested; the
interaction itself is still fully covered separately in
`test_local_prefilter.py` and `test_batch_scoring.py`.

The same principle applies to `test_screening_questions.py` and
`test_budget_timeline_adherence.py`: both set `MATCH_THRESHOLD` explicitly
(sometimes very high, sometimes low) to deterministically isolate whether
proposal/screening-answer drafting fires, rather than relying on
whatever score a fake response happens to produce.

## A note on the screening-question CSS selectors

`test_scraper_screening_questions.py` verifies the EXTRACTION LOGIC in
`scraper.parse_screening_questions` against representative synthetic
HTML for all three strategies (CSS selector, heading-anchored fallback,
raw-regex fallback). It cannot verify that `SCREENING_SELECTORS` matches
real Mostaql markup — that requires a live project page with actual
screening questions to confirm, the same caveat that already applies to
every other selector in `scraper.py` (see its own module docstring).
