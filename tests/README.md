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
| `test_score_cache.py` | `ai_agent.ScoreCache` — get/set, `MY_SKILLS` invalidation, max-entries eviction, corrupt-file handling |
| `test_batch_scoring.py` | Batch scoring call-count reduction, dropped-index handling, total batch failure, cache interaction |
| `test_local_prefilter.py` | `local_skill_prefilter`'s tag-based and title/description-based checks |
| `test_adaptive_threshold.py` | `DailyRequestTracker` persistence/day-rollover, `get_effective_match_threshold`'s ramp |
| `test_client_aware_proposals.py` | `draft_proposal`'s client-info tone adaptation and the no-mention safety rule |
| `test_richer_reasoning.py` | `matched_skills`/`missing_skills` parsing, including malformed-input defensiveness |
| `test_outcome_tracker.py` | Win/loss recording, correction/overwrite, validation |
| `test_repost_detector.py` | Exact/fuzzy repost detection, expiry, size cap, self-exclusion |
| `test_notifier.py` | Telegram message and inline-keyboard construction |
| `test_scraper_categories.py` | `MOSTAQL_CATEGORIES` → request URL building |
| `test_my_skills_config.py` | `MY_SKILLS` env var override behavior |

## How isolation works

`conftest.py` does two things before any test module can import the app
code:

1. Adds the project root to `sys.path` (the app modules live at the repo
   root, not an installed package).
2. Sets placeholder `GEMINI_API_KEYS`/`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`
   env vars, since `config.py` raises immediately at import time if these
   are missing.

An autouse fixture then points every on-disk state file this codebase
writes (score cache, daily request counter, outcomes log, repost history)
at a throwaway path under pytest's per-test `tmp_path` — nothing a test
does ever touches the real files in the project root, and tests can't
leak state into each other.

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
