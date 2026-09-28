# Mostaql Hybrid AI Freelance Assistant

## New features (Sept 2026)

1. **Automated screening-question answers** — when a project's detail
   page has explicit client screening questions ("How do you propose to
   implement this?", "What architecture will you use?"), the bot detects
   them (`scraper.parse_screening_questions`, a best-effort 3-strategy
   extractor — see its section docstring for the caveat on unverified
   CSS selectors), drafts one specific, technically-grounded answer per
   question (`ai_agent.draft_screening_answers`), and sends them in a
   dedicated "أسئلة الفحص" section below the proposal, each answer in its
   own tap-to-copy Telegram code block.
2. **Strict budget/timeline adherence** — `ai_agent.py`'s scoring prompt
   now instructs Gemini to use the client's own stated budget/timeline
   as-is whenever it's a reasonable fit for the scope, and only propose a
   different figure when the client's number is severely unrealistic (in
   which case a short justification is shown next to the price/delivery
   lines in Telegram).
3. **Reply Assistant** — paste a client's message directly into the bot's
   Telegram chat (12+ characters, not a `/command`) and it drafts 2-3
   distinct-tone reply options (brief/direct, detailed/technical,
   warm/reassuring, or firm-on-scope/flexible depending on the situation)
   covering both pre-hire negotiation and post-hire project management
   (milestone delivery, progress updates, scope-creep handling). See
   `reply_assistant.py`. This runs on the SAME Telegram listener thread as
   the existing Won/Lost outcome buttons — Telegram allows only one active
   long-poll connection per bot token, so both are dispatched from one
   shared `main.telegram_feedback_loop()`.

See `tests/README.md` for the full test coverage table, including the six
new test files covering all three features above (`test_budget_timeline_
adherence.py`, `test_screening_questions.py`, `test_scraper_screening_
questions.py`, `test_reply_assistant.py`, `test_main_reply_assistant.py`,
`test_notifier_new_features.py`).

## Stability fix log (silent-hang issue on Render)

If the worker ran for one or two cycles then went quiet with no crash and
no error, the root cause was **`ai_agent.py`'s Gemini calls had no
timeout** — `google-generativeai` does not time out by default, so a
stalled connection to Gemini blocks that thread forever with zero output.
Fixes applied:

1. **`ai_agent.py`** — every `model.generate_content(...)` call now passes
   `request_options={"timeout": config.GEMINI_TIMEOUT}` (default 30s).
2. **`scraper.py`** — `requests.get` now uses a tuple timeout
   `(connect_timeout=10, read_timeout=REQUEST_TIMEOUT)` instead of a single
   value, and a catch-all `except Exception` was added around the retry
   loop so no dependency bug can escape unnoticed.
3. **`main.py`** — `run_cycle()` now runs on a background thread via
   `ThreadPoolExecutor`, and the main thread calls
   `future.result(timeout=CYCLE_TIMEOUT)` (default 600s). This is a hard
   watchdog: even if something inside a cycle hangs despite the timeouts
   above, the main loop itself can never block past `CYCLE_TIMEOUT` and
   will log the timeout and move on to the next cycle.
4. **Logging** — `main.py` now uses a `FlushingStreamHandler` that calls
   `.flush()` after every log record, and forces `sys.stdout` into
   line-buffered mode. All tracebacks are logged in full via
   `traceback.format_exc()`. On Render, also set the environment variable
   `PYTHONUNBUFFERED=1` as a second layer of insurance against delayed logs.
5. **Sleep loop** — the single long `time.sleep(N)` was replaced with
   `safe_sleep()`, which sleeps in 60-second chunks and logs a heartbeat
   each chunk. This doesn't change behavior, but it makes a dead process
   immediately distinguishable from a normally-sleeping one in the logs —
   if heartbeats stop appearing, you know the process actually died at that
   point rather than wondering if it's just still asleep.
6. **`health_server.py`** (new) — since you're deploying as a Render **Web
   Service** (which requires a bound port), this runs a minimal stdlib
   HTTP server on `$PORT` in its own daemon thread, started from `main.py`
   before the bot loop begins. Running it on a separate thread is what
   makes it safe: a hang in the bot loop can't block the health check, and
   vice versa. If you'd rather deploy as a Render **Background Worker**
   instead (no port required), you can remove the `health_server` import
   and its one call in `main.py` and switch service types in Render.

After these changes, `main.py`'s outer loop and the watchdog together
guarantee the process logs a heartbeat or an explicit error at least once
every `CYCLE_TIMEOUT` seconds — it cannot go silent indefinitely.


A 24/7 background worker that monitors Mostaql for new projects, scores them
against your skill set using Gemini, drafts an Arabic proposal for strong
matches, and sends everything to you on Telegram for final human review and
manual submission (human-in-the-loop by design — this bot never auto-submits).

## Architecture

```
main.py            -> orchestration loop (runs forever, isolates failures)
scraper.py          -> polls Mostaql's public listing page, anti-ban measures
ai_agent.py         -> Gemini scoring, proposal drafting, screening-question answers
reply_assistant.py  -> Reply Assistant: drafts multi-option replies to pasted client messages
notifier.py          -> Telegram Bot API notifications
config.py            -> all settings/secrets from environment variables
db.py                -> MongoDB Atlas persistence layer
```

Flow each cycle: `scraper` finds new projects → `ai_agent` scores each one →
if score > `MATCH_THRESHOLD`, `ai_agent` drafts a proposal (and screening-
question answers, if any) → `notifier` sends it to your Telegram → you
review and submit manually on Mostaql.

## Before you deploy — important notes

- **Check Mostaql's `robots.txt` and Terms of Service** before scraping to
  confirm automated polling of the public listing page is allowed, and keep
  the polling interval conservative (default: random 5–10 min).
- Mostaql's HTML structure isn't guaranteed to match the selectors in
  `scraper.py` exactly — open the projects page, inspect the DOM, and update
  the `SELECTORS` dict at the top of `scraper.py` if needed. This also
  applies to `SCREENING_SELECTORS` (see `scraper.py`'s screening-questions
  section) — those are informed guesses, not confirmed against real markup.
- This bot **never submits proposals automatically** — Telegram is the final
  human checkpoint, satisfying the "human-in-the-loop" requirement.

## 1. Get your credentials

**Gemini API key**
1. Go to https://aistudio.google.com/app/apikey
2. Create an API key and copy it.

**Telegram bot**
1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` →
   follow the prompts → copy the bot token.
2. Message your new bot once (anything) so it can message you back.
3. Get your chat ID: message [@userinfobot](https://t.me/userinfobot) or call
   `https://api.telegram.org/bot<TOKEN>/getUpdates` after messaging your bot,
   and read the `chat.id` field from the JSON response.

## 2. Push the code to GitHub

```bash
cd mostaql_agent
git init
git add .
git commit -m "Initial commit: Mostaql AI freelance assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/mostaql-ai-assistant.git
git push -u origin main
```

`.gitignore` already excludes `.env` — never commit real secrets.

## 3. Set up MongoDB Atlas (required — durable persistence)

As of the Sept 2026 migration, the bot's persistence layer (seen-projects
dedup, score cache, daily request counter, token-usage stats, repost
history, win/loss outcomes, and the auto-retry queue) lives in MongoDB
Atlas — see `db.py`'s module docstring for the full "why" (short version:
local files on Render's free tier are wiped on every redeploy, and the
previous GitHub-hosted retry queue caused its own redeploys by committing
to this repo).

1. Sign up at https://cloud.mongodb.com (free) and create a new project.
2. Create a free-tier **M0** cluster (plenty for this workload).
3. Under **Database Access**, create a database user with a password.
4. Under **Network Access**, add `0.0.0.0/0` (allow access from anywhere)
   — Render's outbound IPs aren't static on the free plan, so this is the
   simplest option; Atlas's own username/password auth is still required
   to actually connect.
5. Click **Connect → Drivers**, copy the `mongodb+srv://...` connection
   string, and fill in your real username/password.
6. Set this as `MONGODB_URI` in your `.env` (local) or Render's
   environment variables (deployed) — see step 5 below. The bot will
   refuse to start without it, the same as a missing Gemini key.

## 4. Deploy on Render (recommended — free Background Worker tier)

1. Sign in at https://render.com and click **New → Background Worker**.
2. Connect your GitHub repo and select it.
3. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
4. Under **Environment → Environment Variables**, add each variable from
   `.env.example` with your real values (`GEMINI_API_KEY`,
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `MONGODB_URI`, etc.). Do not
   upload `.env`.
5. Click **Create Background Worker**. Render will build and start the
   process; check the **Logs** tab to confirm you see
   `Starting Mostaql AI Freelance Assistant worker...` and
   `Connected to MongoDB Atlas`.
6. Render's free background workers can spin down after inactivity on some
   plans — if you notice gaps, consider Render's paid always-on tier or the
   PythonAnywhere path below.

## 5. Alternative: Deploy on PythonAnywhere

1. Create a free account at https://www.pythonanywhere.com.
2. Open a **Bash console** and clone your repo:
   ```bash
   git clone https://github.com/<your-username>/mostaql-ai-assistant.git
   cd mostaql-ai-assistant
   pip install --user -r requirements.txt
   ```
3. Set environment variables. PythonAnywhere doesn't have a dashboard secrets
   UI on the free tier, so create a `.env` file directly in the console
   (it's excluded from git, so this is safe and stays server-side only):
   ```bash
   nano .env   # paste in the same keys/values as .env.example
   ```
4. Free PythonAnywhere accounts don't support true always-on background
   processes — use a **Scheduled Task** (Dashboard → Tasks) instead:
   - Command: `cd ~/mostaql-ai-assistant && python3 main.py`
   - Note: free scheduled tasks run once daily. For frequent polling behavior
     on the free tier, consider restructuring `main.py`'s loop into a
     single-pass script (remove the outer `while True`) and schedule it to
     run via a Task every hour instead of running one persistent process —
     or upgrade to a paid "Always-on task" for the true 24/7 worker described
     in this guide.

## 6. Alternative: Any cloud VPS (DigitalOcean, AWS Lightsail, Oracle Free Tier, etc.)

```bash
git clone https://github.com/<your-username>/mostaql-ai-assistant.git
cd mostaql-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...   # or use a real .env + python-dotenv
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
export MONGODB_URI=...
nohup python main.py > worker.log 2>&1 &
```

For a more robust setup, run it under `systemd` or `screen`/`tmux` so it
survives SSH disconnects and auto-restarts on VPS reboot.

## Local testing

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python main.py
```

Run the automated test suite (fully offline, no real API keys needed):

```bash
pip install -r requirements-dev.txt
pytest
```

## Python version

This project targets one Python series across CI, Render, and local
development, tracked in two files that must be kept in sync:

- `runtime.txt` — Render's required filename/format (`python-3.14`, no
  patch — Render resolves the latest matching patch at deploy time).
- `.python-version` — the same series in the bare form (`3.14`, no
  `python-` prefix) that `actions/setup-python` and tools like pyenv/asdf
  read natively. `.github/workflows/tests.yml` reads this file via
  `python-version-file` rather than a hardcoded version, so CI always
  targets the same series Render deploys with, from one edit.

Both intentionally omit the patch version so a runner-image or Render
base-image update can't break the build the way pinning an exact patch
did previously (see `.github/workflows/tests.yml`'s own comment for that
history). Bumping to a new CPython series (e.g. once 3.15 is stable and
every package in `requirements.txt` supports it) means editing both files
together and pushing — the GitHub Actions workflow then runs all tests
against the new version automatically before you'd merge. Dependabot
(`.github/dependabot.yml`) keeps `requirements.txt`/`requirements-dev.txt`
current automatically, but does **not** bump the Python version itself —
no Dependabot ecosystem does that; see `dependabot.yml`'s comment.

**The bump itself IS automated**, just not via Dependabot:
`.github/workflows/auto-update-python.yml` runs weekly, checks the latest
*stable* Python series against the same manifest `actions/setup-python`
itself uses, and opens a PR updating both files when a newer one exists —
`tests.yml` then runs automatically against that PR. The PR is left for
manual merge by default rather than auto-merged, since this project's test
suite (183 tests, all mocked — no real Gemini/Telegram/Mostaql/MongoDB
calls) can't fully characterize whether a Python series bump is safe in
production, and `cloudscraper` in particular has had no upstream release
since April 2023. See that workflow file's own comments for exactly how to
opt into fully unattended auto-merge if you'd rather accept that tradeoff.

## Tuning

- `MATCH_THRESHOLD` (default 60) — raise it to be more selective.
- `POLL_INTERVAL_MIN` / `MAX` — how often to check Mostaql (seconds).
- `MY_SKILLS` — comma-separated env var (see `.env.example`) that overrides
  the built-in skill list entirely; include both English and Arabic terms,
  since Mostaql tags/titles use both. Editing this on Render takes effect
  on the next restart — no code change or redeploy needed. Leave unset to
  use the default list built into `config.py`.
- `REPLY_ASSISTANT_ENABLED` (default true) — disable to turn off the Reply
  Assistant text-message listener entirely.
- `REPLY_ASSISTANT_OPTION_COUNT` (default 3) — how many reply drafts to
  generate per pasted client message.

### Gemini model cascade & key rotation

`ai_agent.py`'s Gemini calls go through `gemini_client.py` — a fully async,
quota-aware CASCADE across models and API keys. Every model is tried across
ALL configured keys (round-robin) before the cascade falls back to the next,
lower-quota model. Default hierarchy (override with `GEMINI_MODEL_CASCADE`,
a comma-separated list):

1. **Primary tier** (15 RPM / 500 RPD each): `gemini-3.5-flash-lite`,
   `gemini-3.1-flash-lite`
2. **Fallback tier** (5 RPM / 20 RPD each), tried only once every primary
   key is exhausted: `gemini-3.8-flash` → `gemini-3.7-flash` →
   `gemini-3.6-flash` → `gemini-3.5-flash` → `gemini-3-flash` →
   `gemini-2.5-flash-lite`

**Blacklisted models** — `gemini-2-flash`, `gemini-2-flash-lite`,
`gemini-2.5-pro`, `gemini-3.1-pro`, and `gemini-omni-flash` (0 RPM / 0 RPD,
deprecated or disabled) can never be configured or routed to. Any of these
names (or a `-preview`/`-001`/dated-suffix variant of one) supplied via
`GEMINI_MODEL_CASCADE` or the legacy `GEMINI_MODEL` env var is dropped at
startup with a logged warning, not silently ignored.

**429 handling**: a per-minute (RPM) 429 rotates to the next API key on the
*same* model immediately; a per-day (RPD) 429 parks that (model, key) pair
until the next Pacific-time midnight (when Google resets RPD) and moves on
to the next model. The two are told apart from Google's own
`QuotaFailure.violations[].quotaId` in the error body, not by guesswork.
When every model × key combination is unavailable, the whole cascade is
retried with bounded exponential backoff before finally raising.

**⚠️ Quotas are per Google Cloud *project*, not per API key.** Several keys
minted inside the SAME project share one quota pool — rotating between them
will not avoid a 429. For `GEMINI_API_KEYS` rotation to actually help, each
key must come from a **different** Google Cloud project (the bot logs a
reminder about this at startup if only one key, or keys sharing a project,
are configured).

Related env vars (all optional, sensible defaults in `config.py`):
`GEMINI_MODEL_CASCADE`, `GEMINI_RPM_FLASH_LITE_35`/`_31`/`_38`/`_37`/`_36`/
`_35`/`_3`/`_LITE_25`, `GEMINI_RPD_FLASH_*` (same suffixes), `GEMINI_MAX_RPM_PER_KEY`,
`GEMINI_BACKOFF_SWEEPS`, `GEMINI_OUTER_BACKOFF_BASE`/`_CAP`,
`GEMINI_TOTAL_DEADLINE_SECONDS`, `GEMINI_MAX_TRANSIENT_RETRIES`,
`GEMINI_RETRY_BACKOFF_BASE`, `GEMINI_QUOTA_BACKOFF_MAX`,
`GEMINI_INTER_REQUEST_DELAY`. `GEMINI_MODEL` (legacy, single-model) is still
honored as a back-compat shim: if set, it's inserted at the front of the
cascade ahead of even the primary tier.
