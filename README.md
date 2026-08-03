# Mostaql Hybrid AI Freelance Assistant

A 24/7 background worker that monitors Mostaql for new projects, scores them
against your skill set using Gemini, drafts an Arabic proposal for strong
matches, and sends everything to you on Telegram for final human review and
manual submission (human-in-the-loop by design — this bot never auto-submits).

## Architecture

```
main.py        -> orchestration loop (runs forever, isolates failures)
scraper.py      -> polls Mostaql's public listing page, anti-ban measures
ai_agent.py    -> Gemini scoring + Arabic proposal drafting
notifier.py     -> Telegram Bot API notifications
config.py       -> all settings/secrets from environment variables
```

Flow each cycle: `scraper` finds new projects → `ai_agent` scores each one →
if score > `MATCH_THRESHOLD`, `ai_agent` drafts a proposal → `notifier` sends
it to your Telegram → you review and submit manually on Mostaql.

## Before you deploy — important notes

- **Check Mostaql's `robots.txt` and Terms of Service** before scraping to
  confirm automated polling of the public listing page is allowed, and keep
  the polling interval conservative (default: random 5–10 min).
- Mostaql's HTML structure isn't guaranteed to match the selectors in
  `scraper.py` exactly — open the projects page, inspect the DOM, and update
  the `SELECTORS` dict at the top of `scraper.py` if needed.
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

`.gitignore` already excludes `.env` and `seen_projects.json` — never commit
real secrets.

## 3. Deploy on Render (recommended — free Background Worker tier)

1. Sign in at https://render.com and click **New → Background Worker**.
2. Connect your GitHub repo and select it.
3. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
4. Under **Environment → Environment Variables**, add each variable from
   `.env.example` with your real values (`GEMINI_API_KEY`,
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, etc.). Do not upload `.env`.
5. Click **Create Background Worker**. Render will build and start the
   process; check the **Logs** tab to confirm you see
   `Starting Mostaql AI Freelance Assistant worker...`.
6. Render's free background workers can spin down after inactivity on some
   plans — if you notice gaps, consider Render's paid always-on tier or the
   PythonAnywhere path below.

> Note: `seen_projects.json` is written to local disk, which is **ephemeral**
> on Render (resets on redeploy/restart). This only affects deduplication
> across restarts, not core functionality. For persistence across restarts,
> swap it for a free key-value store (e.g., Render's own Redis add-on, or a
> free tier of Upstash Redis) — the `_load_seen`/`_save_seen` functions in
> `scraper.py` are isolated so this is a small, contained change.

## 4. Alternative: Deploy on PythonAnywhere

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

## 5. Alternative: Any cloud VPS (DigitalOcean, AWS Lightsail, Oracle Free Tier, etc.)

```bash
git clone https://github.com/<your-username>/mostaql-ai-assistant.git
cd mostaql-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...   # or use a real .env + python-dotenv
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
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

## Tuning

- `MATCH_THRESHOLD` (default 75) — raise it to be more selective.
- `POLL_INTERVAL_MIN` / `MAX` — how often to check Mostaql (seconds).
- `MY_SKILLS` in `config.py` — edit the skill list Gemini scores against.
