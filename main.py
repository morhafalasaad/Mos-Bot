"""
main.py
-------
Entry point for the background worker. Designed to run 24/7 on Render as a
Web Service (bot loop + dummy health-check server side by side).

Why the bot was hanging (root causes fixed in this version):
1. `model.generate_content(prompt)` in ai_agent.py had NO timeout. The
   google-generativeai SDK does not time out by default, so a stalled
   connection to Gemini would block that thread forever with zero error
   output — exactly the "stops logging, never crashes, never restarts"
   symptom described. Fixed via `request_options={"timeout": ...}` in
   ai_agent.py.
2. No watchdog around a full cycle. Even with per-call timeouts, unknown
   edge cases (DNS hangs, SSL handshake stalls, etc.) can occasionally slip
   past a library's own timeout handling. `run_cycle()` is now executed in
   a worker thread with `future.result(timeout=CYCLE_TIMEOUT)` in the main
   thread, so the main loop itself can NEVER block longer than
   CYCLE_TIMEOUT seconds, no matter what happens inside the cycle.
3. Logging wasn't guaranteed to flush immediately when stdout isn't a TTY
   (as on Render). Fixed by forcing line-buffered stdout and flushing
   explicitly after every log record.
4. A single long `time.sleep(N)` doesn't itself cause hangs, but it does
   mean you can't tell, from the logs, whether the process is "sleeping
   normally" or "already dead" during that window. Replaced with a chunked
   sleep that logs a heartbeat periodically, so a silent process is now
   immediately distinguishable from a normal sleeping one in the Render logs.
"""

import concurrent.futures
import logging
import random
import sys
import time
import traceback

import config
import scraper
import ai_agent
import notifier
import health_server
import github_fallback

# ---------------------------------------------------------------------------
# Logging setup: force unbuffered / line-buffered output so Render's log
# dashboard shows lines in real time instead of in delayed batches.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
except Exception:
    pass


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that explicitly flushes after every single record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# One worker is enough — cycles run sequentially, we just need them to run
# OFF the main thread so the main thread can enforce a hard timeout on them.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cycle")


def retry_pending_github_queue():
    """
    Reads the GitHub-hosted pending-projects queue (see github_fallback.py)
    and attempts to re-evaluate each entry with Gemini. This is what
    actually implements "automatically re-evaluate once quota resets" —
    called at the start of every cycle, before scraping new projects, so
    a backlog gets priority attention.

    - Success (AI call works this time): processed exactly like a normal
      new-project evaluation — notified via Telegram if it clears
      MATCH_THRESHOLD, logged either way — then removed from the queue.
    - Still failing (still ai_failed): left in the queue for next cycle,
      with retry_count incremented. No GitHub write happens for this case
      alone (to avoid a wasted API call/write every cycle for an entry
      that's simply still waiting) — only entries that change state
      (succeed, or exceed the retry cap) trigger a queue file update.
    - Exceeds config.GITHUB_QUEUE_MAX_RETRIES: dropped from the auto-retry
      queue (it's still preserved in the human-readable Issue/file record
      created when it was first queued) rather than retried forever.
    """
    pending = github_fallback.load_pending_queue()
    if not pending:
        return

    logger.info("Found %s pending project(s) in the GitHub retry queue", len(pending))
    updated_queue = []
    queue_changed = False

    for entry in pending:
        title = entry.get("title", "?")
        try:
            evaluation = ai_agent.evaluate_project(
                title=title,
                description=entry.get("description", ""),
                budget=entry.get("budget"),
                tags=entry.get("tags") or [],
            )
        except Exception:
            logger.error(
                "Unexpected error re-evaluating pending project '%s':\n%s",
                title, traceback.format_exc(),
            )
            evaluation = None

        if evaluation is None or evaluation.ai_failed:
            retry_count = entry.get("retry_count", 0) + 1
            if retry_count > config.GITHUB_QUEUE_MAX_RETRIES:
                logger.warning(
                    "Pending project '%s' exceeded GITHUB_QUEUE_MAX_RETRIES (%s) — "
                    "dropping from auto-retry queue (still on record in the original "
                    "GitHub issue/file)",
                    title, config.GITHUB_QUEUE_MAX_RETRIES,
                )
                queue_changed = True  # dropped -> queue file needs updating
                continue
            entry["retry_count"] = retry_count
            logger.info("Pending project '%s' still unavailable — will retry next cycle (attempt %s)", title, retry_count)
            updated_queue.append(entry)
            continue

        # Success: AI evaluation worked this time.
        queue_changed = True
        logger.info(
            "Pending project '%s' successfully re-evaluated after quota reset: %.0f%%",
            title, evaluation.match_score,
        )
        if evaluation.match_score >= config.MATCH_THRESHOLD and evaluation.proposal_ar:
            notifier.notify_matched_project(
                title=title,
                url=entry.get("url", ""),
                score=evaluation.match_score,
                proposal_ar=evaluation.proposal_ar,
                budget=entry.get("budget"),
                suggested_price=evaluation.suggested_price,
                delivery_days=evaluation.delivery_days,
                client_warning=entry.get("client_warning"),
            )
        else:
            logger.info(
                "Pending project '%s' scored below threshold after re-evaluation "
                "(%.0f%%, threshold %.0f%%) — no notification",
                title, evaluation.match_score, config.MATCH_THRESHOLD,
            )
        # Either way it's now been properly evaluated, so it's removed from
        # the queue by simply not being appended to updated_queue.

    if queue_changed:
        github_fallback.save_pending_queue(
            updated_queue,
            message=f"Update pending-projects queue ({len(updated_queue)} still pending)",
        )


def run_cycle():
    """One full monitor -> evaluate -> notify pass. Never raises."""
    retry_pending_github_queue()

    new_projects = scraper.get_new_projects()
    logger.info("Cycle fetched %s new project(s) to evaluate", len(new_projects))

    for project in new_projects:
        try:
            logger.info("Evaluating project: %s (tags: %s)", project.title, project.tags)
            evaluation = ai_agent.evaluate_project(
                title=project.title,
                description=project.description,
                budget=project.budget,
                tags=project.tags,
            )

            logger.info(
                "Match score for '%s': %.0f%% (%s) | suggested price: %s | delivery: %s day(s)",
                project.title, evaluation.match_score, evaluation.reasoning,
                evaluation.suggested_price, evaluation.delivery_days,
            )

            # AI call itself failed (e.g. 429 RESOURCE_EXHAUSTED on every key
            # in GEMINI_API_KEYS) — distinct from a successful call that just
            # scored the project low. Per requirements:
            #   1. No local storage — raw project data goes straight to
            #      GitHub, both as a human-readable record (issue/file) AND
            #      as a structured queue entry that retry_pending_github_queue()
            #      reads back and re-evaluates automatically next cycle.
            #   2. Instant Telegram alert with the raw scraped fields, since
            #      there's no AI score/proposal to report yet.
            #   3. Same inline "open project" button as a successful match.
            if evaluation.ai_failed:
                notifier.notify_pending_project(
                    title=project.title,
                    url=project.url,
                    budget=project.budget,
                    duration=project.duration,
                    description=project.description,
                )
                saved_record = github_fallback.save_project_to_github(
                    project,
                    reason=f"Gemini API call failed: {evaluation.reasoning}",
                )
                queued = github_fallback.queue_project(
                    project,
                    reason=f"Gemini API call failed: {evaluation.reasoning}",
                )
                logger.warning(
                    "AI evaluation unavailable for '%s' — record saved: %s | queued for auto-retry: %s",
                    project.title, saved_record, queued,
                )
                continue

            # >= : must match ai_agent.py's evaluate_project() comparison
            # exactly (score >= config.MATCH_THRESHOLD) — that's the gate
            # that actually decides whether proposal_ar gets set at all, so
            # using a different operator here would be meaningless (a
            # boundary-score project would already have no proposal to send).
            if evaluation.match_score >= config.MATCH_THRESHOLD and evaluation.proposal_ar:
                notifier.notify_matched_project(
                    title=project.title,
                    url=project.url,
                    score=evaluation.match_score,
                    proposal_ar=evaluation.proposal_ar,
                    budget=project.budget,
                    suggested_price=evaluation.suggested_price,
                    delivery_days=evaluation.delivery_days,
                    client_warning=project.client_warning,
                )
            else:
                # Bug fix: this used to log config.MATCH_THRESHOLD (the
                # threshold itself) instead of the project's actual score,
                # so EVERY skipped project's log line showed the threshold
                # value regardless of what it actually scored — which is
                # exactly what made a real threshold/rounding bug look like
                # this in the first place. Now logs the real score.
                logger.info(
                    "Below threshold (project scored %.0f%%, threshold is %.0f%%) — skipping notification",
                    evaluation.match_score, config.MATCH_THRESHOLD,
                )

        except Exception:
            # Per-project isolation: one bad project must not stop the batch.
            logger.error(
                "Error processing project '%s':\n%s",
                getattr(project, "title", "?"),
                traceback.format_exc(),
            )
            continue


def run_cycle_with_watchdog() -> bool:
    """
    Runs run_cycle() on a background thread and enforces a hard wall-clock
    timeout from the main thread. Returns True if the cycle completed
    normally within the timeout, False otherwise. The main loop NEVER
    blocks longer than config.CYCLE_TIMEOUT seconds here, regardless of
    what happens inside run_cycle (network hang, library bug, etc.).
    """
    future = _executor.submit(run_cycle)
    try:
        future.result(timeout=config.CYCLE_TIMEOUT)
        return True
    except concurrent.futures.TimeoutError:
        logger.error(
            "Cycle exceeded CYCLE_TIMEOUT (%ss) and was abandoned. "
            "The stuck call will keep running in the background and its "
            "thread will be discarded; the main loop is moving on.",
            config.CYCLE_TIMEOUT,
        )
        return False
    except Exception:
        logger.error("Cycle raised an unexpected exception:\n%s", traceback.format_exc())
        return False


def safe_sleep(total_seconds: int, chunk_seconds: int = 60):
    """
    Sleeps in small chunks instead of one long blocking call, logging a
    heartbeat each chunk. This makes it immediately obvious in the Render
    logs whether the process is alive-and-sleeping vs. actually dead —
    and keeps the sleep itself simple and interruption-safe.
    """
    remaining = total_seconds
    while remaining > 0:
        this_chunk = min(chunk_seconds, remaining)
        time.sleep(this_chunk)
        remaining -= this_chunk
        if remaining > 0:
            logger.info("...still sleeping (%ss remaining until next cycle)", remaining)


def main():
    logger.info("Starting Mostaql AI Freelance Assistant worker...")
    logger.info(
        "Poll interval: %s-%s s | Match threshold: %s%% | Cycle watchdog: %ss | Gemini timeout: %ss",
        config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX,
        config.MATCH_THRESHOLD, config.CYCLE_TIMEOUT, config.GEMINI_TIMEOUT,
    )

    # Health-check server for Render's Web Service port scan — runs
    # independently on a daemon thread so it can never be blocked by the
    # bot loop, and vice versa.
    health_server.start_health_server_in_background()

    consecutive_failures = 0

    while True:
        try:
            success = run_cycle_with_watchdog()
            consecutive_failures = 0 if success else consecutive_failures + 1
        except Exception:
            # Absolute outer safety net — the process must survive this no
            # matter what. Full traceback always logged.
            consecutive_failures += 1
            logger.error(
                "Unhandled error in main loop (failure #%s):\n%s",
                consecutive_failures, traceback.format_exc(),
            )

        if consecutive_failures and consecutive_failures % 5 == 0:
            try:
                notifier.notify_error(
                    "main loop",
                    f"{consecutive_failures} consecutive failed/timed-out cycles.",
                )
            except Exception:
                logger.error("Failed to send error notification:\n%s", traceback.format_exc())

        sleep_seconds = random.randint(config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX)
        logger.info("Cycle complete. Sleeping for %s seconds...", sleep_seconds)
        safe_sleep(sleep_seconds)


if __name__ == "__main__":
    main()
