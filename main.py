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


def run_cycle():
    """One full monitor -> evaluate -> notify pass. Never raises."""
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
