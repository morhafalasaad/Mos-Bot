"""
main.py
-------
Entry point for the background worker. Designed to run 24/7 on a cloud
worker dyno (Render Background Worker, PythonAnywhere Always-On Task, or a
simple `nohup python main.py &` on a VPS).

Stability principles:
- The outer `while True` loop wraps EVERY cycle in a try/except so a single
  unexpected error (network blip, malformed HTML, API hiccup) never kills
  the process — it just logs and waits for the next cycle.
- All logging goes to stdout, which is what cloud platforms capture and
  show in their log dashboards.
- Sleep interval is randomized within a range to also help avoid
  predictable, bot-like request patterns.
"""

import logging
import random
import sys
import time

import config
import scraper
import ai_agent
import notifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


def run_cycle():
    """One full monitor -> evaluate -> notify pass. Never raises."""
    new_projects = scraper.get_new_projects()

    for project in new_projects:
        try:
            logger.info("Evaluating project: %s", project.title)
            evaluation = ai_agent.evaluate_project(
                title=project.title,
                description=project.description,
                budget=project.budget,
            )

            logger.info(
                "Match score for '%s': %.0f%% (%s)",
                project.title, evaluation.match_score, evaluation.reasoning,
            )

            if evaluation.match_score > config.MATCH_THRESHOLD and evaluation.proposal_ar:
                notifier.notify_matched_project(
                    title=project.title,
                    url=project.url,
                    score=evaluation.match_score,
                    proposal_ar=evaluation.proposal_ar,
                    budget=project.budget,
                )
            else:
                logger.info("Below threshold (%.0f%%) — skipping notification", config.MATCH_THRESHOLD)

        except Exception as exc:
            # Per-project isolation: one bad project must not stop the batch.
            logger.exception("Error processing project '%s': %s", getattr(project, "title", "?"), exc)
            continue


def main():
    logger.info("Starting Mostaql AI Freelance Assistant worker...")
    logger.info(
        "Poll interval: %s-%s seconds | Match threshold: %s%%",
        config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX, config.MATCH_THRESHOLD,
    )

    consecutive_failures = 0

    while True:
        try:
            run_cycle()
            consecutive_failures = 0
        except Exception as exc:
            # Absolute outer safety net — the process must survive this.
            consecutive_failures += 1
            logger.exception("Unhandled error in main loop (failure #%s): %s", consecutive_failures, exc)

            if consecutive_failures >= 5:
                try:
                    notifier.notify_error(
                        "main loop",
                        f"{consecutive_failures} consecutive failures. Last error: {exc}",
                    )
                except Exception:
                    pass  # even the error notification must not crash the loop

        sleep_seconds = random.randint(config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX)
        logger.info("Cycle complete. Sleeping for %s seconds...", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
