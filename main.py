"""
main.py
-------
Entry point for the background worker. Designed to run 24/7 on Render as a
Web Service (bot + dummy health-check server side by side).

ARCHITECTURE: Producer/Consumer (this version)
-------------------------------------------------------------------
Previously this was one sequential loop: scrape -> evaluate every new
project via Gemini (blocking on each one, including any retryDelay sleep)
-> sleep -> repeat. That meant a slow/rate-limited Gemini call directly
delayed the NEXT scrape, wasting the time between Mostaql polls.

Now there are two independent daemon threads sharing a thread-safe
queue.Queue:

  - PRODUCER (producer_loop): scrapes Mostaql on its own schedule, applies
    the local tag pre-filter (ai_agent.local_skill_prefilter — zero Gemini
    cost), and pushes projects that pass onto `task_queue`. It NEVER calls
    Gemini and can never be blocked by rate limiting, a retryDelay sleep,
    or a slow API response — it just keeps discovering and queuing work.

  - CONSUMER (consumer_loop): the only thread that ever touches Gemini.
    Pulls projects off `task_queue` and evaluates them one at a time,
    respecting ai_agent's PROACTIVE local rate limiter (see
    ai_agent.KeyRateLimiter) — if every key is at its local RPM cap, the
    Gemini call is skipped entirely (zero requests sent) and the project
    goes straight to the GitHub fallback, rather than firing a request we
    can already tell would be rejected. Also periodically re-processes the
    GitHub-hosted retry queue/issues (still Gemini-bound work, so it
    belongs on this thread, not the producer's).

Because they're on separate threads, ANY blocking on the consumer side
(a retryDelay sleep, a slow response, being fully rate-limited) has zero
effect on the producer's ability to keep scraping and queuing new projects.

Stability principles carried over from the previous single-loop version:
- Every loop iteration is wrapped in try/except so one bad iteration can
  never kill its thread.
- Each individual project evaluation runs under its own watchdog timeout
  (reusing the config.CYCLE_TIMEOUT env var — see
  process_project_with_watchdog) so a single hung Gemini call can't stall
  the consumer thread forever; it just gets abandoned and treated as
  unavailable (routed to the GitHub fallback) so the queue keeps draining.
- All logging is flushed immediately (FlushingStreamHandler) so Render's
  log dashboard shows activity from BOTH threads in real time.
"""

# ---------------------------------------------------------------------------
# CA certificate bundle: MUST be the very first executable code in this
# process, before importing anything that itself imports requests/urllib3/
# httpx (scraper, ai_agent, notifier, github_fallback, cloudscraper, and
# google-genai all pull those in transitively at import time). Setting
# these env vars only takes effect if done before those libraries
# initialize their default SSL context — hence "os" and "certifi" are the
# very first imports in the whole file, ahead of everything else below.
#
# Fixes SSLCertVerificationError ("self-signed certificate") seen on some
# container/buildpack images whose system CA bundle is missing, stale, or
# incomplete — certifi ships its own regularly-updated, known-good bundle,
# so pointing both requests (REQUESTS_CA_BUNDLE) and the stdlib ssl module
# (SSL_CERT_FILE, which urllib3/httpx's default context also honors) at it
# means every outbound HTTPS call in this process — Mostaql, Gemini,
# Telegram, GitHub — verifies against a bundle known to be current, instead
# of whatever the container image happens to ship.
# ---------------------------------------------------------------------------
import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import concurrent.futures
import logging
import queue
import random
import sys
import threading
import time
import traceback

import config  # safe to import first: no logging side effects at import time

# ---------------------------------------------------------------------------
# Logging setup: configured BEFORE importing scraper/ai_agent/etc, since
# those modules log at import time (e.g. "Gemini: N API key(s) configured",
# "Using cloudscraper session"). Configuring logging after importing them
# would silently drop those lines — the root logger has no handlers until
# basicConfig() runs. Also forces unbuffered / line-buffered stdout so
# Render's log dashboard shows lines in real time instead of delayed batches.
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
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    # Short, scannable format: time + level + message. Thread/module names
    # were dropped — nearly every message already states which part of the
    # app it's from ("Producer:", "Consumer:", "Gemini key #N", etc.), so
    # repeating that context in every line's prefix was pure noise.
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

import scraper
import ai_agent
import notifier
import health_server
import github_fallback

# Thread-safe hand-off between the producer and consumer. Bounded so a
# consumer that falls far behind (e.g. heavily rate-limited) applies mild
# backpressure to the producer via a blocking put() rather than growing
# unbounded in memory — acceptable for Mostaql's realistic project volume.
task_queue = queue.Queue(maxsize=config.TASK_QUEUE_MAXSIZE)

# Single-worker executor used ONLY to enforce a hard per-task timeout
# around individual Gemini evaluations — see process_project_with_watchdog.
_eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-watchdog")


# ---------------------------------------------------------------------------
# GitHub retry workers (unchanged in behavior from the previous version —
# still Gemini-bound work, so they now run from the CONSUMER thread instead
# of at the top of a "cycle").
# ---------------------------------------------------------------------------

def retry_pending_github_queue():
    """
    Reads the GitHub-hosted pending-projects queue (see github_fallback.py)
    and attempts to re-evaluate each entry with Gemini.

    - Success (AI call works this time): processed exactly like a normal
      new-project evaluation — notified via Telegram if it clears
      MATCH_THRESHOLD, logged either way — then removed from the queue.
    - Still failing (still ai_failed, including a proactive rate-limit
      bypass): left in the queue for next time, with retry_count
      incremented. No GitHub write happens for this case alone (avoids a
      wasted API call/write for an entry that's simply still waiting) —
      only entries that change state (succeed, or exceed the retry cap)
      trigger a queue file update.
    - Exceeds config.GITHUB_QUEUE_MAX_RETRIES: dropped from the auto-retry
      queue (still preserved in the human-readable Issue/file record
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
            if evaluation is not None:
                ai_agent.record_token_usage(title, evaluation, sent_to_telegram=False)
            retry_count = entry.get("retry_count", 0) + 1
            if retry_count > config.GITHUB_QUEUE_MAX_RETRIES:
                logger.warning(
                    "Pending project '%s' exceeded GITHUB_QUEUE_MAX_RETRIES (%s) — "
                    "dropping from auto-retry queue and closing its linked issue "
                    "(if any) as given-up-on rather than leaving it open forever",
                    title, config.GITHUB_QUEUE_MAX_RETRIES,
                )
                if entry.get("issue_number"):
                    github_fallback.close_issue(
                        entry["issue_number"],
                        comment=f"⚠️ تم التخلي عن إعادة المحاولة بعد تجاوز الحد الأقصى "
                                f"({config.GITHUB_QUEUE_MAX_RETRIES} محاولة) دون نجاح تقييم AI.",
                    )
                queue_changed = True  # dropped -> queue file needs updating
                continue
            entry["retry_count"] = retry_count
            logger.info("Pending project '%s' still unavailable — will retry next check (attempt %s)", title, retry_count)
            updated_queue.append(entry)
            continue

        # Success: AI evaluation worked this time.
        queue_changed = True
        logger.info(
            "Pending project '%s' successfully re-evaluated after quota reset: %.0f%%",
            title, evaluation.match_score,
        )
        sent_to_telegram = False
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
            sent_to_telegram = True
        else:
            logger.info(
                "Pending project '%s' scored below threshold after re-evaluation "
                "(%.0f%%, threshold %.0f%%) — no notification",
                title, evaluation.match_score, config.MATCH_THRESHOLD,
            )
        ai_agent.record_token_usage(title, evaluation, sent_to_telegram=sent_to_telegram)
        if entry.get("issue_number"):
            github_fallback.close_issue(
                entry["issue_number"],
                comment=f"✅ تم إعادة تقييم المشروع تلقائياً بعد تجدد حصة Gemini. "
                        f"نسبة التطابق: {evaluation.match_score:.0f}%",
            )
        # Either way it's now been properly evaluated, so it's removed from
        # the queue by simply not being appended to updated_queue.

    if queue_changed:
        github_fallback.save_pending_queue(
            updated_queue,
            message=f"Update pending-projects queue ({len(updated_queue)} still pending)",
        )


def retry_open_github_issues():
    """
    Explicit "check open GitHub Issues, parse, re-evaluate, close" worker,
    independent of the queue file above — a safety net for "orphaned"
    issues whose queue entry is missing for any reason (e.g. the queue
    write failed even though the issue creation succeeded).

    DEDUP: skips any issue number already tracked by an active queue entry
    (those are handled by retry_pending_github_queue() above, which closes
    their linked issue itself on success) — guarantees a project is never
    evaluated twice just because it's referenced by both mechanisms.
    """
    issues = github_fallback.list_open_fallback_issues()
    if not issues:
        return

    tracked_issue_numbers = {
        entry.get("issue_number")
        for entry in github_fallback.load_pending_queue()
        if entry.get("issue_number")
    }
    orphaned = [i for i in issues if i.get("number") not in tracked_issue_numbers]
    if not orphaned:
        return

    logger.info(
        "Found %s open GitHub issue(s) not tracked by the queue — attempting re-evaluation",
        len(orphaned),
    )

    for issue in orphaned:
        issue_number = issue.get("number")
        title = issue.get("title", "").replace(github_fallback.ISSUE_TITLE_PREFIX, "", 1).strip()
        parsed = github_fallback.parse_issue_body(issue.get("body", ""))

        try:
            evaluation = ai_agent.evaluate_project(
                title=title,
                description=parsed.get("description", ""),
                budget=parsed.get("budget"),
                tags=parsed.get("tags") or [],
            )
        except Exception:
            logger.error(
                "Unexpected error re-evaluating GitHub issue #%s ('%s'):\n%s",
                issue_number, title, traceback.format_exc(),
            )
            continue

        if evaluation.ai_failed:
            ai_agent.record_token_usage(title, evaluation, sent_to_telegram=False)
            logger.info("GitHub issue #%s ('%s') still unavailable for re-evaluation — left open", issue_number, title)
            continue

        logger.info(
            "GitHub issue #%s ('%s') successfully re-evaluated: %.0f%%",
            issue_number, title, evaluation.match_score,
        )
        sent_to_telegram = False
        if evaluation.match_score >= config.MATCH_THRESHOLD and evaluation.proposal_ar:
            notifier.notify_matched_project(
                title=title,
                url=parsed.get("url") or issue.get("html_url", ""),
                score=evaluation.match_score,
                proposal_ar=evaluation.proposal_ar,
                budget=parsed.get("budget"),
                suggested_price=evaluation.suggested_price,
                delivery_days=evaluation.delivery_days,
            )
            sent_to_telegram = True
        ai_agent.record_token_usage(title, evaluation, sent_to_telegram=sent_to_telegram)
        github_fallback.close_issue(
            issue_number,
            comment=f"✅ تم إعادة تقييم المشروع تلقائياً بعد تجدد حصة Gemini. "
                    f"نسبة التطابق: {evaluation.match_score:.0f}%",
        )


# ---------------------------------------------------------------------------
# Per-project processing (consumer side)
# ---------------------------------------------------------------------------

def handle_ai_unavailable(project, reason: str):
    """
    Shared by both the normal 'AI call failed' path and the watchdog-timeout
    path below — anywhere Gemini didn't produce a usable result for a
    project, this is the single place that: sends the instant Telegram
    alert, saves the human-readable GitHub record, and queues it for
    automatic re-evaluation. Never raises.
    """
    notifier.notify_pending_project(
        title=project.title,
        url=project.url,
        budget=project.budget,
        duration=project.duration,
        description=project.description,
    )
    saved_record, issue_number = github_fallback.save_project_to_github(project, reason=reason)
    queued = github_fallback.queue_project(project, reason=reason, issue_number=issue_number)
    logger.warning(
        "AI evaluation unavailable for '%s' — record saved: %s (issue #%s) | queued for auto-retry: %s",
        project.title, saved_record, issue_number, queued,
    )


def process_project(project):
    """
    Evaluates ONE project and handles the result — success (notify if it
    clears threshold), below threshold (log only), or AI-unavailable
    (handle_ai_unavailable, which now includes a PROACTIVE rate-limit
    bypass via ai_agent's KeyRateLimiter, not just reactive 429s).
    """
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

    if evaluation.ai_failed:
        ai_agent.record_token_usage(project.title, evaluation, sent_to_telegram=False)
        handle_ai_unavailable(project, reason=f"Gemini API call failed: {evaluation.reasoning}")
        return

    # >= : must match ai_agent.py's evaluate_project() comparison exactly
    # (score >= config.MATCH_THRESHOLD) — that's the gate that actually
    # decides whether proposal_ar gets set at all.
    sent_to_telegram = False
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
        sent_to_telegram = True
    else:
        logger.info(
            "Below threshold (project scored %.0f%%, threshold is %.0f%%) — skipping notification",
            evaluation.match_score, config.MATCH_THRESHOLD,
        )

    ai_agent.record_token_usage(project.title, evaluation, sent_to_telegram=sent_to_telegram)


def process_project_with_watchdog(project):
    """
    Runs process_project() under a hard wall-clock timeout
    (config.CYCLE_TIMEOUT — name kept for backward compatibility with
    existing deployments' env vars, but its role is now "max seconds for a
    SINGLE evaluation task" rather than a whole batch). If it times out,
    the project is treated as AI-unavailable (routed to the GitHub
    fallback) rather than silently disappearing, and the consumer thread
    moves on to the next queued item instead of stalling forever.
    """
    future = _eval_executor.submit(process_project, project)
    try:
        future.result(timeout=config.CYCLE_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.error(
            "Evaluation of '%s' exceeded the %ss task timeout and was "
            "abandoned — treating as AI-unavailable so the consumer isn't "
            "stalled; the stuck call keeps running in the background and "
            "its thread is discarded.",
            project.title, config.CYCLE_TIMEOUT,
        )
        try:
            handle_ai_unavailable(project, reason=f"Evaluation task exceeded {config.CYCLE_TIMEOUT}s timeout")
        except Exception:
            logger.error("Fallback handling after timeout also failed:\n%s", traceback.format_exc())
    except Exception:
        logger.error(
            "Unexpected error while processing '%s':\n%s",
            project.title, traceback.format_exc(),
        )


# ---------------------------------------------------------------------------
# Producer: scrapes and enqueues, NEVER touches Gemini, can never be
# blocked by rate limiting or a slow API call.
# ---------------------------------------------------------------------------

def producer_loop():
    logger.info(
        "Producer starting — poll interval %s-%ss",
        config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX,
    )
    while True:
        try:
            new_projects = scraper.get_new_projects()
            logger.info("Producer: found %s new project(s) this scrape", len(new_projects))

            for project in new_projects:
                # Local tag pre-filter here (zero Gemini cost) — matches
                # the "Scraper applies local filters, and if a project
                # passes, puts it into the queue" requirement. Fail-open:
                # empty/missing tags still get queued (ai_agent's own
                # prefilter check inside evaluate_project applies the same
                # fail-open rule, so this is consistent either way).
                if project.tags and not ai_agent.local_skill_prefilter(project.tags):
                    logger.info(
                        "Producer: local pre-filter skip '%s' (tags: %s) — zero Gemini cost",
                        project.title, project.tags,
                    )
                    continue

                task_queue.put(project)  # blocks briefly if the queue is full (backpressure)
                logger.info(
                    "Producer: enqueued '%s' for evaluation (queue size ~%s)",
                    project.title, task_queue.qsize(),
                )

        except Exception:
            logger.error("Producer loop iteration failed:\n%s", traceback.format_exc())

        sleep_seconds = random.randint(config.POLL_INTERVAL_MIN, config.POLL_INTERVAL_MAX)
        logger.info("Producer: scrape complete, sleeping %ss before next scrape", sleep_seconds)
        safe_sleep(sleep_seconds)


# ---------------------------------------------------------------------------
# Consumer: the only thread that ever calls Gemini. Drains task_queue,
# respecting ai_agent's proactive rate limiter, and periodically re-checks
# the GitHub retry backlog.
# ---------------------------------------------------------------------------

def consumer_loop():
    logger.info(
        "Consumer starting — GitHub retry backlog checked every ~%ss, "
        "per-task timeout %ss",
        config.GITHUB_RETRY_CHECK_INTERVAL, config.CYCLE_TIMEOUT,
    )
    last_github_retry_check = 0.0
    last_token_stats_sync = 0.0

    while True:
        try:
            now = time.time()
            if now - last_github_retry_check >= config.GITHUB_RETRY_CHECK_INTERVAL:
                retry_pending_github_queue()
                retry_open_github_issues()
                last_github_retry_check = now

            # Sync token_usage_stats.json to GitHub once per this interval —
            # placed right after the retry-backlog check ("after evaluating
            # pending items") and before the queue-drain/wait below acts as
            # this architecture's closest analog to "before the sleep
            # phase" (there's no single discrete scrape-evaluate-sleep
            # cycle anymore now that scraping and evaluating run on
            # independent threads — see the module docstring). Intentionally
            # NOT wrapped in its own logger.info/error — sync_token_stats_to_github()
            # is silent by design and this call site stays silent too, per
            # the "no console output about GitHub syncing" requirement. The
            # bare except is defense-in-depth only; the function itself
            # already never raises.
            if now - last_token_stats_sync >= config.TOKEN_STATS_SYNC_INTERVAL:
                try:
                    github_fallback.sync_token_stats_to_github()
                except Exception:
                    pass
                last_token_stats_sync = now
        except Exception:
            logger.error("Consumer's GitHub retry check failed:\n%s", traceback.format_exc())

        try:
            # Blocks up to a modest timeout so the loop wakes regularly to
            # re-check the GitHub backlog even when task_queue is empty,
            # without busy-looping.
            project = task_queue.get(timeout=10)
        except queue.Empty:
            continue

        try:
            process_project_with_watchdog(project)
        except Exception:
            # Should be unreachable (process_project_with_watchdog already
            # catches everything internally), but this is the absolute
            # last line of defense so the consumer thread can never die.
            logger.error(
                "Unhandled error processing '%s' from the queue:\n%s",
                getattr(project, "title", "?"), traceback.format_exc(),
            )
        finally:
            task_queue.task_done()


def safe_sleep(total_seconds: int, chunk_seconds: int = 60):
    """
    Sleeps in small chunks instead of one long blocking call, logging a
    heartbeat each chunk — makes it obvious in the logs whether a thread is
    alive-and-sleeping vs. actually dead.
    """
    remaining = total_seconds
    while remaining > 0:
        this_chunk = min(chunk_seconds, remaining)
        time.sleep(this_chunk)
        remaining -= this_chunk
        if remaining > 0:
            logger.debug("...still sleeping (%ss remaining)", remaining)


def main():
    logger.info("Starting Mostaql AI Freelance Assistant worker (producer/consumer architecture)...")
    logger.info(
        "Match threshold: %s%% | Gemini timeout: %ss | Local RPM cap/key: %s | Task queue max size: %s",
        config.MATCH_THRESHOLD, config.GEMINI_TIMEOUT,
        config.GEMINI_MAX_RPM_PER_KEY, config.TASK_QUEUE_MAXSIZE,
    )

    # Health-check server for Render's Web Service port scan — independent
    # daemon thread, unaffected by either the producer or consumer.
    health_server.start_health_server_in_background()

    producer_thread = threading.Thread(target=producer_loop, name="producer", daemon=True)
    consumer_thread = threading.Thread(target=consumer_loop, name="consumer", daemon=True)
    producer_thread.start()
    consumer_thread.start()

    # The main thread's only job now is to stay alive and notice if either
    # worker thread unexpectedly dies (both loops already catch every
    # exception internally per-iteration, so this should be rare — it'd
    # take something like an interpreter-level error to actually kill a
    # thread despite that). No auto-respawn: this alerts loudly via both
    # logs and Telegram rather than silently degrading, which is enough
    # given how unlikely it is to trigger in practice.
    alerted_dead_threads = set()
    while True:
        time.sleep(30)
        for t in (producer_thread, consumer_thread):
            if not t.is_alive() and t.name not in alerted_dead_threads:
                alerted_dead_threads.add(t.name)
                logger.critical("Thread '%s' has died unexpectedly — it will NOT be auto-restarted.", t.name)
                try:
                    notifier.notify_error("main supervisor", f"Thread '{t.name}' died unexpectedly and was not restarted.")
                except Exception:
                    logger.error("Failed to send thread-death notification:\n%s", traceback.format_exc())


if __name__ == "__main__":
    main()
