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
    Drains `task_queue` in BATCHES (see _drain_batch, up to
    config.GEMINI_SCORE_BATCH_SIZE projects grouped into one scoring call
    via ai_agent.evaluate_projects_batch) rather than one project per
    Gemini call — the single biggest lever for staying inside a free-tier
    daily request quota (RPD) when several new projects appear in the same
    poll cycle. Proposal drafting still happens one call per accepted
    match (see evaluate_projects_batch's docstring for why). Also respects
    ai_agent's PROACTIVE local rate limiter (see ai_agent.KeyRateLimiter)
    — if every key is at its local RPM cap, the Gemini call is skipped
    entirely (zero requests sent) and the batch goes straight to the
    GitHub fallback, rather than firing a request we can already tell
    would be rejected. Also periodically re-processes the GitHub-hosted
    retry queue/issues (still Gemini-bound work, so it belongs on this
    thread, not the producer's) — those stay one-at-a-time, since the
    retry backlog's volume/burstiness doesn't justify batching complexity.

Because they're on separate threads, ANY blocking on the consumer side
(a retryDelay sleep, a slow response, being fully rate-limited) has zero
effect on the producer's ability to keep scraping and queuing new projects.

Stability principles carried over from the previous single-loop version:
- Every loop iteration is wrapped in try/except so one bad iteration can
  never kill its thread.
- Each batch of projects runs under its own watchdog timeout (reusing the
  config.CYCLE_TIMEOUT env var — see process_batch_with_watchdog) so a
  single hung Gemini call can't stall the consumer thread forever; it just
  gets abandoned and treated as unavailable (routed to the GitHub
  fallback) so the queue keeps draining.
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
import json
import logging
import queue
import random
import sys
import threading
import time
import traceback

import requests

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
import outcome_tracker
import repost_detector

# Thread-safe hand-off between the producer and consumer. Bounded so a
# consumer that falls far behind (e.g. heavily rate-limited) applies mild
# backpressure to the producer via a blocking put() rather than growing
# unbounded in memory — acceptable for Mostaql's realistic project volume.
task_queue = queue.Queue(maxsize=config.TASK_QUEUE_MAXSIZE)

# Single-worker executor used ONLY to enforce a hard per-batch timeout
# around each Gemini evaluation batch — see process_batch_with_watchdog.
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
                client_info=entry.get("client_info"),
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
        if evaluation.match_score >= evaluation.effective_threshold and evaluation.proposal_ar:
            repost_warning = repost_detector.check_and_record(
                entry.get("id"), title, entry.get("description", ""),
            )
            notifier.notify_matched_project(
                title=title,
                url=entry.get("url", ""),
                score=evaluation.match_score,
                proposal_ar=evaluation.proposal_ar,
                budget=entry.get("budget"),
                suggested_price=evaluation.suggested_price,
                delivery_days=evaluation.delivery_days,
                client_warning=entry.get("client_warning"),
                project_id=entry.get("id"),
                matched_skills=evaluation.matched_skills,
                missing_skills=evaluation.missing_skills,
                repost_warning=repost_warning,
            )
            sent_to_telegram = True
        else:
            logger.info(
                "Pending project '%s' scored below threshold after re-evaluation "
                "(%.0f%%, threshold %.0f%%) — no notification",
                title, evaluation.match_score, evaluation.effective_threshold,
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
        if evaluation.match_score >= evaluation.effective_threshold and evaluation.proposal_ar:
            synthetic_id = f"issue-{issue_number}"
            repost_warning = repost_detector.check_and_record(synthetic_id, title, parsed.get("description", ""))
            notifier.notify_matched_project(
                title=title,
                url=parsed.get("url") or issue.get("html_url", ""),
                score=evaluation.match_score,
                proposal_ar=evaluation.proposal_ar,
                budget=parsed.get("budget"),
                suggested_price=evaluation.suggested_price,
                delivery_days=evaluation.delivery_days,
                # No real project.id survives in a GitHub Issue body — the
                # issue number is still a stable, unique identifier for
                # outcome tracking AND repost-detection purposes.
                project_id=synthetic_id,
                matched_skills=evaluation.matched_skills,
                missing_skills=evaluation.missing_skills,
                repost_warning=repost_warning,
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


def _handle_evaluation_result(project, evaluation):
    """
    Shared tail logic for ONE project's already-computed Evaluation —
    identical regardless of whether the scoring call that produced it was
    a single-project call or one entry out of a batched call. Decides
    whether to notify Telegram, logs the outcome, records analytics, and
    routes to the GitHub fallback if the AI call failed for this specific
    project (which now includes a PROACTIVE rate-limit bypass via
    ai_agent's KeyRateLimiter, not just reactive 429s). Never raises.
    """
    logger.info(
        "Match score for '%s': %.0f%% (%s) | matched: %s | missing: %s | "
        "suggested price: %s | delivery: %s day(s)",
        project.title, evaluation.match_score, evaluation.reasoning,
        evaluation.matched_skills or "-", evaluation.missing_skills or "-",
        evaluation.suggested_price, evaluation.delivery_days,
    )

    if evaluation.ai_failed:
        ai_agent.record_token_usage(project.title, evaluation, sent_to_telegram=False)
        handle_ai_unavailable(project, reason=f"Gemini API call failed: {evaluation.reasoning}")
        return

    # >= : must match ai_agent.py's evaluate_project()/evaluate_projects_batch()
    # comparison exactly. Compares against evaluation.effective_threshold
    # (not the static config.MATCH_THRESHOLD) since that reflects whatever
    # threshold was ACTUALLY used to decide whether to draft a proposal —
    # which may be higher under quota pressure (see
    # config.ADAPTIVE_THRESHOLD_ENABLED / ai_agent.get_effective_match_threshold).
    sent_to_telegram = False
    if evaluation.match_score >= evaluation.effective_threshold and evaluation.proposal_ar:
        repost_warning = repost_detector.check_and_record(project.id, project.title, project.description)
        notifier.notify_matched_project(
            title=project.title,
            url=project.url,
            score=evaluation.match_score,
            proposal_ar=evaluation.proposal_ar,
            budget=project.budget,
            suggested_price=evaluation.suggested_price,
            delivery_days=evaluation.delivery_days,
            client_warning=project.client_warning,
            project_id=project.id,
            matched_skills=evaluation.matched_skills,
            missing_skills=evaluation.missing_skills,
            repost_warning=repost_warning,
        )
        sent_to_telegram = True
    else:
        logger.info(
            "Below threshold (project scored %.0f%%, effective threshold is %.0f%%) — skipping notification",
            evaluation.match_score, evaluation.effective_threshold,
        )

    ai_agent.record_token_usage(project.title, evaluation, sent_to_telegram=sent_to_telegram)


def process_project_batch(projects):
    """
    Evaluates a LIST of projects together: scored in ONE Gemini call via
    ai_agent.evaluate_projects_batch (see its docstring for why proposal
    drafting stays per-project), then each result is handled individually
    via _handle_evaluation_result — notify/log/record/fallback, exactly
    like the old one-project-at-a-time path. A batch of size 1 behaves
    identically to the previous single-project flow.
    """
    logger.info(
        "Evaluating batch of %s project(s): %s",
        len(projects), ", ".join(p.title for p in projects),
    )
    batch_input = [
        {"title": p.title, "description": p.description, "budget": p.budget, "tags": p.tags, "client_info": p.client_info}
        for p in projects
    ]
    evaluations = ai_agent.evaluate_projects_batch(batch_input)

    for project, evaluation in zip(projects, evaluations):
        try:
            _handle_evaluation_result(project, evaluation)
        except Exception:
            logger.error(
                "Unexpected error handling evaluation result for '%s':\n%s",
                project.title, traceback.format_exc(),
            )


def process_batch_with_watchdog(projects):
    """
    Runs process_project_batch() under a hard wall-clock timeout
    (config.CYCLE_TIMEOUT — name kept for backward compatibility with
    existing deployments' env vars). If it times out, EVERY project in the
    batch is treated as AI-unavailable (routed to the GitHub fallback)
    rather than silently disappearing, and the consumer thread moves on
    instead of stalling forever.

    Trade-off vs. the old per-project watchdog: a stuck call now risks
    abandoning up to GEMINI_SCORE_BATCH_SIZE projects at once instead of
    just one. Kept acceptable by the default batch size being modest (5)
    and CYCLE_TIMEOUT being generous (600s default) — and it's the same
    trade batching itself makes (fewer, larger calls) applied consistently
    to the failure path too.
    """
    future = _eval_executor.submit(process_project_batch, projects)
    try:
        future.result(timeout=config.CYCLE_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.error(
            "Batch evaluation of %s project(s) exceeded the %ss task timeout "
            "and was abandoned — treating all of them as AI-unavailable so "
            "the consumer isn't stalled; the stuck call keeps running in the "
            "background and its thread is discarded.",
            len(projects), config.CYCLE_TIMEOUT,
        )
        try:
            for project in projects:
                handle_ai_unavailable(project, reason=f"Batch evaluation task exceeded {config.CYCLE_TIMEOUT}s timeout")
        except Exception:
            logger.error("Fallback handling after batch timeout also failed:\n%s", traceback.format_exc())
    except Exception:
        logger.error(
            "Unexpected error while processing a batch of %s project(s):\n%s",
            len(projects), traceback.format_exc(),
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
        health_server.update_status(producer_last_heartbeat=time.time())
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
        health_server.update_status(consumer_last_heartbeat=time.time(), queue_size=task_queue.qsize())
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
            batch = _drain_batch(config.GEMINI_SCORE_BATCH_SIZE, first_item_timeout=10)
        except Exception:
            logger.error("Unexpected error draining the task queue:\n%s", traceback.format_exc())
            continue

        if not batch:
            continue

        try:
            process_batch_with_watchdog(batch)
        except Exception:
            # Should be unreachable (process_batch_with_watchdog already
            # catches everything internally), but this is the absolute
            # last line of defense so the consumer thread can never die.
            logger.error(
                "Unhandled error processing a batch of %s project(s) from the queue:\n%s",
                len(batch), traceback.format_exc(),
            )
        finally:
            for _ in batch:
                task_queue.task_done()


def _drain_batch(max_size: int, first_item_timeout: int = 10, max_wait_seconds: int = None):
    """
    Pulls up to `max_size` items off task_queue, grouping a burst of
    newly-scraped projects into one batch for scoring. Blocks up to
    `first_item_timeout` seconds waiting for the FIRST item — same cadence
    as before, so the consumer still wakes regularly to recheck the GitHub
    backlog when the queue is empty. Once at least one item is in hand, it
    keeps greedily grabbing more (without re-blocking the full timeout)
    until either `max_size` is reached or `max_wait_seconds` has elapsed
    since the first item arrived — whichever comes first — so a slow
    trickle of projects doesn't wait indefinitely for a full batch to form.
    Returns [] if nothing arrived within `first_item_timeout`.
    """
    if max_wait_seconds is None:
        max_wait_seconds = config.GEMINI_BATCH_MAX_WAIT_SECONDS

    try:
        first = task_queue.get(timeout=first_item_timeout)
    except queue.Empty:
        return []

    batch = [first]
    deadline = time.time() + max_wait_seconds
    while len(batch) < max_size:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            batch.append(task_queue.get(timeout=remaining))
        except queue.Empty:
            break
    return batch


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


# ---------------------------------------------------------------------------
# Telegram outcome-feedback listener (Won/Lost buttons)
# ---------------------------------------------------------------------------
# Runs on its OWN daemon thread, entirely separate from the producer/
# consumer — it never touches task_queue or Gemini, so it can't be blocked
# by (or block) anything else in the pipeline. See outcome_tracker.py's
# docstring for what this data is for.

_TELEGRAM_OFFSET_FILE = "telegram_update_offset.txt"


def _load_telegram_offset() -> int:
    """Reads back the last-processed Telegram update_id + 1, so a restart
    doesn't reprocess (and double-toast) already-handled button taps.
    Returns 0 (process from whatever Telegram currently has queued) on
    ANY failure — missing file, corrupt content, or anything else — never
    raises, matching the fail-safe convention used by every other
    persistence helper in this codebase (ScoreCache, DailyRequestTracker,
    outcome_tracker, repost_detector)."""
    try:
        with open(_TELEGRAM_OFFSET_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def _save_telegram_offset(offset: int):
    """Best-effort persistence — if this write fails for ANY reason, worst
    case is a restart re-processes a few already-handled taps (each is
    idempotent via outcome_tracker.record_outcome's overwrite semantics),
    not a crash. Catches broad Exception rather than just OSError, same
    fail-safe convention as every other persistence helper in this
    codebase (ScoreCache, DailyRequestTracker, outcome_tracker,
    repost_detector)."""
    try:
        with open(_TELEGRAM_OFFSET_FILE, "w", encoding="utf-8") as f:
            f.write(str(offset))
    except Exception:
        pass


def _answer_telegram_callback(callback_id: str, text: str):
    """Acknowledges a callback_query with a small toast notification —
    required by Telegram regardless of outcome (an unanswered
    callback_query leaves the tapped button showing an infinite loading
    spinner on the user's phone until it times out client-side)."""
    if not callback_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/answerCallbackQuery",
            data={"callback_query_id": callback_id, "text": text},
            timeout=config.REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to answer a Telegram callback query: %s", exc)


def _handle_feedback_callback(callback: dict):
    """
    Parses one callback_query update from a Won/Lost button tap, records
    the outcome, and answers it with a confirmation toast. Never raises —
    a malformed or unexpected callback is just logged and acknowledged
    with a generic notice rather than crashing the listener thread.
    """
    callback_id = callback.get("id")
    try:
        data = callback.get("data", "") or ""
        action, _, project_id = data.partition(":")
        if action not in ("won", "lost") or not project_id:
            logger.warning("Ignoring unrecognized Telegram callback_data: %r", data)
            _answer_telegram_callback(callback_id, "⚠️ إجراء غير معروف")
            return

        # Best-effort only: pulls the project title back out of the
        # notification message's own text (the "📌 *العنوان:* ..." line
        # notifier.build_message() already wrote) purely for a nicer log
        # line / outcomes.json entry — recording still succeeds without it.
        title = None
        message_text = (callback.get("message") or {}).get("text") or ""
        for line in message_text.splitlines():
            if "العنوان" in line:
                title = line.split(":", 1)[-1].strip().strip("*").strip()
                break

        ok = outcome_tracker.record_outcome(project_id, title=title, outcome=action)
        if ok:
            confirmation = "✅ تم تسجيل الفوز بالمشروع — شكراً لتحديث النتيجة" if action == "won" else "📝 تم تسجيل عدم الفوز — شكراً لتحديث النتيجة"
        else:
            confirmation = "⚠️ تعذر حفظ النتيجة، حاول مرة أخرى"
        _answer_telegram_callback(callback_id, confirmation)
    except Exception:
        logger.error("Unexpected error handling a Telegram feedback callback:\n%s", traceback.format_exc())
        _answer_telegram_callback(callback_id, "⚠️ حدث خطأ غير متوقع")


def telegram_feedback_loop():
    """
    Long-polls Telegram's getUpdates for callback_query button taps from
    the Won/Lost buttons (see notifier.build_inline_keyboard). Uses
    Telegram's own server-side long-polling (the request blocks up to
    config.TELEGRAM_FEEDBACK_POLL_TIMEOUT seconds waiting for a new
    update, or returns immediately if one's already pending) rather than
    sleep-then-poll — near-instant button response without hammering the
    API between taps.

    Every iteration is wrapped in try/except (same stability principle as
    producer_loop/consumer_loop) so a transient network error can't kill
    this thread — it just waits a few seconds and retries.
    """
    logger.info("Telegram feedback listener starting (Won/Lost outcome-button tracking)")
    offset = _load_telegram_offset()

    while True:
        health_server.update_status(feedback_last_heartbeat=time.time())
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
                params={
                    "offset": offset,
                    "timeout": config.TELEGRAM_FEEDBACK_POLL_TIMEOUT,
                    # Telegram expects a JSON-encoded array here, same
                    # reason reply_markup gets json.dumps()'d in notifier.py
                    # — a raw Python list would be form-encoded wrong.
                    "allowed_updates": json.dumps(["callback_query"]),
                },
                # A bit of slack over Telegram's own long-poll timeout, so
                # a legitimately slow-but-successful long poll isn't
                # mistaken for a hung connection and aborted right as
                # Telegram was about to respond.
                timeout=config.TELEGRAM_FEEDBACK_POLL_TIMEOUT + 10,
            )
            if resp.status_code == 409:
                # Telegram allows only ONE active long-poll connection per
                # bot token — a 409 here means a second process (a
                # leftover local dev run, a duplicate Render service, or
                # the brief overlap between old/new instances during a
                # zero-downtime deploy) is also calling getUpdates with
                # the SAME token right now. Retrying fast doesn't help a
                # conflict resolve any sooner than it already will on its
                # own, so this backs off longer than a generic transient
                # error to avoid spamming logs with a warning every 5s
                # for something retrying won't fix.
                logger.warning(
                    "Telegram getUpdates got HTTP 409 Conflict — another process is "
                    "already long-polling this bot token. If this is happening during "
                    "a deploy, it should clear on its own within a normal deploy window "
                    "as the old instance shuts down. If it persists, check for a leftover "
                    "local run or a duplicate service using the same TELEGRAM_BOT_TOKEN.",
                )
                time.sleep(config.TELEGRAM_CONFLICT_BACKOFF_SECONDS)
                continue
            if resp.status_code != 200:
                logger.warning("Telegram getUpdates returned HTTP %s: %s", resp.status_code, resp.text[:200])
                time.sleep(5)
                continue

            updates = resp.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                callback = update.get("callback_query")
                if callback:
                    _handle_feedback_callback(callback)

            if updates:
                _save_telegram_offset(offset)
        except requests.exceptions.RequestException as exc:
            logger.warning("Telegram feedback listener request failed (will retry): %s", exc)
            time.sleep(5)
        except Exception:
            logger.error("Unexpected error in Telegram feedback listener:\n%s", traceback.format_exc())
            time.sleep(5)


def main():
    logger.info("Starting Mostaql AI Freelance Assistant worker (producer/consumer architecture)...")
    logger.info(
        "Match threshold: %s%% | Gemini timeout: %ss | Local RPM cap/key: %s | "
        "Task queue max size: %s | Score batch size: %s (max wait %ss)",
        config.MATCH_THRESHOLD, config.GEMINI_TIMEOUT,
        config.GEMINI_MAX_RPM_PER_KEY, config.TASK_QUEUE_MAXSIZE,
        config.GEMINI_SCORE_BATCH_SIZE, config.GEMINI_BATCH_MAX_WAIT_SECONDS,
    )

    # Health-check server for Render's Web Service port scan — independent
    # daemon thread, unaffected by either the producer or consumer.
    health_server.start_health_server_in_background()

    producer_thread = threading.Thread(target=producer_loop, name="producer", daemon=True)
    consumer_thread = threading.Thread(target=consumer_loop, name="consumer", daemon=True)
    feedback_thread = threading.Thread(target=telegram_feedback_loop, name="telegram-feedback", daemon=True)
    producer_thread.start()
    consumer_thread.start()
    feedback_thread.start()

    # The main thread's only job now is to stay alive and notice if any
    # worker thread unexpectedly dies (all three loops already catch every
    # exception internally per-iteration, so this should be rare — it'd
    # take something like an interpreter-level error to actually kill a
    # thread despite that). No auto-respawn: this alerts loudly via both
    # logs and Telegram rather than silently degrading, which is enough
    # given how unlikely it is to trigger in practice.
    alerted_dead_threads = set()
    while True:
        time.sleep(30)
        for t in (producer_thread, consumer_thread, feedback_thread):
            if not t.is_alive() and t.name not in alerted_dead_threads:
                alerted_dead_threads.add(t.name)
                logger.critical("Thread '%s' has died unexpectedly — it will NOT be auto-restarted.", t.name)
                try:
                    notifier.notify_error("main supervisor", f"Thread '{t.name}' died unexpectedly and was not restarted.")
                except Exception:
                    logger.error("Failed to send thread-death notification:\n%s", traceback.format_exc())


if __name__ == "__main__":
    main()
