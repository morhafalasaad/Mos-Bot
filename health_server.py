"""
health_server.py
-----------------
Render's "Web Service" type requires the process to bind to $PORT and
respond to HTTP requests, or Render considers the deploy failed / unhealthy
and may restart or kill it. Since the actual bot logic is a background
polling loop (not a web app), this module runs a tiny stdlib HTTP server
in its own daemon thread purely to satisfy that health check.

Running it in a background thread (not the main thread) is critical: if the
dummy server were started with `serve_forever()` on the main thread before
the bot loop, the bot loop would never run. If it were started AFTER a
blocking bot loop, Render's port scan would time out during boot. Doing it
in a daemon thread lets both run concurrently and independently — a hang in
one cannot block the other.

Two endpoints:
  - `/` (or anything else) — the ORIGINAL fast plain-text 200. Kept
    deliberately trivial and dependency-free, since this is what Render's
    port scan hits; it must never be slowed down or risk failing just
    because some other part of the app (e.g. a corrupt local state file)
    is having a bad moment.
  - `/status` — a JSON snapshot of actual liveness: when each of the
    three worker threads (producer/consumer/telegram-feedback) last
    reported in (see update_status(), called from main.py's loops), a
    derived `healthy` bool per thread based on how stale that heartbeat
    is, current queue depth, today's Gemini request count vs. the
    estimated quota, the current effective match threshold, and
    win/loss outcome counts. Meant for a human checking on the bot (or an
    external uptime monitor hitting this specific path) — NOT used by
    Render's own health check, which only needs the root path above.
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

logger = logging.getLogger("health_server")

_lock = threading.Lock()
_started_at = time.time()
# Populated by main.py via update_status() — keys are only ever heartbeat
# timestamps and simple current-state values pushed by the loops
# themselves (queue depth, etc.), NOT anything independently retrievable
# from another module (see get_status_snapshot for the latter).
_status = {
    "producer_last_heartbeat": None,
    "consumer_last_heartbeat": None,
    "feedback_last_heartbeat": None,
    "queue_size": None,
}

# How stale a thread's last heartbeat can be before /status reports it as
# unhealthy. Deliberately generous — each accounts for that loop's own
# worst-case legitimate blocking duration (e.g. the consumer can
# legitimately go quiet for up to CYCLE_TIMEOUT while a single batch is
# being evaluated) plus a buffer, so a slow-but-fine cycle never produces
# a false alarm.
_PRODUCER_STALE_AFTER = lambda: 2 * config.POLL_INTERVAL_MAX + 120
_CONSUMER_STALE_AFTER = lambda: config.CYCLE_TIMEOUT + 120
_FEEDBACK_STALE_AFTER = lambda: config.TELEGRAM_FEEDBACK_POLL_TIMEOUT + 60


def update_status(**kwargs) -> None:
    """
    Thread-safe update of the shared status dict — called from main.py's
    producer_loop/consumer_loop/telegram_feedback_loop to report a
    heartbeat (and, for the consumer, current queue depth) once per
    iteration. Never raises: a failure updating a status dashboard must
    never be allowed to crash an actual worker loop over something this
    low-stakes.
    """
    try:
        with _lock:
            _status.update(kwargs)
    except Exception:
        logger.error("Failed to update health status", exc_info=True)


def _thread_health(last_heartbeat, stale_after_seconds):
    if last_heartbeat is None:
        # Never reported in yet — could be genuine startup (a few seconds
        # old process) or a thread that died before its first iteration.
        # Left for the caller to interpret via "seconds_since_start"
        # rather than guessing here.
        return None, None
    age = round(time.time() - last_heartbeat, 1)
    return age, age <= stale_after_seconds


def get_status_snapshot() -> dict:
    """
    Assembles the full /status payload: main.py-pushed heartbeats/queue
    depth (see update_status), PLUS live data pulled directly from
    ai_agent and outcome_tracker — those two are read fresh on every
    request rather than pushed, since they're already retrievable from
    their own module state and there's no reason to duplicate/relay them
    through main.py. Never raises: any single piece failing (e.g. a
    corrupt outcomes.json) is reported as an "error" string for just that
    section rather than taking down the whole snapshot.
    """
    with _lock:
        snapshot = dict(_status)

    snapshot["uptime_seconds"] = round(time.time() - _started_at, 1)

    producer_age, producer_healthy = _thread_health(
        snapshot.get("producer_last_heartbeat"), _PRODUCER_STALE_AFTER(),
    )
    consumer_age, consumer_healthy = _thread_health(
        snapshot.get("consumer_last_heartbeat"), _CONSUMER_STALE_AFTER(),
    )
    feedback_age, feedback_healthy = _thread_health(
        snapshot.get("feedback_last_heartbeat"), _FEEDBACK_STALE_AFTER(),
    )
    snapshot["producer"] = {"seconds_since_heartbeat": producer_age, "healthy": producer_healthy}
    snapshot["consumer"] = {"seconds_since_heartbeat": consumer_age, "healthy": consumer_healthy}
    snapshot["telegram_feedback"] = {"seconds_since_heartbeat": feedback_age, "healthy": feedback_healthy}
    for key in ("producer_last_heartbeat", "consumer_last_heartbeat", "feedback_last_heartbeat"):
        snapshot.pop(key, None)  # superseded by the nested per-thread dicts above

    try:
        import ai_agent
        snapshot["gemini"] = {
            "requests_today": ai_agent._daily_request_tracker.get_today_count(),
            "estimated_daily_quota": config.GEMINI_ESTIMATED_DAILY_QUOTA,
            "effective_match_threshold": ai_agent.get_effective_match_threshold(),
        }
    except Exception:
        snapshot["gemini"] = {"error": "unavailable"}

    try:
        import outcome_tracker
        snapshot["outcomes"] = outcome_tracker.get_stats()
    except Exception:
        snapshot["outcomes"] = {"error": "unavailable"}

    return snapshot


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            self._handle_status()
        else:
            self._handle_root()

    def _handle_root(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Mostaql AI Freelance Assistant worker is running.\n")

    def _handle_status(self):
        try:
            body = json.dumps(get_status_snapshot(), indent=2, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            logger.error("Failed to serve /status", exc_info=True)
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default per-request access logs — they'd otherwise flood
        # the Render log dashboard on every health-check ping.
        pass


def _serve():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", config.PORT), HealthCheckHandler)
        logger.info("Health-check server listening on 0.0.0.0:%s (see /status for details)", config.PORT)
        server.serve_forever()
    except Exception:
        logger.exception("Health-check server crashed")


def start_health_server_in_background():
    """Starts the dummy server on a daemon thread and returns immediately."""
    thread = threading.Thread(target=_serve, name="health-server", daemon=True)
    thread.start()
    return thread
