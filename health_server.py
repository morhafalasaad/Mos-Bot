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
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

logger = logging.getLogger("health_server")


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Mostaql AI Freelance Assistant worker is running.\n")

    def log_message(self, format, *args):
        # Silence default per-request access logs — they'd otherwise flood
        # the Render log dashboard on every health-check ping.
        pass


def _serve():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", config.PORT), HealthCheckHandler)
        logger.info("Health-check server listening on 0.0.0.0:%s", config.PORT)
        server.serve_forever()
    except Exception:
        logger.exception("Health-check server crashed")


def start_health_server_in_background():
    """Starts the dummy server on a daemon thread and returns immediately."""
    thread = threading.Thread(target=_serve, name="health-server", daemon=True)
    thread.start()
    return thread
