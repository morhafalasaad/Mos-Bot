"""
Tests for health_server's /health route: an ACTIVE MongoDB ping that
returns 200 when reachable and 503 when the database is down, while the
root path stays the instant dependency-free 200 Render's port scan needs.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import db
import health_server


def test_check_database_connected():
    status, body = health_server.check_database()
    assert status == 200
    assert body == {"status": "ok", "db": "connected"}


def test_check_database_unreachable(monkeypatch):
    def boom():
        raise ConnectionError("mongodb://user:secret@host unreachable")

    monkeypatch.setattr(db, "get_db", boom)
    status, body = health_server.check_database()
    assert status == 503
    assert body["status"] == "degraded"
    assert body["db"] == "disconnected"
    assert body["error"] == "ConnectionError"
    assert "secret" not in json.dumps(body)  # only the class name is exposed


def test_check_database_ping_timeout(monkeypatch):
    class _Admin:
        def command(self, name):
            raise TimeoutError("server selection timed out")

    class _Client:
        admin = _Admin()

    class _Db:
        client = _Client()

    monkeypatch.setattr(db, "get_db", lambda: _Db())
    status, body = health_server.check_database()
    assert status == 503
    assert body["error"] == "TimeoutError"


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), health_server.HealthCheckHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass any sandbox proxy
    try:
        with opener.open(url, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def test_health_route_200_when_db_healthy(server):
    status, body = _get(server + "/health")
    assert status == 200
    assert json.loads(body) == {"status": "ok", "db": "connected"}


def test_health_route_503_when_db_down(server, monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_db", boom)
    status, body = _get(server + "/health")
    assert status == 503
    payload = json.loads(body)
    assert payload["status"] == "degraded"
    assert payload["db"] == "disconnected"
    assert payload["error"] == "RuntimeError"


def test_health_route_ignores_query_string_and_trailing_slash(server):
    assert _get(server + "/health?probe=1")[0] == 200
    assert _get(server + "/health/")[0] == 200


def test_root_stays_fast_and_independent_of_db(server, monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_db", boom)
    status, body = _get(server + "/")
    assert status == 200
    assert "running" in body
