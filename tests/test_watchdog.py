"""
Tests for main.process_batch_with_watchdog — the hard per-batch timeout
around Gemini evaluation.

Regression guarded here: the watchdog used to run every batch through ONE
module-level single-worker ThreadPoolExecutor, so a batch that timed out
left its hung thread occupying the pool's only worker and every later batch
queued behind it (and timed out too). Each batch must now be independent.
"""

import logging
import threading
import time
from types import SimpleNamespace

import config
import main


def _project(pid="1", title="T"):
    return SimpleNamespace(id=pid, title=title)


def test_timed_out_batch_does_not_block_the_next_batch(monkeypatch):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 0.3)

    release_hung_batch = threading.Event()
    hung_started = threading.Event()
    processed = []
    unavailable = []

    def fake_process(projects):
        if projects[0].id == "hung":
            hung_started.set()
            release_hung_batch.wait(timeout=30)  # simulates a Gemini call that never returns
            return
        processed.append([p.id for p in projects])

    monkeypatch.setattr(main, "process_project_batch", fake_process)
    monkeypatch.setattr(main, "handle_ai_unavailable", lambda project, reason: unavailable.append(project.id))

    try:
        main.process_batch_with_watchdog([_project("hung")])
        assert hung_started.is_set()
        assert unavailable == ["hung"]  # timed-out batch routed to the fallback

        # The hung worker is STILL alive here. The very next batch must run
        # right away — not queue behind it until its own timeout fires.
        start = time.monotonic()
        main.process_batch_with_watchdog([_project("next-1"), _project("next-2")])
        elapsed = time.monotonic() - start

        assert processed == [["next-1", "next-2"]]
        assert elapsed < 0.25  # well under CYCLE_TIMEOUT: it did not wait on the hung thread
        assert unavailable == ["hung"]  # the healthy batch was NOT treated as unavailable
    finally:
        release_hung_batch.set()


def test_several_consecutive_timeouts_each_fall_back_independently(monkeypatch):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 0.2)

    release = threading.Event()
    unavailable = []
    monkeypatch.setattr(main, "process_project_batch", lambda projects: release.wait(timeout=30))
    monkeypatch.setattr(main, "handle_ai_unavailable", lambda project, reason: unavailable.append(project.id))

    try:
        main.process_batch_with_watchdog([_project("a")])
        main.process_batch_with_watchdog([_project("b")])
        main.process_batch_with_watchdog([_project("c")])
    finally:
        release.set()

    assert unavailable == ["a", "b", "c"]


def test_timeout_calls_handle_ai_unavailable_for_every_project_in_the_batch(monkeypatch):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 0.2)

    release = threading.Event()
    calls = []
    monkeypatch.setattr(main, "process_project_batch", lambda projects: release.wait(timeout=30))
    monkeypatch.setattr(main, "handle_ai_unavailable", lambda project, reason: calls.append((project.id, reason)))

    try:
        main.process_batch_with_watchdog([_project("1"), _project("2"), _project("3")])
    finally:
        release.set()

    assert [pid for pid, _ in calls] == ["1", "2", "3"]
    assert all("exceeded" in reason and "timeout" in reason for _, reason in calls)


def test_exception_in_batch_is_caught_and_logged_without_fallback(monkeypatch, caplog):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 5)
    fallback = []

    def boom(projects):
        raise RuntimeError("kaboom inside the batch")

    monkeypatch.setattr(main, "process_project_batch", boom)
    monkeypatch.setattr(main, "handle_ai_unavailable", lambda project, reason: fallback.append(project.id))

    with caplog.at_level(logging.ERROR, logger="main"):
        main.process_batch_with_watchdog([_project("1"), _project("2")])  # must not raise

    assert fallback == []  # an ordinary exception is logged, not treated as a timeout
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Unexpected error while processing a batch of 2 project(s)" in m
        and "kaboom inside the batch" in m
        and "RuntimeError" in m
        for m in messages
    )


def test_fast_successful_batch_runs_and_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 5)
    processed = []
    fallback = []
    monkeypatch.setattr(main, "process_project_batch", lambda projects: processed.append(len(projects)))
    monkeypatch.setattr(main, "handle_ai_unavailable", lambda project, reason: fallback.append(project.id))

    main.process_batch_with_watchdog([_project("1"), _project("2")])

    assert processed == [2]
    assert fallback == []


def test_failure_inside_timeout_fallback_never_propagates(monkeypatch, caplog):
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 0.2)
    release = threading.Event()
    monkeypatch.setattr(main, "process_project_batch", lambda projects: release.wait(timeout=30))

    def broken_fallback(project, reason):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(main, "handle_ai_unavailable", broken_fallback)

    try:
        with caplog.at_level(logging.ERROR, logger="main"):
            main.process_batch_with_watchdog([_project("1")])  # must not raise
    finally:
        release.set()

    assert any("Fallback handling after batch timeout also failed" in r.getMessage() for r in caplog.records)


def test_watchdog_worker_is_a_daemon_thread(monkeypatch):
    """An abandoned (hung) worker must never keep the process alive."""
    monkeypatch.setattr(config, "CYCLE_TIMEOUT", 5)
    seen = {}

    def capture(projects):
        seen["daemon"] = threading.current_thread().daemon

    monkeypatch.setattr(main, "process_project_batch", capture)
    main.process_batch_with_watchdog([_project("1")])

    assert seen["daemon"] is True
