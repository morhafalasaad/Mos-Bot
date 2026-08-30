"""
Tests for repost_detector.py — advisory-only detection of likely
reposted/duplicate projects via stdlib text similarity.
"""

import json
from datetime import datetime, timedelta, timezone

import config
import repost_detector

TITLE = "Build a Python web scraper for e-commerce data"
DESC = (
    "I need an experienced developer to build a robust web scraper using "
    "Python that extracts product prices and details from several "
    "e-commerce websites, handling pagination and anti-bot measures."
)


def test_first_occurrence_has_no_warning():
    warning = repost_detector.check_and_record("proj_001", TITLE, DESC)
    assert warning is None


def test_exact_repost_with_new_id_is_flagged():
    repost_detector.check_and_record("proj_001", TITLE, DESC)
    warning = repost_detector.check_and_record("proj_002", TITLE, DESC)

    assert warning is not None
    assert "معاد نشره" in warning


def test_slightly_reworded_repost_is_still_caught():
    repost_detector.check_and_record("proj_001", TITLE, DESC)

    reworded_title = TITLE.replace("web scraper", "scraper")
    reworded_desc = DESC.replace("robust web scraper", "robust scraper")
    warning = repost_detector.check_and_record("proj_002", reworded_title, reworded_desc)

    assert warning is not None


def test_genuinely_different_project_is_not_flagged():
    repost_detector.check_and_record("proj_001", TITLE, DESC)

    warning = repost_detector.check_and_record(
        "proj_002",
        "Design a company logo and brand identity",
        "Looking for a creative designer to create a modern logo and full "
        "brand identity package for a new startup in the food industry.",
    )
    assert warning is None


def test_self_comparison_is_excluded():
    """Re-checking the exact same project_id against its own prior record
    must not match itself."""
    repost_detector.check_and_record("proj_A", TITLE, DESC)
    warning = repost_detector.check_and_record("proj_A", TITLE, DESC)
    assert warning is None


def test_expired_entries_are_pruned_before_comparison(monkeypatch):
    monkeypatch.setattr(config, "REPOST_HISTORY_MAX_AGE_DAYS", 30)

    old_entry = {
        "project_id": "old_1",
        "title": TITLE,
        "normalized_text": repost_detector._normalize(f"{TITLE} {DESC}"),
        "recorded_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(),
    }
    with open(config.REPOST_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([old_entry], f)

    warning = repost_detector.check_and_record("new_1", TITLE, DESC)
    assert warning is None


def test_history_size_is_capped(monkeypatch):
    monkeypatch.setattr(config, "REPOST_HISTORY_MAX_ENTRIES", 3)

    for i in range(5):
        repost_detector.check_and_record(f"p{i}", f"Unique title {i} xyz", f"Unique description {i} abc")

    with open(config.REPOST_HISTORY_FILE, encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) <= 3


def test_disabled_feature_always_returns_none(monkeypatch):
    monkeypatch.setattr(config, "REPOST_DETECTION_ENABLED", False)
    repost_detector.check_and_record("proj_001", TITLE, DESC)
    warning = repost_detector.check_and_record("proj_002", TITLE, DESC)
    assert warning is None


def test_corrupt_history_file_degrades_to_no_warning(monkeypatch):
    with open(config.REPOST_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write("not valid json {{{")

    warning = repost_detector.check_and_record("proj_001", TITLE, DESC)
    assert warning is None  # must not raise


def test_empty_title_and_description_returns_none():
    assert repost_detector.check_and_record("proj_x", "", "") is None
