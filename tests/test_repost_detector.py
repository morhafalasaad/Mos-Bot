"""
Tests for repost_detector.py — advisory-only detection of likely
reposted/duplicate projects via stdlib text similarity. Storage is
MongoDB Atlas's `repost_history` collection (see db.py) as of the Sept
2026 migration off a local JSON file — see tests/conftest.py's
`isolated_state` fixture for the fresh in-memory (`mongomock`) database
each test gets.
"""

from datetime import datetime, timedelta, timezone

import config
import db
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

    db.get_collection("repost_history").insert_one({
        "project_id": "old_1",
        "title": TITLE,
        "normalized_text": repost_detector._normalize(f"{TITLE} {DESC}"),
        "recorded_at": datetime.now(timezone.utc) - timedelta(days=40),
    })

    warning = repost_detector.check_and_record("new_1", TITLE, DESC)
    assert warning is None


def test_history_size_is_capped(monkeypatch):
    monkeypatch.setattr(config, "REPOST_HISTORY_MAX_ENTRIES", 3)

    for i in range(5):
        repost_detector.check_and_record(f"p{i}", f"Unique title {i} xyz", f"Unique description {i} abc")

    count = db.get_collection("repost_history").count_documents({})
    assert count <= 3


def test_disabled_feature_always_returns_none(monkeypatch):
    monkeypatch.setattr(config, "REPOST_DETECTION_ENABLED", False)
    repost_detector.check_and_record("proj_001", TITLE, DESC)
    warning = repost_detector.check_and_record("proj_002", TITLE, DESC)
    assert warning is None


def test_backend_error_degrades_to_no_warning_and_never_raises(monkeypatch):
    def _broken_get_repost_history(*a, **kw):
        raise RuntimeError("Atlas unreachable")

    monkeypatch.setattr(db, "get_repost_history", _broken_get_repost_history)

    warning = repost_detector.check_and_record("proj_001", TITLE, DESC)
    assert warning is None  # must not raise


def test_empty_title_and_description_returns_none():
    assert repost_detector.check_and_record("proj_x", "", "") is None


# ---------------------------------------------------------------------------
# Telegram legacy-Markdown safety of the warning string. An unescaped `_`,
# `*`, `` ` `` or `[` inside the interpolated title makes Telegram reject
# the whole message (HTTP 400), forcing the plain-text fallback.
# ---------------------------------------------------------------------------

def _unescaped_specials(text: str) -> dict:
    """Counts each legacy-Markdown special character that is NOT preceded by
    a backslash escape."""
    import re
    without_escaped = re.sub(r"\\.", "", text)
    return {ch: without_escaped.count(ch) for ch in ("_", "*", "`", "[")}


def test_warning_escapes_underscores_and_asterisks_in_title():
    special_title = "Need scraper_v2 built *urgently* for my_shop"
    repost_detector.check_and_record("proj_001", special_title, DESC)

    warning = repost_detector.check_and_record("proj_002", special_title, DESC)

    assert warning is not None
    assert "scraper\\_v2" in warning
    assert "\\*urgently\\*" in warning
    assert "my\\_shop" in warning
    # The only unescaped specials left are the two `*` of the bold header.
    assert _unescaped_specials(warning) == {"_": 0, "*": 2, "`": 0, "[": 0}


def test_warning_escapes_backticks_and_brackets_in_title():
    special_title = "Fix `main.py` crash [phase 1] in python scraper"
    repost_detector.check_and_record("proj_001", special_title, DESC)

    warning = repost_detector.check_and_record("proj_002", special_title, DESC)

    assert warning is not None
    assert "\\`main.py\\`" in warning
    assert "\\[phase 1]" in warning
    assert _unescaped_specials(warning) == {"_": 0, "*": 2, "`": 0, "[": 0}


def test_warning_with_plain_title_is_unchanged_by_escaping():
    repost_detector.check_and_record("proj_001", TITLE, DESC)

    warning = repost_detector.check_and_record("proj_002", TITLE, DESC)

    assert warning is not None
    assert f"مع: {TITLE})" in warning  # nothing to escape, so the title appears verbatim
    assert "\\" not in warning


def test_repost_warning_with_missing_title_does_not_raise():
    """A history entry lacking a title (title=None) must still yield a
    well-formed warning rather than raising or printing 'None'."""
    db.get_collection("repost_history").insert_one({
        "project_id": "old_1",
        "title": None,
        "normalized_text": repost_detector._normalize(f"{TITLE} {DESC}"),
        "recorded_at": datetime.now(timezone.utc),
    })

    warning = repost_detector.check_and_record("new_1", TITLE, DESC)

    assert warning is not None
    assert "None" not in warning
