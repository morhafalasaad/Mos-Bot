"""
Tests for the seen_projects persistence path:

* scraper.get_new_projects() must hand ONLY this cycle's newly discovered
  project ids to _save_seen / db.mark_seen — never the whole historical set
  (which used to rewrite every old document and reset its TTL clock).
* db.mark_seen() must upsert cleanly, leave unrelated documents (and the
  timestamps of already-present ids) untouched, and never raise.
"""

from datetime import datetime, timedelta, timezone

import config
import db
import scraper
from scraper import Project


def _project(pid: str) -> Project:
    return Project(
        id=pid,
        title=f"Project {pid}",
        description="desc",
        url=f"https://mostaql.com/project/{pid}-x",
    )


def _stub_scraper_io(monkeypatch, listing):
    """Replaces every network-touching piece of get_new_projects()."""
    monkeypatch.setattr(scraper, "_build_session", lambda: _FakeSession())
    monkeypatch.setattr(scraper, "_category_urls", lambda: ["https://mostaql.com/projects"])
    monkeypatch.setattr(scraper, "fetch_page", lambda session, url: "<html></html>")
    monkeypatch.setattr(scraper, "parse_projects", lambda html: list(listing))
    monkeypatch.setattr(config, "FETCH_PROJECT_TAGS", False)


class _FakeSession:
    def close(self):
        pass


# ---------------------------------------------------------------------------
# scraper.get_new_projects
# ---------------------------------------------------------------------------

def test_get_new_projects_saves_only_newly_discovered_ids(monkeypatch):
    # 2000 historical ids already in Mongo; the listing has 2 old + 2 new.
    old_ids = [str(i) for i in range(1, 2001)]
    db.mark_seen(old_ids)

    listing = [_project("1"), _project("2"), _project("9001"), _project("9002")]
    _stub_scraper_io(monkeypatch, listing)

    saved_calls = []
    monkeypatch.setattr(scraper, "_save_seen", lambda ids: saved_calls.append(list(ids)))

    new_projects = scraper.get_new_projects()

    assert [p.id for p in new_projects] == ["9001", "9002"]
    assert len(saved_calls) == 1
    assert sorted(saved_calls[0]) == ["9001", "9002"]  # NOT the 2000 historical ids


def test_get_new_projects_passes_only_new_ids_to_db_mark_seen(monkeypatch):
    db.mark_seen(["1", "2", "3"])

    listing = [_project("2"), _project("3"), _project("4")]
    _stub_scraper_io(monkeypatch, listing)

    received = []
    monkeypatch.setattr(db, "mark_seen", lambda ids: received.append(list(ids)))

    new_projects = scraper.get_new_projects()

    assert [p.id for p in new_projects] == ["4"]
    assert received == [["4"]]


def test_get_new_projects_does_not_save_when_nothing_is_new(monkeypatch):
    db.mark_seen(["1", "2"])
    _stub_scraper_io(monkeypatch, [_project("1"), _project("2")])

    received = []
    monkeypatch.setattr(db, "mark_seen", lambda ids: received.append(list(ids)))

    assert scraper.get_new_projects() == []
    assert received == []


# ---------------------------------------------------------------------------
# db.mark_seen
# ---------------------------------------------------------------------------

def test_mark_seen_upserts_new_ids():
    db.mark_seen(["a", "b"])

    assert db.get_seen_ids() == {"a", "b"}


def test_mark_seen_does_not_reset_timestamps_of_existing_documents():
    coll = db.get_collection("seen_projects")
    # BSON datetimes have millisecond precision, so use a whole-second value.
    long_ago = (datetime.now(timezone.utc) - timedelta(days=50)).replace(microsecond=0)
    coll.insert_one({"_id": "old", "seen_at": long_ago})
    coll.insert_one({"_id": "unrelated", "seen_at": long_ago})

    # Re-marking "old" alongside a brand-new id must not touch either
    # existing document's seen_at.
    db.mark_seen(["old", "fresh"])

    old_doc = coll.find_one({"_id": "old"})
    unrelated_doc = coll.find_one({"_id": "unrelated"})
    fresh_doc = coll.find_one({"_id": "fresh"})
    assert old_doc["seen_at"].replace(tzinfo=timezone.utc) == long_ago
    assert unrelated_doc["seen_at"].replace(tzinfo=timezone.utc) == long_ago
    assert fresh_doc["seen_at"].replace(tzinfo=timezone.utc) > long_ago


def test_mark_seen_accepts_sets_generators_and_duplicates():
    db.mark_seen({"s1", "s2"})
    db.mark_seen(x for x in ["g1", "g1", "g2"])

    assert db.get_seen_ids() == {"s1", "s2", "g1", "g2"}


def test_mark_seen_with_empty_input_is_a_noop():
    db.mark_seen([])
    db.mark_seen(None)
    db.mark_seen(set())

    assert db.get_seen_ids() == set()


def test_mark_seen_falls_back_to_per_id_upserts_when_bulk_insert_fails(monkeypatch):
    coll = db.get_collection("seen_projects")

    def boom(*args, **kwargs):
        raise RuntimeError("bulk path unavailable")

    monkeypatch.setattr(type(coll), "insert_many", boom)

    db.mark_seen(["x", "y"])

    assert db.get_seen_ids() == {"x", "y"}


def test_mark_seen_never_raises_when_database_is_unavailable(monkeypatch):
    def broken(name):
        raise RuntimeError("mongo down")

    monkeypatch.setattr(db, "get_collection", broken)

    db.mark_seen(["a", "b"])  # must not raise
