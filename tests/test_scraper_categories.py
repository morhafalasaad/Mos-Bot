"""
Tests for scraper._category_urls — turns MOSTAQL_CATEGORIES into actual
request URLs.
"""

import config
import scraper


def test_no_categories_configured_returns_default_url(monkeypatch):
    monkeypatch.setattr(config, "MOSTAQL_CATEGORIES", "")
    monkeypatch.setattr(config, "MOSTAQL_PROJECTS_URL", "https://mostaql.com/projects")

    urls = scraper._category_urls()

    assert urls == ["https://mostaql.com/projects"]


def test_multiple_categories_build_one_url_each(monkeypatch):
    monkeypatch.setattr(config, "MOSTAQL_CATEGORIES", "python,data-science,machine-learning")
    monkeypatch.setattr(config, "MOSTAQL_CATEGORY_URL_TEMPLATE", "https://mostaql.com/projects?category={category}")

    urls = scraper._category_urls()

    assert urls == [
        "https://mostaql.com/projects?category=python",
        "https://mostaql.com/projects?category=data-science",
        "https://mostaql.com/projects?category=machine-learning",
    ]


def test_whitespace_and_empty_entries_are_cleaned(monkeypatch):
    monkeypatch.setattr(config, "MOSTAQL_CATEGORIES", "  python , , data-science  ")
    monkeypatch.setattr(config, "MOSTAQL_CATEGORY_URL_TEMPLATE", "https://mostaql.com/projects?category={category}")

    urls = scraper._category_urls()

    assert urls == [
        "https://mostaql.com/projects?category=python",
        "https://mostaql.com/projects?category=data-science",
    ]
