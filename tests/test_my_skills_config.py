"""
Tests for config.py's MY_SKILLS env var override — requires reloading the
config module, since MY_SKILLS is computed once at import time from
os.environ. Each test restores config to its original state afterward so
later tests aren't affected.
"""

import importlib

import config


def test_default_used_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("MY_SKILLS", raising=False)
    importlib.reload(config)
    try:
        assert config.MY_SKILLS == config._DEFAULT_MY_SKILLS
        assert "Python" in config.MY_SKILLS
    finally:
        importlib.reload(config)  # restore original module state for other tests


def test_env_var_overrides_default_entirely(monkeypatch):
    monkeypatch.setenv("MY_SKILLS", "React,ريأكت,WordPress,ووردبريس")
    importlib.reload(config)
    try:
        assert config.MY_SKILLS == ["React", "ريأكت", "WordPress", "ووردبريس"]
        assert "Python" not in config.MY_SKILLS  # confirms full override, not additive
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_env_var_handles_messy_whitespace_and_empty_entries(monkeypatch):
    monkeypatch.setenv("MY_SKILLS", "  React ,  ريأكت  ,, WordPress ")
    importlib.reload(config)
    try:
        assert config.MY_SKILLS == ["React", "ريأكت", "WordPress"]
    finally:
        monkeypatch.undo()
        importlib.reload(config)
