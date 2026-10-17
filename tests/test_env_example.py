"""
Guards the repository's .env.example template (referenced by README.md's
setup steps): it must exist, list every required variable, and only
document variable names that config.py really reads — so a typo or a
renamed setting can't silently leave users configuring a dead variable.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"

REQUIRED = ["GEMINI_API_KEYS", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "MONGODB_URI"]
OPTIONAL_TUNABLES = [
    "MY_SKILLS", "MATCH_THRESHOLD", "MOSTAQL_CATEGORIES", "GITHUB_TOKEN",
    "GITHUB_REPO", "GEMINI_PROXY_URL", "REPLY_ASSISTANT_ENABLED",
]

# KEY= at the start of a line, optionally commented out ("# KEY=" / "#   KEY=").
_VAR_LINE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _documented_names() -> set:
    return set(_VAR_LINE.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


def test_env_example_exists():
    assert ENV_EXAMPLE.is_file(), ".env.example is referenced by README.md but missing"


@pytest.mark.parametrize("name", REQUIRED)
def test_required_variables_are_set_as_active_entries(name):
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    # Required variables must be ACTIVE (uncommented) lines the user fills in.
    assert re.search(rf"^{name}=", text, re.MULTILINE)


@pytest.mark.parametrize("name", OPTIONAL_TUNABLES)
def test_key_optional_tunables_are_documented(name):
    assert name in _documented_names()


def test_active_entries_parse_with_python_dotenv():
    dotenv = pytest.importorskip("dotenv")
    values = dotenv.dotenv_values(ENV_EXAMPLE)
    for name in REQUIRED:
        assert name in values


def test_every_documented_variable_is_actually_read_by_config():
    config_source = (ROOT / "config.py").read_text(encoding="utf-8")
    unknown = sorted(n for n in _documented_names() if f'"{n}"' not in config_source)
    assert unknown == [], f".env.example documents variables config.py never reads: {unknown}"
