"""
Shared fixtures and helpers for the AI Dev Team test suite.
Import via: from conftest_dev import ...
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

GOLDEN_PATH = Path(__file__).parent / "golden_dataset_dev.json"

_NO_KEY = not os.environ.get("OPENAI_API_KEY") and not os.environ.get("API_KEY")
skip_no_key = pytest.mark.skipif(_NO_KEY, reason="No API key configured")


def load_golden(category: str | None = None) -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    if category:
        return [ex for ex in data if ex["category"] == category]
    return data


def extract_output(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if content and not getattr(msg, "tool_calls", None):
            return content if isinstance(content, str) else str(content)
    return ""


def extract_tool_names(result: dict) -> set[str]:
    names = set()
    for msg in result.get("messages", []):
        for tc in getattr(msg, "tool_calls", None) or []:
            names.add(tc["name"])
    return names
