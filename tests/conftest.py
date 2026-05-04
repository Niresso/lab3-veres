import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# deepeval uses OPENAI_API_KEY; mirror it from the project's API_KEY if needed
if "OPENAI_API_KEY" not in os.environ and "API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

_NO_API_KEY = not os.environ.get("OPENAI_API_KEY") and not os.environ.get("API_KEY")
skip_no_key = pytest.mark.skipif(_NO_API_KEY, reason="No API key configured")


def load_golden_dataset(category: str | None = None) -> list[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    if category:
        return [ex for ex in data if ex["category"] == category]
    return data


def extract_agent_output(result: dict) -> str:
    """Return the last non-tool-call text content from an agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not content:
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return content if isinstance(content, str) else str(content)
    return ""


def extract_tool_calls_from_result(result: dict) -> list:
    """Return ToolCall objects for every tool invoked during agent execution."""
    from deepeval.test_case import ToolCall

    messages = result.get("messages", [])
    calls = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append(
                ToolCall(
                    name=tc["name"],
                    input_parameters=tc.get("args", {}),
                )
            )
    return calls


def extract_retrieval_context(result: dict) -> list[str]:
    """Collect text content returned by retrieval/search tool messages."""
    messages = result.get("messages", [])
    retrieval_tools = {"knowledge_search", "web_search", "read_url"}
    context = []
    for msg in messages:
        name = getattr(msg, "name", None)
        if name in retrieval_tools:
            content = str(getattr(msg, "content", ""))
            if content and content not in ("[]", ""):
                context.append(content[:2000])
    return context
