"""
LLM-as-a-Judge tests for the Business Analyst agent.

Criteria verified:
- Spec completeness: requirements testable, acceptance_criteria are binary pass/fail
- Spec contains edge cases and error handling
- BA uses search tools before producing spec
- Structured output fields are valid
"""

import json
import sys
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_dev import extract_output, extract_tool_names, load_golden, skip_no_key

spec_completeness = GEval(
    name="Spec Completeness",
    evaluation_steps=[
        "Check that 'actual output' contains at least 3 functional requirements.",
        "Check that acceptance criteria are specific and testable (binary pass/fail, not vague).",
        "Check that requirements mention error handling or edge cases (e.g. invalid input, empty input).",
        "Check that tech_stack is populated with concrete library or stdlib names.",
        "Check that estimated_complexity is one of: simple, medium, complex.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

spec_accuracy = GEval(
    name="Spec Accuracy",
    evaluation_steps=[
        "Check that the specification in 'actual output' accurately addresses the user story in 'input'.",
        "Check that no requirements were invented that contradict the user story.",
        "Check that the title matches the user story intent.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


@pytest.fixture(scope="module")
def ba():
    from agents.ba import ba_agent
    return ba_agent


HAPPY_PATH = load_golden("happy_path")


@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH[:3], ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:3]])
def test_ba_spec_completeness(ba, example):
    story = example["user_story"]
    result = ba.invoke({"messages": [("user", story)]})

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = extract_output(result)

    test_case = LLMTestCase(input=story, actual_output=output)
    assert_test(test_case, [spec_completeness])


@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH[:3], ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:3]])
def test_ba_spec_accuracy(ba, example):
    story = example["user_story"]
    result = ba.invoke({"messages": [("user", story)]})

    structured = result.get("structured_response")
    output = structured.model_dump_json(indent=2) if structured else extract_output(result)

    test_case = LLMTestCase(input=story, actual_output=output)
    assert_test(test_case, [spec_accuracy])


@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH[:2], ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:2]])
def test_ba_calls_search_tools(ba, example):
    result = ba.invoke({"messages": [("user", example["user_story"])]})
    tools = extract_tool_names(result)
    assert tools & {"web_search", "knowledge_search"}, (
        f"BA must call at least one search tool. Called: {tools}"
    )


@skip_no_key
def test_ba_structured_output_fields(ba):
    story = "As a user, I want a function that checks if a string is a palindrome."
    result = ba.invoke({"messages": [("user", story)]})
    spec = result.get("structured_response")

    assert spec is not None, "BA did not return a structured SpecOutput"
    assert spec.title, "SpecOutput.title must not be empty"
    assert len(spec.requirements) >= 2, f"Expected ≥2 requirements, got {len(spec.requirements)}"
    assert len(spec.acceptance_criteria) >= 2, f"Expected ≥2 criteria, got {len(spec.acceptance_criteria)}"
    assert spec.estimated_complexity in ("simple", "medium", "complex")
    assert spec.tech_stack, "SpecOutput.tech_stack must not be empty"


@skip_no_key
def test_ba_edge_case_ambiguous_story(ba):
    """BA should still return a structured spec (possibly with assumptions) for vague input."""
    result = ba.invoke({"messages": [("user", "Build me a thing that works with data")]})
    spec = result.get("structured_response")
    # We accept either a spec with explicit assumptions or a request for clarification
    output = spec.model_dump_json(indent=2) if spec else extract_output(result)
    assert output, "BA returned empty output for ambiguous user story"
