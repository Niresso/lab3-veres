"""
Component tests for the Planner Agent.

Verifies that:
- the plan contains specific, actionable search queries
- sources_to_check references valid source types
- output_format is described clearly
- the agent calls at least one search tool before producing the plan
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import (
    extract_agent_output,
    extract_tool_calls_from_result,
    skip_no_key,
)

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague generic phrases like 'search for X').",
        "Check that sources_to_check includes at least one of 'knowledge_base' or 'web'.",
        "Check that the output_format field clearly describes what the final report should look like.",
        "Check that goal accurately reflects what the user asked for.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


@pytest.fixture(scope="module")
def planner():
    from agents.planner import planner_agent
    return planner_agent


PLANNER_QUERIES = [
    "Compare naive RAG vs sentence-window retrieval",
    "What are best practices for chunking documents in RAG systems?",
    "How does hybrid search (BM25 + vector) improve retrieval quality?",
]


@skip_no_key
@pytest.mark.parametrize("query", PLANNER_QUERIES)
def test_plan_quality(planner, query):
    result = planner.invoke({"messages": [("user", query)]})

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = extract_agent_output(result)

    test_case = LLMTestCase(input=query, actual_output=output)
    assert_test(test_case, [plan_quality])


@skip_no_key
@pytest.mark.parametrize("query", PLANNER_QUERIES[:2])
def test_planner_calls_search_tools(planner, query):
    result = planner.invoke({"messages": [("user", query)]})
    tool_calls = extract_tool_calls_from_result(result)
    tool_names = {tc.name for tc in tool_calls}

    search_tools = {"web_search", "knowledge_search"}
    assert search_tools & tool_names, (
        f"Planner must call at least one search tool before producing a plan. "
        f"Called: {tool_names}"
    )


@skip_no_key
def test_planner_structured_response_fields(planner):
    """Planner must return a valid ResearchPlan with all required fields."""
    query = "Explain the role of cross-encoder reranking in RAG pipelines"
    result = planner.invoke({"messages": [("user", query)]})

    structured = result.get("structured_response")
    assert structured is not None, "Planner did not return a structured ResearchPlan"

    assert structured.goal, "ResearchPlan.goal must not be empty"
    assert len(structured.search_queries) >= 2, (
        f"Expected at least 2 search queries, got {len(structured.search_queries)}"
    )
    assert structured.sources_to_check, "ResearchPlan.sources_to_check must not be empty"
    valid_sources = {"knowledge_base", "web"}
    for src in structured.sources_to_check:
        assert src in valid_sources, f"Unknown source: {src!r}"
    assert structured.output_format, "ResearchPlan.output_format must not be empty"
