"""
Component tests for the Research Agent.

Verifies that:
- research findings are grounded in retrieved context (no hallucination)
- the output is a non-trivial markdown document
- the agent uses the tools listed in sources_to_check
"""

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import (
    extract_agent_output,
    extract_tool_calls_from_result,
    extract_retrieval_context,
    skip_no_key,
)

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from the 'actual output'.",
        "For each claim, check if it can be directly supported by the 'retrieval context'.",
        "Claims not present in retrieval context count as ungrounded, even if true in general.",
        "Score = number of grounded claims / total claims. Higher is better.",
        "If retrieval context is empty, score 0 because no grounding is possible.",
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
    model="gpt-4o-mini",
    threshold=0.5,
)

completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Read the research plan in 'input' and identify the stated goal.",
        "Check whether 'actual output' addresses all search queries listed in the plan.",
        "Penalise if major subtopics from the plan are absent from the output.",
        "Check that the output contains citations or source references.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.6,
)


@pytest.fixture(scope="module")
def planner():
    from agents.planner import planner_agent
    return planner_agent


@pytest.fixture(scope="module")
def researcher():
    from agents.research import research_agent
    return research_agent


RESEARCH_QUERIES = [
    "Compare naive RAG vs sentence-window retrieval",
    "How does hybrid search (BM25 + vector) improve retrieval quality?",
]


def _build_research_request(planner, query: str) -> tuple[str, str]:
    """Run planner and return (plan_json, combined_request_for_researcher)."""
    plan_result = planner.invoke({"messages": [("user", query)]})
    structured = plan_result.get("structured_response")
    if structured is not None:
        plan_json = structured.model_dump_json(indent=2)
    else:
        plan_json = extract_agent_output(plan_result)
    combined = f"Original question: {query}\n\nResearch plan:\n{plan_json}"
    return plan_json, combined


@skip_no_key
@pytest.mark.parametrize("query", RESEARCH_QUERIES)
def test_research_groundedness(planner, researcher, query):
    plan_json, combined_request = _build_research_request(planner, query)

    result = researcher.invoke({"messages": [("user", combined_request)]})
    output = extract_agent_output(result)
    context = extract_retrieval_context(result)

    assert output, "Research agent returned empty output"

    test_case = LLMTestCase(
        input=combined_request,
        actual_output=output,
        retrieval_context=context if context else ["No retrieval context captured."],
    )
    assert_test(test_case, [groundedness])


@skip_no_key
@pytest.mark.parametrize("query", RESEARCH_QUERIES)
def test_research_completeness(planner, researcher, query):
    plan_json, combined_request = _build_research_request(planner, query)

    result = researcher.invoke({"messages": [("user", combined_request)]})
    output = extract_agent_output(result)

    test_case = LLMTestCase(input=combined_request, actual_output=output)
    assert_test(test_case, [completeness])


@skip_no_key
def test_researcher_uses_search_tools(planner, researcher):
    query = "What embedding models work best for scientific document retrieval?"
    _, combined_request = _build_research_request(planner, query)

    result = researcher.invoke({"messages": [("user", combined_request)]})
    tool_calls = extract_tool_calls_from_result(result)
    tool_names = {tc.name for tc in tool_calls}

    assert tool_names & {"web_search", "knowledge_search", "read_url"}, (
        f"Researcher must call at least one search or read tool. Called: {tool_names}"
    )


@skip_no_key
def test_research_output_is_markdown(planner, researcher):
    query = "Explain the role of cross-encoder reranking in RAG pipelines"
    _, combined_request = _build_research_request(planner, query)

    result = researcher.invoke({"messages": [("user", combined_request)]})
    output = extract_agent_output(result)

    assert len(output) > 200, "Research output is too short to be a meaningful report"
    has_markdown = any(marker in output for marker in ("#", "##", "**", "-", "*"))
    assert has_markdown, "Research output should contain markdown formatting"
