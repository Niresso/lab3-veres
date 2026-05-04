"""
Tool correctness tests.

Verifies that each agent calls the expected tools given a specific input,
using deepeval's ToolCorrectnessMetric.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from conftest import (
    extract_agent_output,
    extract_tool_calls_from_result,
    skip_no_key,
)

tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model="gpt-4o-mini",
)


@pytest.fixture(scope="module")
def planner():
    from agents.planner import planner_agent
    return planner_agent


@pytest.fixture(scope="module")
def researcher():
    from agents.research import research_agent
    return research_agent


@pytest.fixture(scope="module")
def critic():
    from agents.critic import critic_agent
    return critic_agent


# ── Planner tool correctness ──────────────────────────────────────────────────

@skip_no_key
def test_planner_calls_search_tools_metric(planner):
    """Planner must explore domain via web_search and/or knowledge_search before planning."""
    query = "Compare naive RAG vs sentence-window retrieval"
    result = planner.invoke({"messages": [("user", query)]})
    output = extract_agent_output(result)
    structured = result.get("structured_response")
    if structured:
        output = structured.model_dump_json(indent=2)

    tools_called = extract_tool_calls_from_result(result)

    expected_tools = [
        ToolCall(name="web_search"),
        ToolCall(name="knowledge_search"),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


@skip_no_key
def test_planner_uses_knowledge_search_for_local_topics(planner):
    """Planner should check the local knowledge base for topics that may be in ingested docs."""
    query = "What does our knowledge base say about RAG chunking strategies?"
    result = planner.invoke({"messages": [("user", query)]})
    output = extract_agent_output(result)
    structured = result.get("structured_response")
    if structured:
        output = structured.model_dump_json(indent=2)

    tools_called = extract_tool_calls_from_result(result)
    tool_names = {tc.name for tc in tools_called}

    assert "knowledge_search" in tool_names, (
        f"Planner should call knowledge_search for local-knowledge queries. Called: {tool_names}"
    )


# ── Researcher tool correctness ───────────────────────────────────────────────

@skip_no_key
def test_researcher_uses_web_search_when_plan_requires_it(researcher):
    """Researcher must call web_search when sources_to_check includes 'web'."""
    plan_request = """\
Original question: How does HNSW indexing improve ANN search speed?

Research plan:
{
  "goal": "Explain HNSW and its advantages for approximate nearest-neighbour search",
  "search_queries": ["HNSW algorithm explained", "HNSW vs flat index performance"],
  "sources_to_check": ["web"],
  "output_format": "markdown report with sections: Overview, Algorithm, Benchmarks, Conclusion"
}"""

    result = researcher.invoke({"messages": [("user", plan_request)]})
    output = extract_agent_output(result)
    tools_called = extract_tool_calls_from_result(result)

    expected_tools = [
        ToolCall(name="web_search"),
        ToolCall(name="read_url"),
    ]

    test_case = LLMTestCase(
        input=plan_request,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])


@skip_no_key
def test_researcher_uses_knowledge_search_when_plan_requires_it(researcher):
    """Researcher must call knowledge_search when sources_to_check includes 'knowledge_base'."""
    plan_request = """\
Original question: What do our ingested documents say about embedding models?

Research plan:
{
  "goal": "Summarise information about embedding models from the local knowledge base",
  "search_queries": ["embedding models comparison", "sentence transformers performance"],
  "sources_to_check": ["knowledge_base"],
  "output_format": "markdown summary with key findings and citations"
}"""

    result = researcher.invoke({"messages": [("user", plan_request)]})
    tools_called = extract_tool_calls_from_result(result)
    tool_names = {tc.name for tc in tools_called}

    assert "knowledge_search" in tool_names, (
        f"Researcher must use knowledge_search for knowledge_base sources. Called: {tool_names}"
    )


# ── Critic tool correctness ───────────────────────────────────────────────────

@skip_no_key
def test_critic_uses_verification_tools(critic):
    """Critic should verify facts by calling web_search or knowledge_search."""
    findings = """\
# Hybrid Search Performance

According to BEIR 2021 benchmarks, hybrid search outperforms pure dense retrieval
by approximately 4% on the MSMARCO dataset when using BM25 weights of 0.3 and
vector weights of 0.7, with BAAI/bge-large-en-v1.5 as the embedding model.

Sources: https://arxiv.org/abs/2104.08663
"""
    result = critic.invoke({"messages": [("user", findings)]})
    output = extract_agent_output(result)
    structured = result.get("structured_response")
    if structured:
        output = structured.model_dump_json(indent=2)

    tools_called = extract_tool_calls_from_result(result)

    expected_tools = [
        ToolCall(name="web_search"),
        ToolCall(name="knowledge_search"),
    ]

    test_case = LLMTestCase(
        input=findings,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    assert_test(test_case, [tool_metric])
