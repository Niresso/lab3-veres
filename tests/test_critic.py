"""
Component tests for the Critic Agent.

Verifies that:
- critique identifies specific, actionable issues (not vague complaints)
- verdict is consistent with populated fields (REVISE → revision_requests exist)
- critique quality meets the GEval threshold
"""

import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import (
    extract_agent_output,
    skip_no_key,
)

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints like 'needs improvement'.",
        "Check that revision_requests are actionable — the researcher can act on each one.",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items.",
        "If verdict is REVISE, there must be at least one revision_request.",
        "Check that strengths and gaps are substantive (more than one word each).",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

SAMPLE_FINDINGS_GOOD = """
# Hybrid Search in RAG Systems

## Overview
Hybrid search combines BM25 sparse retrieval with dense vector similarity search.
BM25 excels at exact keyword matching while dense retrieval captures semantic similarity.

## How it Works
1. Both retrievers return candidate documents independently
2. Scores are fused using Reciprocal Rank Fusion (RRF) or weighted combination
3. A cross-encoder reranker (e.g., BAAI/bge-reranker-base) produces final ordering

## Benchmark Results
According to the BEIR benchmark (Thakur et al., 2021), hybrid search outperforms
pure BM25 by ~4% and pure dense retrieval by ~2% on average across datasets.

## Sources
- Thakur et al. (2021) BEIR: https://arxiv.org/abs/2104.08663
- Pinecone blog on hybrid search: https://www.pinecone.io/learn/hybrid-search/
"""

SAMPLE_FINDINGS_POOR = """
# RAG

RAG is good. It retrieves stuff. You should use it.
Sources: internet.
"""

SAMPLE_FINDINGS_INCOMPLETE = """
# Naive RAG vs Sentence-Window Retrieval

Naive RAG is simple. Sentence-window is better but more complex.
There are also other approaches. The topic is broad.
"""


@pytest.fixture(scope="module")
def critic():
    from agents.critic import critic_agent
    return critic_agent


@skip_no_key
def test_critique_quality_on_good_findings(critic):
    result = critic.invoke({"messages": [("user", SAMPLE_FINDINGS_GOOD)]})

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = extract_agent_output(result)

    test_case = LLMTestCase(
        input=SAMPLE_FINDINGS_GOOD,
        actual_output=output,
    )
    assert_test(test_case, [critique_quality])


@skip_no_key
def test_critique_quality_on_poor_findings(critic):
    result = critic.invoke({"messages": [("user", SAMPLE_FINDINGS_POOR)]})

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = extract_agent_output(result)

    test_case = LLMTestCase(
        input=SAMPLE_FINDINGS_POOR,
        actual_output=output,
    )
    assert_test(test_case, [critique_quality])


@skip_no_key
def test_critic_structured_response_consistency(critic):
    """REVISE verdict must be accompanied by at least one revision_request."""
    result = critic.invoke({"messages": [("user", SAMPLE_FINDINGS_INCOMPLETE)]})

    structured = result.get("structured_response")
    assert structured is not None, "Critic did not return a structured CritiqueResult"

    assert structured.verdict in ("APPROVE", "REVISE"), (
        f"Verdict must be APPROVE or REVISE, got {structured.verdict!r}"
    )

    if structured.verdict == "REVISE":
        assert structured.revision_requests, (
            "Verdict is REVISE but revision_requests is empty — critic must list what to fix"
        )

    if structured.verdict == "APPROVE":
        assert not structured.revision_requests or all(
            r.strip() for r in structured.revision_requests
        ), "APPROVE revision_requests must be non-empty strings if present"


@skip_no_key
def test_critic_returns_all_fields(critic):
    result = critic.invoke({"messages": [("user", SAMPLE_FINDINGS_GOOD)]})
    structured = result.get("structured_response")
    assert structured is not None

    assert hasattr(structured, "verdict")
    assert hasattr(structured, "is_fresh")
    assert hasattr(structured, "is_complete")
    assert hasattr(structured, "is_well_structured")
    assert isinstance(structured.strengths, list)
    assert isinstance(structured.gaps, list)
    assert isinstance(structured.revision_requests, list)
