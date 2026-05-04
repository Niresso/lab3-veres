"""
End-to-end pipeline tests evaluated against the golden dataset.

Runs the full Planner → Researcher → Critic pipeline for each happy_path
golden example and evaluates with AnswerRelevancyMetric + GEval Correctness.
Results are saved to tests/e2e_results.json after each run.
"""

import json
import time
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest import (
    extract_agent_output,
    load_golden_dataset,
    skip_no_key,
)

RESULTS_PATH = Path(__file__).parent / "e2e_results.json"

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'.",
        "Penalise omission of critical details that appear in 'expected output'.",
        "Different wording of the same concept is acceptable.",
        "Do not penalise extra correct information that is not in 'expected output'.",
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)


def run_pipeline(query: str) -> tuple[str, str]:
    """
    Run Planner → Researcher → Critic in sequence.
    Returns (research_output, critique_json).
    """
    from agents.planner import planner_agent
    from agents.research import research_agent
    from agents.critic import critic_agent

    # Step 1: Plan
    plan_result = planner_agent.invoke({"messages": [("user", query)]})
    structured_plan = plan_result.get("structured_response")
    if structured_plan is not None:
        plan_json = structured_plan.model_dump_json(indent=2)
    else:
        plan_json = extract_agent_output(plan_result)

    combined_request = f"Original question: {query}\n\nResearch plan:\n{plan_json}"

    # Step 2: Research
    research_result = research_agent.invoke({"messages": [("user", combined_request)]})
    research_output = extract_agent_output(research_result)

    # Step 3: Critique
    critique_result = critic_agent.invoke({"messages": [("user", research_output)]})
    structured_critique = critique_result.get("structured_response")
    if structured_critique is not None:
        critique_json = structured_critique.model_dump_json(indent=2)
    else:
        critique_json = extract_agent_output(critique_result)

    return research_output, critique_json


def _save_result(entry: dict) -> None:
    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Happy path E2E tests ──────────────────────────────────────────────────────

HAPPY_PATH = load_golden_dataset("happy_path")


@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH, ids=[ex["input"][:50] for ex in HAPPY_PATH])
def test_e2e_answer_relevancy(example):
    query = example["input"]
    research_output, _ = run_pipeline(query)

    assert research_output, f"Pipeline returned empty output for: {query!r}"

    test_case = LLMTestCase(
        input=query,
        actual_output=research_output,
    )

    start = time.time()
    assert_test(test_case, [answer_relevancy])
    elapsed = time.time() - start

    _save_result({
        "input": query,
        "category": example["category"],
        "actual_output": research_output[:500],
        "answer_relevancy_score": answer_relevancy.score,
        "answer_relevancy_passed": answer_relevancy.is_successful(),
        "elapsed_seconds": round(elapsed, 1),
    })


@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH, ids=[ex["input"][:50] for ex in HAPPY_PATH])
def test_e2e_correctness(example):
    query = example["input"]
    expected_output = example["expected_output"]

    research_output, _ = run_pipeline(query)

    assert research_output, f"Pipeline returned empty output for: {query!r}"

    test_case = LLMTestCase(
        input=query,
        actual_output=research_output,
        expected_output=expected_output,
    )
    assert_test(test_case, [correctness])


# ── Critique verdict consistency on E2E output ────────────────────────────────

@skip_no_key
@pytest.mark.parametrize("example", HAPPY_PATH[:2], ids=[ex["input"][:50] for ex in HAPPY_PATH[:2]])
def test_e2e_critique_verdict_is_approve(example):
    """For high-quality happy-path queries the critic should eventually APPROVE."""
    from agents.planner import planner_agent
    from agents.research import research_agent
    from agents.critic import critic_agent

    query = example["input"]

    plan_result = planner_agent.invoke({"messages": [("user", query)]})
    structured_plan = plan_result.get("structured_response")
    plan_json = structured_plan.model_dump_json(indent=2) if structured_plan else extract_agent_output(plan_result)

    research_result = research_agent.invoke(
        {"messages": [("user", f"Original question: {query}\n\nResearch plan:\n{plan_json}")]}
    )
    research_output = extract_agent_output(research_result)

    critique_result = critic_agent.invoke({"messages": [("user", research_output)]})
    structured_critique = critique_result.get("structured_response")

    assert structured_critique is not None, "Critic did not return a structured response"
    assert structured_critique.verdict in ("APPROVE", "REVISE"), (
        f"Unexpected verdict: {structured_critique.verdict!r}"
    )

    _save_result({
        "input": query,
        "category": "happy_path_critique",
        "verdict": structured_critique.verdict,
        "is_fresh": structured_critique.is_fresh,
        "is_complete": structured_critique.is_complete,
        "is_well_structured": structured_critique.is_well_structured,
        "gaps": structured_critique.gaps,
        "revision_requests": structured_critique.revision_requests,
    })
