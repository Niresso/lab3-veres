"""
End-to-end pipeline tests: User story → BA → Developer → QA (without HITL).

Runs the BA→Developer→QA chain programmatically (bypassing the HITL gate
that requires interactive input) and evaluates the final output quality.

Results are saved to tests/e2e_dev_results.json.
"""

import json
import time
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from conftest_dev import extract_output, load_golden, skip_no_key

RESULTS_PATH = Path(__file__).parent / "e2e_dev_results.json"

e2e_quality = GEval(
    name="E2E Pipeline Quality",
    evaluation_steps=[
        "The 'input' is a user story. The 'actual output' is the final ReviewOutput JSON.",
        "Check that the review contains acceptance_criteria_met with at least one passing criterion.",
        "Check that the final code (visible in review or code context) addresses the user story.",
        "Check that the pipeline produced a REVISION_NEEDED or APPROVED verdict (not a crash).",
        "Penalise if the review shows 0 criteria met and no useful feedback.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.6,
)

final_code_correctness = GEval(
    name="Final Code Correctness",
    evaluation_steps=[
        "The 'input' is the user story. The 'expected output' describes expected behaviour.",
        "The 'actual output' is the source code produced by Developer.",
        "Check whether the actual code would produce the described expected behaviour.",
        "Penalise for missing edge case handling described in expected_code_behavior.",
        "Different implementation approaches are acceptable as long as the behaviour matches.",
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)


def _save_result(entry: dict) -> None:
    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    results.append(entry)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_pipeline_no_hitl(user_story: str) -> tuple[dict, dict, dict]:
    """
    Run BA → Developer → QA without HITL.
    Returns (spec_data, code_data, review_data).
    """
    from agents.ba import ba_agent
    from agents.developer import developer_agent
    from agents.qa import qa_agent
    from dev_config import dev_settings

    # BA
    ba_result = ba_agent.invoke({"messages": [("user", user_story)]})
    spec = ba_result.get("structured_response")
    spec_data = spec.model_dump() if spec else {}

    # Developer
    dev_request = (
        f"Specification:\n{json.dumps(spec_data, indent=2)}\n\n"
        f"Workspace directory: {dev_settings.workspace_dir}"
    )
    dev_result = developer_agent.invoke({"messages": [("user", dev_request)]})
    code = dev_result.get("structured_response")
    code_data = code.model_dump() if code else {}

    # QA
    qa_request = (
        f"Specification:\n{json.dumps(spec_data, indent=2)}\n\n"
        f"Developer submission (iteration 1/5):\n"
        f"{json.dumps(code_data, indent=2)}\n\n"
        f"Workspace directory: {dev_settings.workspace_dir}\n"
        "Review the code against every acceptance criterion."
    )
    qa_result = qa_agent.invoke({"messages": [("user", qa_request)]})
    review = qa_result.get("structured_response")
    review_data = review.model_dump() if review else {}

    return spec_data, code_data, review_data


HAPPY_PATH = load_golden("happy_path")


@skip_no_key
@pytest.mark.parametrize(
    "example",
    HAPPY_PATH[:3],
    ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:3]],
)
def test_e2e_pipeline_produces_review(example):
    """Full pipeline must complete and produce a structured ReviewOutput."""
    start = time.time()
    spec_data, code_data, review_data = run_pipeline_no_hitl(example["user_story"])
    elapsed = time.time() - start

    assert review_data, "Pipeline produced no review output"
    assert review_data.get("verdict") in ("APPROVED", "REVISION_NEEDED"), (
        f"Unexpected verdict: {review_data.get('verdict')}"
    )

    _save_result({
        "user_story": example["user_story"],
        "category": example["category"],
        "verdict": review_data.get("verdict"),
        "score": review_data.get("score"),
        "criteria_met": review_data.get("acceptance_criteria_met", []),
        "criteria_failed": review_data.get("acceptance_criteria_failed", []),
        "files_created": code_data.get("files_created", []),
        "elapsed_seconds": round(elapsed, 1),
    })


@skip_no_key
@pytest.mark.parametrize(
    "example",
    HAPPY_PATH[:2],
    ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:2]],
)
def test_e2e_final_code_correctness(example):
    """Developer code should satisfy the expected behaviour from the golden dataset."""
    _, code_data, _ = run_pipeline_no_hitl(example["user_story"])

    source_code = code_data.get("source_code", "")
    assert source_code, "Developer returned no source code"

    test_case = LLMTestCase(
        input=example["user_story"],
        actual_output=source_code,
        expected_output=example["expected_code_behavior"],
    )
    assert_test(test_case, [final_code_correctness])


@skip_no_key
@pytest.mark.parametrize(
    "example",
    HAPPY_PATH[:2],
    ids=[ex["user_story"][:50] for ex in HAPPY_PATH[:2]],
)
def test_e2e_review_quality(example):
    """QA review should be meaningful: issues or criteria_met must be non-empty."""
    _, code_data, review_data = run_pipeline_no_hitl(example["user_story"])

    test_case = LLMTestCase(
        input=example["user_story"],
        actual_output=json.dumps(review_data, indent=2),
    )
    assert_test(test_case, [e2e_quality])


@skip_no_key
def test_e2e_repl_sandboxing():
    """Python REPL must block forbidden modules."""
    from tools_repl import python_repl

    result = python_repl.invoke({"code": "import os; print(os.getcwd())"})
    assert "ImportError" in result or "not allowed" in result.lower(), (
        f"REPL must block 'os' import. Got: {result}"
    )

    result_ok = python_repl.invoke({"code": "import math; print(math.sqrt(16))"})
    assert "4.0" in result_ok, f"REPL must allow 'math'. Got: {result_ok}"


@skip_no_key
def test_e2e_repl_timeout():
    """Python REPL must enforce the execution timeout."""
    from tools_repl import python_repl

    result = python_repl.invoke({"code": "while True: pass"})
    assert "timed out" in result.lower(), f"REPL must time out infinite loops. Got: {result}"


@skip_no_key
def test_e2e_workspace_tools():
    """write_file and read_file must work correctly in the sandbox."""
    from tools_fs import write_file, read_file

    write_result = write_file.invoke({"path": "test_sandbox.txt", "content": "hello workspace"})
    assert "Written" in write_result, f"write_file failed: {write_result}"

    read_result = read_file.invoke({"path": "test_sandbox.txt"})
    assert read_result == "hello workspace", f"read_file returned: {read_result!r}"

    # Path traversal must be blocked
    traversal = write_file.invoke({"path": "../../etc/passwd", "content": "pwned"})
    assert "traversal" in traversal.lower() or "error" in traversal.lower(), (
        f"Path traversal must be blocked. Got: {traversal}"
    )
