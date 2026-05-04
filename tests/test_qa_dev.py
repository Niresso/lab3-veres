"""
LLM-as-a-Judge tests for the QA Engineer agent.

Key test: submit intentionally bad code → QA must detect the issues.
Also tests that QA verdict is consistent with findings.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_dev import extract_output, extract_tool_names, skip_no_key

qa_issue_detection = GEval(
    name="QA Issue Detection",
    evaluation_steps=[
        "The 'input' contains a specification and intentionally flawed code.",
        "Check that 'actual output' (ReviewOutput) identifies the specific bugs present in the code.",
        "Check that issues list is non-empty and each issue is specific (not generic).",
        "Check that suggestions are actionable — each one tells the developer what to change.",
        "If verdict is REVISION_NEEDED, check that acceptance_criteria_failed is non-empty.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)

qa_verdict_consistency = GEval(
    name="QA Verdict Consistency",
    evaluation_steps=[
        "If verdict is REVISION_NEEDED: issues must be non-empty and acceptance_criteria_failed must be non-empty.",
        "If verdict is APPROVED: score should be ≥ 0.7 and acceptance_criteria_failed should be empty or contain only minor items.",
        "Check that score is between 0.0 and 1.0.",
        "Check that verdict matches the overall quality assessment in the issues list.",
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.8,
)


SPEC = {
    "title": "Email Validator",
    "requirements": [
        "validate_email(email: str) -> bool returns True for valid emails",
        "Returns False for empty string",
        "Returns False for email missing @ symbol",
        "Returns False for email missing domain",
        "Handles None input without raising exceptions",
    ],
    "acceptance_criteria": [
        "validate_email('user@example.com') == True",
        "validate_email('') == False",
        "validate_email('nodomain@') == False",
        "validate_email('nodot') == False",
        "validate_email(None) returns False (no exception)",
    ],
    "tech_stack": ["re"],
    "estimated_complexity": "simple",
    "coding_standards": ["type hints", "docstrings"],
}

# This code has multiple known issues:
# 1. No None guard — will crash on None input
# 2. Accepts 'nodomain@' as valid (only checks for @)
# 3. No type hints
# 4. No docstring
BAD_CODE_CONTENT = """\
def validate_email(email):
    return '@' in email
"""

GOOD_CODE_CONTENT = """\
import re

def validate_email(email: str | None) -> bool:
    \"\"\"Return True if email is a valid RFC-style address, False otherwise.\"\"\"
    if not isinstance(email, str) or not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""


@pytest.fixture(scope="module")
def qa():
    from agents.qa import qa_agent
    return qa_agent


def _write_test_file(filename: str, content: str) -> None:
    from dev_config import dev_settings
    path = os.path.join(dev_settings.workspace_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _build_qa_request(spec: dict, code_content: str, filename: str, iteration: int = 1) -> str:
    code_output = {
        "description": "Email validator implementation",
        "files_created": [filename],
        "source_code": code_content,
        "test_results": "(not tested)",
    }
    return (
        f"Specification:\n{json.dumps(spec, indent=2)}\n\n"
        f"Developer submission (iteration {iteration}/5):\n"
        f"{json.dumps(code_output, indent=2)}\n\n"
        "Review the code against every acceptance criterion."
    )


@skip_no_key
def test_qa_detects_issues_in_bad_code(qa):
    """QA must identify problems in intentionally flawed code."""
    _write_test_file("email_validator.py", BAD_CODE_CONTENT)
    request = _build_qa_request(SPEC, BAD_CODE_CONTENT, "email_validator.py")

    result = qa.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    output = structured.model_dump_json(indent=2) if structured else extract_output(result)

    test_case = LLMTestCase(input=request, actual_output=output)
    assert_test(test_case, [qa_issue_detection])


@skip_no_key
def test_qa_verdict_is_revision_on_bad_code(qa):
    """Bad code should receive REVISION_NEEDED verdict."""
    _write_test_file("email_validator_bad.py", BAD_CODE_CONTENT)
    request = _build_qa_request(SPEC, BAD_CODE_CONTENT, "email_validator_bad.py")

    result = qa.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")

    assert structured is not None, "QA must return a structured ReviewOutput"
    assert structured.verdict == "REVISION_NEEDED", (
        f"Bad code should fail review. Got verdict: {structured.verdict}, "
        f"score: {structured.score}, issues: {structured.issues}"
    )


@skip_no_key
def test_qa_approves_good_code(qa):
    """Good code that satisfies all acceptance criteria should be APPROVED."""
    _write_test_file("email_validator_good.py", GOOD_CODE_CONTENT)
    request = _build_qa_request(SPEC, GOOD_CODE_CONTENT, "email_validator_good.py")

    result = qa.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")

    assert structured is not None
    # Score should be high for good code
    assert structured.score >= 0.6, (
        f"Good code should score ≥ 0.6. Got {structured.score}. Issues: {structured.issues}"
    )


@skip_no_key
def test_qa_verdict_consistency(qa):
    _write_test_file("email_validator_c.py", BAD_CODE_CONTENT)
    request = _build_qa_request(SPEC, BAD_CODE_CONTENT, "email_validator_c.py")

    result = qa.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    output = structured.model_dump_json(indent=2) if structured else extract_output(result)

    test_case = LLMTestCase(input=request, actual_output=output)
    assert_test(test_case, [qa_verdict_consistency])


@skip_no_key
def test_qa_uses_repl_and_read_file(qa):
    _write_test_file("email_validator_r.py", BAD_CODE_CONTENT)
    request = _build_qa_request(SPEC, BAD_CODE_CONTENT, "email_validator_r.py")

    result = qa.invoke({"messages": [("user", request)]})
    tools = extract_tool_names(result)

    assert tools & {"python_repl", "read_file"}, (
        f"QA must use python_repl or read_file. Called: {tools}"
    )
