"""
LangGraph StateGraph implementing the BA → HITL → Developer ↔ QA pipeline.

Pattern: Evaluator-Optimizer (Anthropic)
- BA produces SpecOutput → user approves (HITL gate)
- Developer produces CodeOutput → QA evaluates
- QA returns ReviewOutput; if REVISION_NEEDED and iterations < max → back to Developer
"""

import json
import uuid
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from dev_config import dev_settings


# ── State ─────────────────────────────────────────────────────────────────────

class DevState(TypedDict):
    user_story: str
    spec_feedback: str              # accumulated user feedback for BA revisions
    spec: dict | None               # SpecOutput.model_dump()
    code: dict | None               # CodeOutput.model_dump()
    review: dict | None             # ReviewOutput.model_dump()
    iteration: int                  # Developer→QA iteration counter (1-based)
    session_id: str
    messages: Annotated[list, add_messages]


# ── Helper ────────────────────────────────────────────────────────────────────

def _extract_output(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if content and not getattr(msg, "tool_calls", None):
            return content if isinstance(content, str) else str(content)
    return ""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def ba_node(state: DevState) -> dict:
    from agents.ba import ba_agent

    user_story = state["user_story"]
    feedback = state.get("spec_feedback", "")

    request = user_story
    if feedback:
        request = (
            f"Original user story:\n{user_story}\n\n"
            f"User feedback on the previous specification:\n{feedback}\n\n"
            "Please revise and return an improved SpecOutput."
        )

    result = ba_agent.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    spec_data = structured.model_dump() if structured else {}

    return {
        "spec": spec_data,
        "messages": result["messages"],
    }


def hitl_spec_review(state: DevState) -> Command:
    """Pause and ask the user to approve the spec or provide feedback."""
    spec = state.get("spec", {})

    user_input = interrupt({
        "spec": spec,
        "instructions": (
            "Review the specification above.\n"
            "Type 'approve' to proceed to development, "
            "or describe what needs to change."
        ),
    })

    decision = str(user_input).strip().lower()
    if decision == "approve":
        return Command(goto="developer")
    return Command(
        update={"spec_feedback": str(user_input)},
        goto="ba",
    )


def developer_node(state: DevState) -> Command:
    from agents.developer import developer_agent

    spec = state.get("spec", {})
    review = state.get("review")
    iteration = state.get("iteration", 0)

    request = (
        f"Specification:\n{json.dumps(spec, indent=2)}\n\n"
        f"Workspace directory: {dev_settings.workspace_dir}"
    )

    if review and review.get("verdict") == "REVISION_NEEDED":
        issues = "\n".join(f"  - {i}" for i in review.get("issues", []))
        suggestions = "\n".join(f"  - {s}" for s in review.get("suggestions", []))
        failed = "\n".join(f"  - {c}" for c in review.get("acceptance_criteria_failed", []))
        request += (
            f"\n\n⚠️  QA REVISION NEEDED — iteration {iteration}/{dev_settings.max_qa_iterations}\n"
            f"Issues:\n{issues}\n\n"
            f"Suggestions:\n{suggestions}\n\n"
            f"Failed acceptance criteria:\n{failed}\n\n"
            "Fix all issues and resubmit."
        )

    result = developer_agent.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    code_data = structured.model_dump() if structured else {}

    return Command(
        update={
            "code": code_data,
            "iteration": iteration + 1,
            "messages": result["messages"],
        },
        goto="qa",
    )


def qa_node(state: DevState) -> Command:
    from agents.qa import qa_agent

    spec = state.get("spec", {})
    code = state.get("code", {})
    iteration = state.get("iteration", 1)

    request = (
        f"Specification:\n{json.dumps(spec, indent=2)}\n\n"
        f"Developer submission (iteration {iteration}/{dev_settings.max_qa_iterations}):\n"
        f"{json.dumps(code, indent=2)}\n\n"
        f"Workspace directory: {dev_settings.workspace_dir}\n"
        "Review the code against every acceptance criterion. "
        "Read files, run tests, check edge cases."
    )

    result = qa_agent.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    review_data = structured.model_dump() if structured else {
        "verdict": "APPROVED",
        "score": 0.5,
        "issues": [],
        "suggestions": [],
        "acceptance_criteria_met": [],
        "acceptance_criteria_failed": [],
    }

    verdict = review_data.get("verdict", "APPROVED")
    if verdict == "APPROVED" or iteration >= dev_settings.max_qa_iterations:
        return Command(
            update={"review": review_data, "messages": result["messages"]},
            goto=END,
        )
    return Command(
        update={"review": review_data, "messages": result["messages"]},
        goto="developer",
    )


# ── Graph ─────────────────────────────────────────────────────────────────────

_builder = StateGraph(DevState)
_builder.add_node("ba", ba_node)
_builder.add_node("hitl_spec_review", hitl_spec_review)
_builder.add_node("developer", developer_node)
_builder.add_node("qa", qa_node)

_builder.add_edge(START, "ba")
_builder.add_edge("ba", "hitl_spec_review")
# hitl_spec_review, developer, and qa use Command for dynamic routing

_checkpointer = MemorySaver()
dev_graph = _builder.compile(checkpointer=_checkpointer)


def make_config(thread_id: str | None = None) -> dict:
    tid = thread_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": tid}}
