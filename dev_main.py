"""
Entry point for the AI Development Team multi-agent system.

Workflow:
  User → BA → [HITL spec review] → Developer → QA → (loop ≤5×) → final report
"""

import json
import sys
import io
import uuid

from langgraph.types import Command

from dev_graph import dev_graph, make_config, DevState

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _stream(input_, config: dict) -> tuple[str | None, dict | None]:
    """
    Stream the graph until completion or HITL interrupt.
    Returns (final_text, interrupt_data); exactly one is non-None.
    """
    final_text = None
    interrupt_data = None

    for chunk in dev_graph.stream(input_, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_data = chunk["__interrupt__"][0].value
            break

        for node_name, node_output in chunk.items():
            if node_name.startswith("_"):
                continue
            # Print tool calls as they happen
            for msg in node_output.get("messages", []):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        print(f"  [{node_name}] → {tc['name']}({str(tc['args'])[:80]})")
                elif getattr(msg, "content", None) and not getattr(msg, "name", None):
                    content = str(msg.content)
                    if content.strip():
                        final_text = content
                        print(f"  [{node_name}] ← {content[:120]}")

    return final_text, interrupt_data


def _show_spec(spec: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  SPECIFICATION: {spec.get('title', '(untitled)')}")
    print("=" * 60)
    print(f"  Complexity: {spec.get('estimated_complexity', '?')}")
    print("\nRequirements:")
    for req in spec.get("requirements", []):
        print(f"  • {req}")
    print("\nAcceptance Criteria:")
    for crit in spec.get("acceptance_criteria", []):
        print(f"  ✓ {crit}")
    print("\nTech Stack:", ", ".join(spec.get("tech_stack", [])))
    print("=" * 60)


def _handle_hitl(interrupt_data: dict, config: dict) -> str | None:
    spec = interrupt_data.get("spec", {})
    _show_spec(spec)

    print("\n" + interrupt_data.get("instructions", "Review the spec above."))
    print("Options: type 'approve' or describe what needs to change.\n")

    while True:
        try:
            raw = input("Decision: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "approve"

        if not raw:
            continue

        resume_value = "approve" if raw.lower() == "approve" else raw
        final, next_interrupt = _stream(Command(resume=resume_value), config)

        if next_interrupt:
            return _handle_hitl(next_interrupt, config)
        return final


def _show_review(review: dict) -> None:
    verdict = review.get("verdict", "?")
    score = review.get("score", 0.0)
    symbol = "✅" if verdict == "APPROVED" else "🔄"
    print(f"\n{symbol}  QA Verdict: {verdict}  (score: {score:.2f})")

    if review.get("acceptance_criteria_met"):
        print("Passed criteria:")
        for c in review["acceptance_criteria_met"]:
            print(f"  ✓ {c}")

    if review.get("acceptance_criteria_failed"):
        print("Failed criteria:")
        for c in review["acceptance_criteria_failed"]:
            print(f"  ✗ {c}")

    if review.get("issues"):
        print("Issues:")
        for i in review["issues"]:
            print(f"  ⚠ {i}")

    if review.get("suggestions"):
        print("Suggestions:")
        for s in review["suggestions"]:
            print(f"  → {s}")


def run(user_story: str, thread_id: str | None = None) -> dict:
    """
    Run the full pipeline for a user story.
    Returns the final graph state.
    """
    config = make_config(thread_id)
    session_id = config["configurable"]["thread_id"]

    initial_state: DevState = {
        "user_story": user_story,
        "spec_feedback": "",
        "spec": None,
        "code": None,
        "review": None,
        "iteration": 0,
        "session_id": session_id,
        "messages": [],
    }

    print(f"\n{'='*60}")
    print(f"  AI DEV TEAM  |  session: {session_id[:8]}")
    print(f"{'='*60}")
    print(f"  User story: {user_story}\n")

    final_text, interrupt_data = _stream({"messages": [("user", user_story)], **{k: v for k, v in initial_state.items() if k != "messages"}}, config)

    if interrupt_data:
        final_text = _handle_hitl(interrupt_data, config)

    # Retrieve final state for the caller
    final_state = dev_graph.get_state(config)
    values = final_state.values if final_state else {}

    review = values.get("review", {})
    if review:
        _show_review(review)

    code = values.get("code", {})
    if code and code.get("files_created"):
        print(f"\nFiles created: {', '.join(code['files_created'])}")

    print(f"\n{'='*60}\n")
    return values


def main() -> None:
    print("AI Development Team — Multi-Agent System")
    print("Type a user story to start. 'quit' to exit.\n")

    while True:
        try:
            story = input("User story: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not story or story.lower() in ("quit", "exit"):
            break

        try:
            run(story)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
