import sys
import io
import uuid

from langgraph.types import Command

from supervisor import supervisor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def stream_supervisor(input_, config: dict) -> tuple[str | None, dict | None]:
    """
    Stream supervisor until completion or HITL interrupt.
    Returns (final_answer, interrupt_data) — exactly one is non-None.
    """
    final_answer = None
    interrupt_data = None

    for chunk in supervisor.stream(input_, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_data = chunk["__interrupt__"][0].value
            break

        if "agent" in chunk:
            for msg in chunk["agent"]["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_str = str(tc["args"])[:100]
                        print(f"  -> {tc['name']}({args_str})")
                elif getattr(msg, "content", None):
                    final_answer = msg.content

        if "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                preview = str(msg.content)[:120].replace("\n", " ")
                print(f"  <- {msg.name}: {preview}...")

    return final_answer, interrupt_data


def show_interrupt(interrupt_data: dict) -> None:
    action_requests = interrupt_data.get("action_requests", [])
    if not action_requests:
        print("\n[HITL] save_report requires approval.")
        return

    req = action_requests[0]
    filename = req.get("args", {}).get("filename", "report.md")
    content = req.get("args", {}).get("content", "")
    preview = content[:600].replace("\n", "\n    ")

    print("\n" + "=" * 60)
    print(f"[HITL] Supervisor wants to save: {filename}")
    print("-" * 60)
    print(f"    {preview}")
    if len(content) > 600:
        print(f"    ... ({len(content)} chars total)")
    print("=" * 60)


def handle_hitl(interrupt_data: dict, config: dict) -> str | None:
    show_interrupt(interrupt_data)

    while True:
        print("\nOptions: [approve] / [edit <your feedback>] / [reject]")
        try:
            raw = input("Decision: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "reject"

        if raw.lower() == "approve":
            resume = {"decisions": [{"type": "approve"}]}
            print("  Approved — saving report...")
            final, next_interrupt = stream_supervisor(Command(resume=resume), config)
            if next_interrupt:
                return handle_hitl(next_interrupt, config)
            return final

        elif raw.lower().startswith("edit"):
            feedback = raw[4:].strip()
            if not feedback:
                print("  Provide feedback after 'edit', e.g.: edit add more examples")
                continue
            resume = {
                "decisions": [
                    {"type": "reject", "message": f"User requested changes: {feedback}"}
                ]
            }
            print("  Sending feedback to Supervisor for revision...")
            final, next_interrupt = stream_supervisor(Command(resume=resume), config)
            if next_interrupt:
                return handle_hitl(next_interrupt, config)
            return final

        elif raw.lower().startswith("reject"):
            reason = raw[6:].strip() or "User rejected the report."
            resume = {"decisions": [{"type": "reject", "message": reason}]}
            print("  Rejected — cancelling save.")
            final, next_interrupt = stream_supervisor(Command(resume=resume), config)
            if next_interrupt:
                return handle_hitl(next_interrupt, config)
            return final

        else:
            print("  Unknown option. Type 'approve', 'edit <feedback>', or 'reject'.")


def main():
    print("Multi-Agent Research System (hw9)")
    print("Supervisor → ACP(Planner/Researcher/Critic) → MCP(Search/Report)")
    print("Type 'exit' to quit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())
        config = make_config(thread_id)

        try:
            final, interrupt_data = stream_supervisor(
                {"messages": [("user", user_input)]}, config
            )

            if interrupt_data:
                final = handle_hitl(interrupt_data, config)

            if final:
                print(f"\nAgent: {final}")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
