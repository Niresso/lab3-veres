from agent import agent
from config import settings

THREAD_ID = "session-1"

CONFIG = {
    "configurable": {"thread_id": THREAD_ID},
    "recursion_limit": settings.max_iterations * 2 + 1,
}


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

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

        try:
            final_answer = None

            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config=CONFIG,
                stream_mode="updates",
            ):
                if "agent" in chunk:
                    for msg in chunk["agent"]["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_str = str(tc["args"])[:80]
                                print(f"  -> {tc['name']}({args_str})")
                        elif getattr(msg, "content", None):
                            final_answer = msg.content

                if "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        preview = str(msg.content)[:120].replace("\n", " ")
                        print(f"  <- {msg.name}: {preview}...")

            if final_answer:
                print(f"\nAgent: {final_answer}")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
