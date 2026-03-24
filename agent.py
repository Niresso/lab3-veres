import json
import logging

from openai import OpenAI

from config import settings, SYSTEM_PROMPT
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.api_key.get_secret_value())

_history: list[dict] = []


def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with given args, returning a string result."""
    tool_function = TOOL_FUNCTIONS.get(name)
    if tool_function is None:
        return f"Error: unknown tool '{name}'"
    try:
        result = tool_function(**args)
        if isinstance(result, list):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {e}"


def run(user_message: str) -> str:
    """Run the ReAct loop for a single user turn.

    Appends the user message and assistant responses to the shared
    conversation history so context is preserved across calls.

    Args:
        user_message: the user's input text

    Returns:
        The agent's final text answer.
    """
    _history.append({"role": "user", "content": user_message})

    for iteration in range(1, settings.max_iterations + 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history

        response = _client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        choice = response.choices[0]
        openai_message = choice.message

        assistant_openai_message: dict = {"role": "assistant", "content": openai_message.content or ""}
        if openai_message.tool_calls:
            assistant_openai_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
                for tool_call in openai_message.tool_calls
            ]

        _history.append(assistant_openai_message)

        if not openai_message.tool_calls:
            return openai_message.content or ""

        for tc in openai_message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            print(f"  [{iteration}] -> {tool_name}({ str(tool_args)[:80]})")

            result = _execute_tool(tool_name, tool_args)

            preview = result[:120].replace("\n", " ")
            print(f"  [{iteration}] <- {tool_name}: {preview}...")

            _history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    logger.warning("Max iterations (%d) reached, requesting final answer.", settings.max_iterations)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history
    response = _client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
    )
    final = response.choices[0].message.content or ""
    _history.append({"role": "assistant", "content": final})
    return final


def reset_history() -> None:
    """Clear the conversation history (start a new session)."""
    _history.clear()
