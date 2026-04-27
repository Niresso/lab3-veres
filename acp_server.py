"""
ACP Server — port 8903
Agents: planner, researcher, critic
Each agent connects to SearchMCP (port 8901) to get its tools.
"""
import traceback

from acp_sdk import Message, MessagePart
from acp_sdk.server import Context, Server
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from fastmcp import Client as MCPClient

from config import settings, PLANNER_PROMPT, RESEARCH_PROMPT, CRITIC_PROMPT
from schemas import ResearchPlan, CritiqueResult

server = Server()

SEARCH_MCP_URL = f"http://localhost:{settings.search_mcp_port}/mcp"


def _make_model() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.model_name,
    )


def _last_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        for part in msg.parts:
            if hasattr(part, "content") and isinstance(part.content, str):
                return part.content
    return ""


# ── Planner ───────────────────────────────────────────────────────────────────

@server.agent(name="planner", description="Decomposes a user request into a structured ResearchPlan")
async def planner_agent(input: list[Message], context: Context):
    try:
        # Keep MCPClient open for the entire agent invocation
        async with MCPClient(SEARCH_MCP_URL) as client:
            tools = await load_mcp_tools(session=client.session)
            tools = [t for t in tools if t.name in ("web_search", "knowledge_search")]

            agent = create_agent(
                model=_make_model(),
                tools=tools,
                system_prompt=PLANNER_PROMPT,
                response_format=ResearchPlan,
            )
            user_text = _last_text(input)
            result = await agent.ainvoke({"messages": [("user", user_text)]})

        structured: ResearchPlan | None = result.get("structured_response")
        output = structured.model_dump_json(indent=2) if structured else result["messages"][-1].content
    except Exception as e:
        output = f"ERROR in planner: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    yield Message(role="agent", parts=[MessagePart(content=output)])


# ── Researcher ────────────────────────────────────────────────────────────────

@server.agent(name="researcher", description="Executes research according to the plan")
async def researcher_agent(input: list[Message], context: Context):
    try:
        async with MCPClient(SEARCH_MCP_URL) as client:
            tools = await load_mcp_tools(session=client.session)
            tools = [t for t in tools if t.name in ("web_search", "read_url", "knowledge_search")]

            agent = create_agent(
                model=_make_model(),
                tools=tools,
                system_prompt=RESEARCH_PROMPT,
            )
            user_text = _last_text(input)
            result = await agent.ainvoke({"messages": [("user", user_text)]})

        output = result["messages"][-1].content
    except Exception as e:
        output = f"ERROR in researcher: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    yield Message(role="agent", parts=[MessagePart(content=output)])


# ── Critic ────────────────────────────────────────────────────────────────────

@server.agent(name="critic", description="Evaluates research quality and returns a CritiqueResult")
async def critic_agent(input: list[Message], context: Context):
    try:
        async with MCPClient(SEARCH_MCP_URL) as client:
            tools = await load_mcp_tools(session=client.session)
            tools = [t for t in tools if t.name in ("web_search", "read_url", "knowledge_search")]

            agent = create_agent(
                model=_make_model(),
                tools=tools,
                system_prompt=CRITIC_PROMPT,
                response_format=CritiqueResult,
            )
            user_text = _last_text(input)
            result = await agent.ainvoke({"messages": [("user", user_text)]})

        structured: CritiqueResult | None = result.get("structured_response")
        output = structured.model_dump_json(indent=2) if structured else result["messages"][-1].content
    except Exception as e:
        output = f"ERROR in critic: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    yield Message(role="agent", parts=[MessagePart(content=output)])


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=settings.acp_port)
