"""
ACP Server — port 8903
Agents: planner, researcher, critic
Each agent connects to SearchMCP (port 8901) to get its tools.
"""
import asyncio

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


async def _get_search_tools(names: list[str] | None = None):
    """Connect to SearchMCP and return tools as LangChain tools."""
    async with MCPClient(SEARCH_MCP_URL) as client:
        tools = await load_mcp_tools(session=client.session)
        if names:
            tools = [t for t in tools if t.name in names]
        return tools


def _last_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        for part in msg.parts:
            if hasattr(part, "content") and isinstance(part.content, str):
                return part.content
    return ""


# ── Planner ────────────────────────────────────────────────────────────────────

@server.agent(name="planner", description="Decomposes a user request into a structured ResearchPlan")
async def planner_agent(input: list[Message], context: Context):
    tools = await _get_search_tools(["web_search", "knowledge_search"])

    agent = create_agent(
        model=_make_model(),
        tools=tools,
        system_prompt=PLANNER_PROMPT,
        response_format=ResearchPlan,
    )

    user_text = _last_text(input)
    result = agent.invoke({"messages": [("user", user_text)]})

    structured: ResearchPlan | None = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = result["messages"][-1].content

    yield Message(role="agent", parts=[MessagePart(content=output)])


# ── Researcher ─────────────────────────────────────────────────────────────────

@server.agent(name="researcher", description="Executes research according to the plan")
async def researcher_agent(input: list[Message], context: Context):
    tools = await _get_search_tools(["web_search", "read_url", "knowledge_search"])

    agent = create_agent(
        model=_make_model(),
        tools=tools,
        system_prompt=RESEARCH_PROMPT,
    )

    user_text = _last_text(input)
    result = agent.invoke({"messages": [("user", user_text)]})
    output = result["messages"][-1].content

    yield Message(role="agent", parts=[MessagePart(content=output)])


# ── Critic ─────────────────────────────────────────────────────────────────────

@server.agent(name="critic", description="Evaluates research quality and returns a CritiqueResult")
async def critic_agent(input: list[Message], context: Context):
    tools = await _get_search_tools(["web_search", "read_url", "knowledge_search"])

    agent = create_agent(
        model=_make_model(),
        tools=tools,
        system_prompt=CRITIC_PROMPT,
        response_format=CritiqueResult,
    )

    user_text = _last_text(input)
    result = agent.invoke({"messages": [("user", user_text)]})

    structured: CritiqueResult | None = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json(indent=2)
    else:
        output = result["messages"][-1].content

    yield Message(role="agent", parts=[MessagePart(content=output)])


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=settings.acp_port)
