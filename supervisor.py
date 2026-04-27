"""
Supervisor — local create_agent that orchestrates ACP sub-agents.
Tools:
  - delegate_to_planner / delegate_to_researcher / delegate_to_critic → ACP (port 8903)
  - save_report → ReportMCP (port 8902), HITL-gated
"""
import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from acp_sdk.client import Client as ACPClient
from fastmcp import Client as MCPClient

from config import settings, SUPERVISOR_PROMPT

ACP_BASE_URL = f"http://localhost:{settings.acp_port}"
REPORT_MCP_URL = f"http://localhost:{settings.report_mcp_port}/mcp"


# ── ACP helper ────────────────────────────────────────────────────────────────

async def _call_acp_agent(agent_name: str, message: str) -> str:
    async with ACPClient(base_url=ACP_BASE_URL) as client:
        run = await client.run_sync(input=message, agent=agent_name)
        run.raise_for_status()
        for msg in run.output:
            for part in msg.parts:
                if hasattr(part, "content") and isinstance(part.content, str):
                    return part.content
    return "(no output)"


# ── ACP delegation tools ──────────────────────────────────────────────────────

@tool
def delegate_to_planner(request: str) -> str:
    """
    Delegate to the Planner Agent via ACP.
    Returns a structured ResearchPlan as JSON.

    Args:
        request: full user request
    """
    return asyncio.run(_call_acp_agent("planner", request))


@tool
def delegate_to_researcher(request: str) -> str:
    """
    Delegate to the Research Agent via ACP.
    Returns markdown research findings.

    Args:
        request: research plan JSON + original question (and optional critic feedback)
    """
    return asyncio.run(_call_acp_agent("researcher", request))


@tool
def delegate_to_critic(findings: str) -> str:
    """
    Delegate to the Critic Agent via ACP.
    Returns a structured CritiqueResult as JSON (verdict, gaps, revision_requests, …).

    Args:
        findings: full markdown research findings to evaluate
    """
    return asyncio.run(_call_acp_agent("critic", findings))


# ── ReportMCP save_report tool ────────────────────────────────────────────────

async def _load_report_tool():
    async with MCPClient(REPORT_MCP_URL) as client:
        tools = await load_mcp_tools(session=client.session)
        return next(t for t in tools if t.name == "save_report")


save_report = asyncio.run(_load_report_tool())


# ── Supervisor ────────────────────────────────────────────────────────────────

_model = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
)

supervisor = create_agent(
    model=_model,
    tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, save_report],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
    ],
    checkpointer=MemorySaver(),
)
