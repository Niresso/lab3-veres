from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from config import settings, SUPERVISOR_PROMPT
from tools import save_report
from agents.planner import planner_agent
from agents.research import research_agent
from agents.critic import critic_agent



@tool
def plan(request: str) -> str:
    """
    Викликає Planner Agent для декомпозиції запиту у структурований план дослідження.

    Args:
        request: повний запит користувача

    Returns:
        JSON-рядок з ResearchPlan (goal, search_queries, sources_to_check, output_format)
    """
    result = planner_agent.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2)
    return result["messages"][-1].content


@tool
def research(request: str) -> str:
    """
    Викликає Research Agent для виконання дослідження за планом.

    Args:
        request: план дослідження + оригінальне питання (або фідбек від Critic)

    Returns:
        Markdown-документ з результатами дослідження
    """
    result = research_agent.invoke({"messages": [("user", request)]})
    return result["messages"][-1].content


@tool
def critique(findings: str) -> str:
    """
    Викликає Critic Agent для оцінки якості дослідження.

    Args:
        findings: повний markdown-текст результатів дослідження

    Returns:
        JSON-рядок з CritiqueResult (verdict, is_fresh, is_complete, is_well_structured,
        strengths, gaps, revision_requests)
    """
    result = critic_agent.invoke({"messages": [("user", findings)]})
    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json(indent=2)
    return result["messages"][-1].content


_model = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
)

supervisor = create_agent(
    model=_model,
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
    ],
    checkpointer=MemorySaver(),
)
