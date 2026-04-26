from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import settings, PLANNER_PROMPT
from schemas import ResearchPlan
from tools import web_search, knowledge_search

_model = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
)

planner_agent = create_agent(
    model=_model,
    tools=[web_search, knowledge_search],
    system_prompt=PLANNER_PROMPT,
    response_format=ResearchPlan,
)
