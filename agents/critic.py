from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import settings, CRITIC_PROMPT
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search

_model = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
)

critic_agent = create_agent(
    model=_model,
    tools=[web_search, read_url, knowledge_search],
    system_prompt=CRITIC_PROMPT,
    response_format=CritiqueResult,
)
