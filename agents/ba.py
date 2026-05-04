from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from dev_config import dev_settings, BA_PROMPT
from dev_schemas import SpecOutput
from tools import web_search, knowledge_search

_model = ChatOpenAI(
    api_key=dev_settings.api_key.get_secret_value(),
    model=dev_settings.model_name,
)

ba_agent = create_agent(
    model=_model,
    tools=[web_search, knowledge_search],
    system_prompt=BA_PROMPT,
    response_format=SpecOutput,
)
