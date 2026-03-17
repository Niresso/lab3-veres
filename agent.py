from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report

TOOLS = [web_search, read_url, write_report]

_model = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
)

agent = create_react_agent(
    model=_model,
    tools=TOOLS,
    checkpointer=MemorySaver(),
    prompt=SYSTEM_PROMPT,
)
