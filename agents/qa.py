from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from dev_config import dev_settings, QA_PROMPT
from dev_schemas import ReviewOutput
from tools import web_search
from tools_repl import python_repl
from tools_fs import read_file, list_files

_model = ChatOpenAI(
    api_key=dev_settings.api_key.get_secret_value(),
    model=dev_settings.model_name,
)

qa_agent = create_agent(
    model=_model,
    tools=[python_repl, read_file, list_files, web_search],
    system_prompt=QA_PROMPT,
    response_format=ReviewOutput,
)
