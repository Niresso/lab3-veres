from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from dev_config import dev_settings, DEVELOPER_PROMPT
from dev_schemas import CodeOutput
from tools import web_search
from tools_repl import python_repl
from tools_fs import write_file, read_file, list_files

_model = ChatOpenAI(
    api_key=dev_settings.api_key.get_secret_value(),
    model=dev_settings.model_name,
)

developer_agent = create_agent(
    model=_model,
    tools=[web_search, python_repl, write_file, read_file, list_files],
    system_prompt=DEVELOPER_PROMPT,
    response_format=CodeOutput,
)
