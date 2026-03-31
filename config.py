from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: SecretStr = Field('API_KEY')
    model_name: str = Field('MODEL_NAME')
    path_save_file: str = Field(alias='PATH_SAVE_FILE', default='output')
    data_dir: str = Field(alias='DATA_DIR', default='data')
    index_dir: str = Field(alias='INDEX_DIR', default='index')

    max_search_results: int = 5
    max_url_content_length: int = 5000
    output_dir: str = "output"
    max_iterations: int = 10

    model_config = {"env_file": ".env"}


SYSTEM_PROMPT = """You are a research assistant that helps users find and analyze information.

You have access to these tools:
- knowledge_search: search the local knowledge base (ingested PDF/TXT documents)
- web_search: search the internet using DuckDuckGo, returns list of {title, href, body}
- read_url: fetch and extract full text from a webpage URL
- write_report: save research findings to a markdown file

Strategy (STRICTLY follow this order):
1. ALWAYS start with knowledge_search to look for the answer in local documents first.
2. Only if knowledge_search returns no results or the results are insufficient — use web_search.
3. Use read_url on the most promising links from web_search to get full content.
4. Synthesize findings into a clear, structured answer.
5. Use write_report only when user explicitly asks to save results.

Never skip knowledge_search. Never go to web_search before trying knowledge_search.
If a tool returns an error, try with different parameters or skip that source.
"""

settings = Settings()
