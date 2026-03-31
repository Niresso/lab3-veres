import os
import logging
import trafilatura

from ddgs import DDGS
from config import settings

logging.basicConfig(level=logging.INFO)

os.makedirs(settings.path_save_file, exist_ok=True)


def web_search(query: str) -> list[dict]:
    """Search the internet via DuckDuckGo.

    Args:
        query: search query string

    Returns:
        List of up to 5 results, each with:
          - title (str): page title
          - href (str): page URL
          - body (str): short snippet/description

    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        if not results:
            return []
        for r in results:
            if "body" in r and len(r["body"]) > settings.max_snippet_length:
                r["body"] = r["body"][:settings.max_snippet_length] + "…"
        return results
    except Exception as e:
        return [{"error": str(e)}]


def read_url(url: str) -> str:
    """Download and extract text content from a webpage.

    Args:
        url: page address

    Returns:
        Extracted page text (string, up to max_url_content_length characters),
        or an error message string if fetching/extraction fails.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: could not fetch URL: {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Error: could not extract text from URL: {url}"
        return text[:settings.max_url_content_length]
    except Exception as e:
        return f"Error reading URL {url}: {e}"


def write_report(filename: str, content: str) -> str:
    """Save a text report to a file.

    Args:
        filename: file name (e.g. 'report.md')
        content: text content to save

    Returns:
        Success message with saved path, or error message string.
    """
    try:
        path = os.path.join(settings.path_save_file, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error writing report: {e}"


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet using DuckDuckGo. "
                "Use when you need current information on a topic. "
                "Returns a list of up to 5 results with title, href, and body snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Fetch and extract the full text content of a webpage by URL. "
                "Use after web_search when you need the complete article or page content. "
                "Returns extracted text (up to 5000 characters) or an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to read",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "Save research findings to a markdown file on disk. "
                "Use ONLY when the user explicitly asks to save or export results. "
                "Returns a success message with the file path, or an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name including extension, e.g. 'report.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown text content to write to the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]
