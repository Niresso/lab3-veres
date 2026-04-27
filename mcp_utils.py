"""Helper: load MCP tools and convert them to LangChain BaseTool format."""
import asyncio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from fastmcp import Client as MCPClient


def get_mcp_tools(server_url: str, tool_names: list[str] | None = None) -> list[BaseTool]:
    """
    Connect to an MCP server, load tools, and return them as LangChain tools.

    Args:
        server_url: HTTP URL of the MCP server (e.g. 'http://localhost:8901/mcp')
        tool_names: optional whitelist of tool names to load; None means all tools

    Returns:
        List of LangChain BaseTool instances
    """
    async def _load() -> list[BaseTool]:
        async with MCPClient(server_url) as client:
            tools = await load_mcp_tools(session=client.session)
            if tool_names:
                tools = [t for t in tools if t.name in tool_names]
            return tools

    return asyncio.run(_load())
