"""
ReportMCP — port 8902
Tools: save_report
Resources: output-dir
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from config import settings

mcp = FastMCP(name="ReportMCP")

os.makedirs(settings.path_save_file, exist_ok=True)


@mcp.tool
def save_report(filename: str, content: str) -> str:
    """
    Save the final research report to a file.

    Args:
        filename: file name, e.g. 'report.md'
        content: full report text in markdown

    Returns:
        Success message or error description
    """
    try:
        path = os.path.join(settings.path_save_file, filename)
        content_clean = content.encode("utf-8", errors="replace").decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content_clean)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error writing report: {e}"


@mcp.resource("resource://output-dir")
def output_dir_info() -> dict:
    """Returns the output directory path and list of saved reports."""
    try:
        files = [
            f for f in os.listdir(settings.path_save_file)
            if os.path.isfile(os.path.join(settings.path_save_file, f))
        ]
        return {"path": settings.path_save_file, "reports": sorted(files)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.report_mcp_port)
