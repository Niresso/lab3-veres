"""
Sandboxed file-system tools for Developer and QA agents.

All paths are resolved relative to the configured workspace directory.
Directory traversal is blocked.
"""

import os

from langchain_core.tools import tool

from dev_config import dev_settings

_WORKSPACE = os.path.abspath(dev_settings.workspace_dir)


def _safe_path(relative_path: str) -> str | None:
    """Return absolute path inside workspace, or None if traversal detected."""
    abs_path = os.path.abspath(os.path.join(_WORKSPACE, relative_path))
    if not abs_path.startswith(_WORKSPACE + os.sep) and abs_path != _WORKSPACE:
        return None
    return abs_path


@tool
def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a file inside the project workspace.

    Use this to save source code, tests, requirements.txt, README, etc.

    Args:
        path: relative file path, e.g. "src/main.py" or "tests/test_main.py"
        content: full file content as a string

    Returns:
        Success message or error description
    """
    abs_path = _safe_path(path)
    if abs_path is None:
        return f"Error: path traversal detected for '{path}'"
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Written: {path} ({len(content)} chars)"
    except Exception as exc:  # noqa: BLE001
        return f"Error writing '{path}': {exc}"


@tool
def read_file(path: str) -> str:
    """
    Read a file from the project workspace.

    Use this to review source code, test files, or any other artefact.

    Args:
        path: relative file path, e.g. "src/main.py"

    Returns:
        File content (up to 10 000 chars) or error message
    """
    abs_path = _safe_path(path)
    if abs_path is None:
        return f"Error: path traversal detected for '{path}'"
    if not os.path.isfile(abs_path):
        return f"Error: file not found — '{path}'"
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            return fh.read(10_000)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading '{path}': {exc}"


@tool
def list_files(directory: str = ".") -> str:
    """
    List all files in a workspace directory (recursive).

    Args:
        directory: relative directory path (default: workspace root)

    Returns:
        Newline-separated list of relative file paths, or error message
    """
    abs_dir = _safe_path(directory)
    if abs_dir is None:
        return f"Error: path traversal detected for '{directory}'"
    if not os.path.isdir(abs_dir):
        return f"Error: directory not found — '{directory}'"
    try:
        files = []
        for root, _, names in os.walk(abs_dir):
            for name in names:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, _WORKSPACE)
                files.append(rel)
        return "\n".join(sorted(files)) or "(empty)"
    except Exception as exc:  # noqa: BLE001
        return f"Error listing '{directory}': {exc}"
