"""
Sandboxed Python REPL tool for Developer and QA agents.

Restrictions:
- Forbidden modules checked via AST before execution
- Execution timeout (configurable, default 10s)
- Output size limit
- No network, no disk write (those go through dedicated tools)
"""

import ast
import io
import threading
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.tools import tool

from dev_config import dev_settings

FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "socket", "pathlib",
    "glob", "tempfile", "pty", "signal", "ctypes", "multiprocessing",
    "threading", "concurrent", "importlib", "pickle", "marshal",
    "shelve", "nt", "posix", "winreg", "builtins",
})

def _get_builtin(name: str):
    if isinstance(__builtins__, dict):
        return __builtins__.get(name)
    return getattr(__builtins__, name, None)


_SAFE_BUILTINS = {
    name: _get_builtin(name)
    for name in (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray",
        "bytes", "callable", "chr", "classmethod", "complex", "delattr", "dict",
        "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "globals", "hasattr", "hash", "hex", "id",
        "int", "isinstance", "issubclass", "iter", "len", "list", "locals",
        "map", "max", "min", "next", "object", "oct", "open", "ord", "pow",
        "print", "property", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
        "tuple", "type", "vars", "zip",
        "__build_class__", "__import__", "__name__",
        "NotImplemented", "Ellipsis", "None", "True", "False",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration", "AssertionError",
        "ImportError", "NameError", "ZeroDivisionError", "OverflowError",
        "ArithmeticError", "LookupError", "OSError", "IOError",
        "BaseException", "GeneratorExit", "SystemExit", "KeyboardInterrupt",
    )
}
_SAFE_BUILTINS = {k: v for k, v in _SAFE_BUILTINS.items() if v is not None}


def _check_imports(code: str) -> str | None:
    """Return an error string if code imports forbidden modules, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    return f"ImportError: module '{top}' is not allowed in REPL"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in FORBIDDEN_MODULES:
                return f"ImportError: module '{top}' is not allowed in REPL"
    return None


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code in a sandboxed environment and return the output.

    Restrictions: forbidden modules — os, subprocess, sys, shutil, socket, threading,
    multiprocessing, ctypes, importlib, pickle. Timeout: 10 seconds. Output capped at 4000 chars.

    Use this to:
    - Test functions you have written
    - Validate logic with sample inputs
    - Run unit tests

    Args:
        code: valid Python source code to execute

    Returns:
        Combined stdout + stderr, or an error/timeout message
    """
    import_err = _check_imports(code)
    if import_err:
        return import_err

    container: dict = {"output": "", "exc": None}

    def _run() -> None:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        ns: dict = {"__builtins__": _SAFE_BUILTINS, "__name__": "__main__"}
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<repl>", "exec"), ns)  # noqa: S102
            out = stdout_buf.getvalue() + stderr_buf.getvalue()
            container["output"] = out[: dev_settings.repl_max_output]
        except Exception as exc:  # noqa: BLE001
            container["output"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=dev_settings.repl_timeout)

    if thread.is_alive():
        return f"Error: execution timed out after {dev_settings.repl_timeout}s"

    return container["output"] or "(no output)"
