"""MCP-specific error handling: convert the same exception set the CLI's
run_or_exit catches into a clean is_error CallToolResult, instead of an
unhandled exception -- an MCP server can't exit the process on one bad
tool call, it has to keep serving other requests."""

import sqlite3
from collections.abc import Callable
from typing import TypeVar

from mcp.types import CallToolResult, TextContent

from pyqmd_mlx.llm import ExpansionModelError

T = TypeVar("T", bound=CallToolResult)

EXPECTED_EXCEPTIONS = (
    sqlite3.IntegrityError,
    sqlite3.OperationalError,
    ValueError,
    FileNotFoundError,
    NotADirectoryError,
    ExpansionModelError,
)


def error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def tool_result_or_error(fn: Callable[[], T]) -> T | CallToolResult:
    try:
        return fn()
    except EXPECTED_EXCEPTIONS as exc:
        return error_result(f"Error: {exc}")
