"""Shared error handling for CLI commands: catch expected Store-boundary
exceptions (bad input, missing resource) and exit cleanly with a one-line
message; let anything else propagate with its real traceback, matching the
original CLI's exitWithError pattern -- report the error, but never
silently swallow a real bug."""

import sqlite3
from collections.abc import Callable
from typing import TypeVar

import typer

from pyqmd_mlx.llm import ExpansionModelError

T = TypeVar("T")

EXPECTED_EXCEPTIONS = (
    sqlite3.IntegrityError,
    ValueError,
    FileNotFoundError,
    NotADirectoryError,
    ExpansionModelError,
)


class CliError(Exception):
    """An expected, user-facing failure whose stderr text is already shaped
    to match Node qmd's: one or more lines (optionally styled via
    pyqmd_mlx.cli._theme), printed as-is with no "Error: " prefix. Raise it
    inside a run_or_exit callable."""

    def __init__(self, *lines: str) -> None:
        if not lines:
            raise ValueError("CliError needs at least one line")
        super().__init__("\n".join(lines))
        self.lines = lines


def run_or_exit(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except CliError as exc:
        for line in exc.lines:
            typer.echo(line, err=True)
        raise typer.Exit(1) from exc
    except EXPECTED_EXCEPTIONS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
