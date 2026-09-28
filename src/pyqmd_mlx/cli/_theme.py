"""Shared ANSI color scheme, mirroring Node qmd's per-message `c.*` usage
(src/cli/qmd.ts) across status/get/collection/context/ls/update/embed/
cleanup. Thin wrappers over typer.style(); typer/click already strip ANSI
codes automatically when stdout isn't a TTY (piping/redirection just
works), so no extra color-disabling logic is needed here."""

import typer


def bold(text: str) -> str:
    return typer.style(text, bold=True)


def dim(text: str) -> str:
    return typer.style(text, dim=True)


def cyan(text: str) -> str:
    return typer.style(text, fg=typer.colors.CYAN)


def green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN)


def yellow(text: str) -> str:
    return typer.style(text, fg=typer.colors.YELLOW)
