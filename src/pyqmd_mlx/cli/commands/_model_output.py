"""Shared printing for `pull` and `models`."""

import typer

from pyqmd_mlx.llm._download import LABELS, DownloadResult


def print_download_results(results: list[DownloadResult]) -> bool:
    """One line per model; each failure's message goes to stderr under it.
    Returns True if any model failed."""
    failed = False
    for r in results:
        typer.echo(f"  {LABELS[r.role]:<16}{r.status:<12}{r.model_id}")
        if r.status == "failed":
            failed = True
            typer.echo(f"Error: {r.message}", err=True)
    return failed
