"""Shared printing for `pull` and `models`."""

import sys

import typer

from pyqmd_mlx.llm._download import LABELS, DownloadResult, download_models


def download_with_progress(models: list[tuple[str, str]]) -> list[DownloadResult]:
    """download_models, announcing each real download on stderr before
    huggingface_hub draws its progress bar."""
    announced = 0

    def announce(role: str, model_id: str) -> None:
        nonlocal announced
        # The previous bar leaves the cursor mid-line (it redraws with \r).
        prefix = "\n" if announced else ""
        announced += 1
        typer.echo(f"{prefix}Downloading {LABELS[role]} model {model_id}", err=True)

    return download_models(models, on_start=announce)


def print_download_results(results: list[DownloadResult]) -> bool:
    """One line per model; each failure's message goes to stderr under it.
    Returns True if any model failed."""
    if any(r.status in ("downloaded", "failed") for r in results):
        # End the last progress bar's line so the table starts on a fresh one.
        sys.stderr.flush()
        typer.echo("", err=True)
    failed = False
    for r in results:
        typer.echo(f"  {LABELS[r.role]:<16}{r.status:<12}{r.model_id}")
        if r.status == "failed":
            failed = True
            typer.echo(f"Error: {r.message}", err=True)
    return failed
