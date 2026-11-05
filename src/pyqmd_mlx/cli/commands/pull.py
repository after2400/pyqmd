"""`pyqmd pull` command."""

import typer

from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli.commands._model_output import download_with_progress, print_download_results
from pyqmd_mlx.llm._download import active_models

app = typer.Typer(help="Download the models pyqmd is set to use.")


@app.callback()
def _pull_callback() -> None:
    """Download the models pyqmd is set to use.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["pull", ...])`-style invocation (the
    literal "pull" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so this callback is not expected to be
    removed later.
    """


@app.command("pull")
def pull(
    refresh: bool = typer.Option(False, "--refresh", hidden=True),
    progress: bool = typer.Option(False, "--progress", hidden=True),
) -> None:
    """Download the embedding, rerank and expansion models pyqmd is set to use.

    Fetches only what a query would use under the current settings (flag-free:
    environment variables, `pyqmd models use`, the config file, then the
    default set), and never a model you did not select. Models already
    downloaded are skipped. It downloads files; it does not load a model.
    `--refresh` and `--progress` are Node qmd's flags and are accepted and
    ignored.
    """
    results = run_or_exit(lambda: download_with_progress(active_models()))
    if print_download_results(results):
        raise typer.Exit(1)
