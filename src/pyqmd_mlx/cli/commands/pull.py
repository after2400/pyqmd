"""`pyqmd pull` command."""

import typer

app = typer.Typer(help="No-op stub: pyqmd's MLX models download automatically on first use.")


@app.callback()
def _pull_callback() -> None:
    """No-op stub: pyqmd's MLX models download automatically on first use.

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
    """Explain that pyqmd has no separate model-download step.

    Node's `qmd pull [--refresh] [--progress]` downloads embedding/
    generation/rerank GGUF model files. pyqmd's MLX models load straight
    from the Hugging Face Hub cache on first use, so there's nothing to
    pull -- this stub exists only so someone typing `pull` out of Node
    habit gets an explanation instead of a "no such command" error.
    `--refresh`/`--progress` are accepted and silently ignored for the
    same reason.
    """
    typer.echo(
        "pyqmd's MLX models download automatically on first use — no separate pull step is needed."
    )
