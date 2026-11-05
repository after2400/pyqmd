"""`pyqmd models` -- choose and inspect the models pyqmd uses.

See docs/specs/2026-11-04-model-sets-and-config-design.md.
"""

import os

import typer

from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli.commands._model_output import download_with_progress, print_download_results
from pyqmd_mlx.llm._config import config_path, load_models_config, write_models_config
from pyqmd_mlx.llm._constants import (
    DEFAULT_EMBED_MODEL,
    EXPAND_MODEL_ENV_VAR,
    MODEL_SET_ENV_VAR,
    MODEL_SETS,
    RERANK_MODEL_ENV_VAR,
)
from pyqmd_mlx.llm._hub import is_downloaded
from pyqmd_mlx.llm._models import model_set, resolve_model_source

app = typer.Typer(
    help="Choose and inspect the models pyqmd uses.",
    invoke_without_command=True,
)

_ROLE_ORDER = (("embed", "Embedding"), ("rerank", "Reranking"), ("expand", "Generation"))


def _state(model_id: str) -> str:
    if os.path.isdir(model_id):
        return "local directory"
    return "downloaded" if is_downloaded(model_id) else "not downloaded"


def _show() -> None:
    path = config_path()
    typer.echo(f"Config file: {path}{'' if path.exists() else ' (not found)'}")
    typer.echo("Models:")
    for role, name in _ROLE_ORDER:
        if role == "embed":
            model_id, source = DEFAULT_EMBED_MODEL, "built in"
        else:
            model_id, source = resolve_model_source(role)
        typer.echo(f"  {name + ':':<13}{model_id}")
        typer.echo(f"  {'':<13}{source}; {_state(model_id)}")


@app.callback()
def _models_callback(ctx: typer.Context) -> None:
    """Choose and inspect the models pyqmd uses. Bare `pyqmd models` shows them."""
    if ctx.invoked_subcommand is None:
        run_or_exit(_show)


@app.command("show")
def show() -> None:
    """Show the models pyqmd will use, where each choice came from, and
    whether it is already downloaded. Never touches the network."""
    run_or_exit(_show)


@app.command("list")
def list_sets() -> None:
    """List the model sets you can choose with `pyqmd models use`."""

    def body() -> None:
        current = {role: resolve_model_source(role)[0] for role in ("expand", "rerank")}
        for name, ms in MODEL_SETS.items():
            active = ms.expand == current["expand"] and ms.rerank == current["rerank"]
            typer.echo(f"{name}{'  (active)' if active else ''}")
            typer.echo(f"  {ms.summary}")
            typer.echo(f"  generation: {ms.expand}")
            typer.echo(f"  reranking:  {ms.rerank}")

    run_or_exit(body)


@app.command("use")
def use(
    set_name: str = typer.Argument(
        ..., metavar="SET", help="default or granite (see `models list`)."
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Also download the chosen set's models (and the embedding model). "
        "Without it nothing is downloaded until a command needs a model.",
    ),
) -> None:
    """Choose a model set and remember it in the config file.

    Selecting a set never downloads, loads or contacts any model, in
    particular not the other set's: `pyqmd models use granite` is safe on a
    network that blocks the default models. Explicit `expand` / `rerank`
    entries in the config file are removed, since they would still win.
    """

    def body() -> bool:
        chosen = model_set(set_name, "the command line")
        existing = load_models_config()
        path = config_path()
        had_file = path.exists()
        removed = [key for key in ("expand", "rerank") if key in existing]
        updated = {k: v for k, v in existing.items() if k not in ("expand", "rerank")}
        updated["set"] = set_name
        write_models_config(updated)
        typer.echo(f"Using the '{set_name}' model set (saved in {path}).")
        typer.echo(f"  generation: {chosen.expand}")
        typer.echo(f"  reranking:  {chosen.rerank}")
        if removed:
            typer.echo(f"Removed explicit {' and '.join(removed)} entries from the config file.")
        if had_file:
            typer.echo("Comments in the config file are not preserved when it is rewritten.")
        for var in (MODEL_SET_ENV_VAR, EXPAND_MODEL_ENV_VAR, RERANK_MODEL_ENV_VAR):
            if os.environ.get(var):
                typer.echo(f"Note: {var} is set in this environment and still takes precedence.")
        if not download:
            return False
        typer.echo("Downloading:")
        models = [
            ("embed", DEFAULT_EMBED_MODEL),
            ("rerank", chosen.rerank),
            ("expand", chosen.expand),
        ]
        return print_download_results(download_with_progress(models))

    if run_or_exit(body):
        raise typer.Exit(1)
