"""pyqmd: the Python CLI entry point. Mounts every command module onto one
Typer app. `collection` and `context` are named sub-groups
(collection: add/list/show/remove/rename; context: add/list/remove);
everything else (get, multi-get, ls, search, vsearch, query, embed, status)
mounts flat as a top-level command, matching the original CLI's
`qmd <command>` shape.

`embed.app`, `status.app`, `update.app`, `cleanup.app`, and `mcp.app` are
each permanently single-command Typer apps that carry a no-op `@app.callback()` (added in
Tasks 13/14 so they can be invoked as `pyqmd embed` / `pyqmd status`
instead of collapsing into a callback-only app). Mounting either via
`app.add_typer(embed.app)` / `app.add_typer(status.app)` (flat, no `name=`)
would make Typer detect that callback and emit `UserWarning: The
'callback' parameter is not supported by Typer when using add_typer
without a name`, silently dropping the callback -- and that warning would
fire on every real invocation of this CLI, including unrelated commands
and `--help`. To avoid that, we register the already `@app.command()`-
decorated functions directly on the root app instead of merging their
sub-Typer apps; `app.command()` returns its function unmodified, so this
is safe.
"""

import typer

from pyqmd_mlx.cli.commands import (
    cleanup,
    collection,
    context,
    documents,
    embed,
    mcp,
    pull,
    search,
    status,
    update,
)

app = typer.Typer(
    help="pyqmd: hybrid search over your markdown collections.",
    # Node qmd accepts -h too; child command contexts inherit this.
    context_settings={"help_option_names": ["--help", "-h"]},
)

app.add_typer(collection.app, name="collection")
app.add_typer(context.app, name="context")
app.add_typer(documents.app)
app.add_typer(search.app)

app.command("embed")(embed.embed)
app.command("status")(status.status)
app.command("update")(update.update)
app.command("cleanup")(cleanup.cleanup)
app.command("mcp")(mcp.mcp)
app.command("pull")(pull.pull)


def _version_callback(value: bool) -> None:
    if value:
        from pyqmd_mlx.version import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the pyqmd version and exit.",
    ),
) -> None:
    """pyqmd: hybrid search over your markdown collections."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
