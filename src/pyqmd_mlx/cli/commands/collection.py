"""`pyqmd collection` subcommands: add, list, show, remove, rename."""

from datetime import UTC, datetime

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import CliError, run_or_exit
from pyqmd_mlx.cli._format import format_time_ago
from pyqmd_mlx.cli._index_report import echo_embed_hint, echo_index_summary
from pyqmd_mlx.cli._theme import bold, cyan, dim, green, yellow
from pyqmd_mlx.store._indexing import scan_and_register_collection

app = typer.Typer(help="Manage collections.")


@app.command("add")
def add(
    path: str = typer.Argument(..., help="Directory to index."),
    name: str = typer.Option(..., "--name", help="Collection name."),
    mask: str = typer.Option(
        "**/*.md", "--mask", help="Glob pattern (comma-separated for multiple)."
    ),
    exclude: list[str] = typer.Option([], "--exclude", help="Additional ignore glob pattern(s)."),
    update_cmd: str = typer.Option(
        None, "--update-cmd", help="Command to run before each future 'pyqmd update'."
    ),
) -> None:
    """Add a new collection and index its documents."""
    store = get_store()

    def _check_no_duplicate() -> None:
        if store.get_collection(name) is not None:
            raise CliError(
                yellow(f"Collection '{name}' already exists."),
                "Use a different name with --name <name>",
            )
        for c in store.list_collections():
            if c["path"] == path and c["pattern"] == mask:
                raise CliError(
                    yellow("A collection already exists for this path and pattern:"),
                    f"  Name: {c['name']} (qmd://{c['name']}/)",
                    f"  Pattern: {mask}",
                    f"\nUse 'pyqmd update' to re-index it, or remove it first with "
                    f"'pyqmd collection remove {c['name']}'",
                )

    def _register():
        store.add_collection(
            name,
            path,
            pattern=mask,
            ignore_patterns=",".join(exclude) if exclude else None,
            update_command=update_cmd,
        )
        return scan_and_register_collection(store, path, mask, name, ignore_patterns=list(exclude))

    run_or_exit(_check_no_duplicate)
    typer.echo(f"Creating collection '{name}'...")
    typer.echo(f"Collection: {path} ({mask})")
    result = run_or_exit(_register)
    echo_index_summary(result)
    echo_embed_hint(store)
    typer.echo(f"{green('✓')} Collection '{name}' created successfully")


@app.command("list")
def list_collections() -> None:
    """List all collections."""
    store = get_store()
    collections = store.list_collections()
    if not collections:
        typer.echo("No collections found. Run 'pyqmd collection add .' to create one.")
        return

    stats = store.get_collection_document_stats()
    typer.echo(bold(f"Collections ({len(collections)}):") + "\n")
    for c in collections:
        name = c["name"]
        excluded_tag = yellow(" [excluded]") if not c["include_by_default"] else ""
        typer.echo(f"{cyan(name)} {dim(f'(qmd://{name}/)')}{excluded_tag}")
        typer.echo(f"  {dim('Pattern:')}  {c['pattern']}")
        if c["ignore_patterns"]:
            ignore_display = ", ".join(c["ignore_patterns"].split(","))
            typer.echo(f"  {dim('Ignore:')}   {ignore_display}")
        collection_stats = stats.get(name, {"count": 0, "latest_modified": None})
        typer.echo(f"  {dim('Files:')}    {collection_stats['count']}")
        updated = collection_stats["latest_modified"] or datetime.now(UTC).isoformat()
        typer.echo(f"  {dim('Updated:')}  {format_time_ago(updated)}")
        typer.echo()


@app.command("show")
def show(name: str = typer.Argument(...)) -> None:
    """Show details for a single collection."""
    store = get_store()

    def _run():
        collection = store.get_collection(name)
        if collection is None:
            raise CliError(f"Collection not found: {name}")
        return collection

    collection = run_or_exit(_run)
    doc_count = len(store.get_active_document_paths(name))
    context_count = sum(1 for ctx in store.list_all_contexts() if ctx["collection"] == name)
    include_state = "yes (default)" if collection["include_by_default"] else "no"
    # Uncolored, like Node's `collection show`. "Documents:" is a
    # pyqmd-only superset line (declared in the parity flow's text_subs).
    typer.echo(f"Collection: {collection['name']}")
    typer.echo(f"  Path:     {collection['path']}")
    typer.echo(f"  Pattern:  {collection['pattern']}")
    typer.echo(f"  Include:  {include_state}")
    if collection["update_command"]:
        typer.echo(f"  Update:   {collection['update_command']}")
    if context_count:
        typer.echo(f"  Contexts: {context_count}")
    typer.echo(f"  Documents: {doc_count}")


@app.command("remove")
def remove(
    name: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove a collection and delete its indexed documents."""
    store = get_store()

    def _check():
        collection = store.get_collection(name)
        if collection is None:
            raise CliError(
                yellow(f"Collection not found: {name}"),
                "Run 'pyqmd collection list' to see available collections.",
            )
        return collection

    run_or_exit(_check)
    doc_count = len(store.get_active_document_paths(name))
    if not yes and not typer.confirm(
        f"Remove '{name}' and its {doc_count} document(s)? This cannot be undone."
    ):
        typer.echo("Cancelled.")
        raise typer.Exit(0)
    removal = store.remove_collection_detailed(name)
    typer.echo(f"{green('✓')} Removed collection '{name}'")
    typer.echo(f"  Deleted {removal.deleted_docs} documents")
    if removal.cleaned_hashes > 0:
        typer.echo(f"  Cleaned up {removal.cleaned_hashes} orphaned content hashes")


@app.command("rename")
def rename(old_name: str = typer.Argument(...), new_name: str = typer.Argument(...)) -> None:
    """Rename a collection."""
    store = get_store()

    def _run():
        if store.get_collection(old_name) is None:
            raise CliError(
                yellow(f"Collection not found: {old_name}"),
                "Run 'pyqmd collection list' to see available collections.",
            )
        if store.get_collection(new_name) is not None:
            raise CliError(
                yellow(f"Collection name already exists: {new_name}"),
                "Choose a different name or remove the existing collection first.",
            )
        store.rename_collection(old_name, new_name)

    run_or_exit(_run)
    typer.echo(f"{green('✓')} Renamed collection '{old_name}' to '{new_name}'")
    typer.echo(
        f"  Virtual paths updated: {cyan(f'qmd://{old_name}/')} → {cyan(f'qmd://{new_name}/')}"
    )


@app.command("update-cmd")
def update_cmd(
    name: str = typer.Argument(...),
    command: list[str] = typer.Argument(None, help="Command to run before re-indexing."),
) -> None:
    """Set or clear a collection's pre-update hook command."""
    store = get_store()
    cmd = " ".join(command or []).strip() or None

    def _run():
        if store.get_collection(name) is None:
            raise CliError(f"Collection not found: {name}")
        store.set_collection_update_command(name, cmd)

    run_or_exit(_run)
    if cmd:
        typer.echo(f"✓ Set update command for '{name}': {cmd}")
    else:
        typer.echo(f"✓ Cleared update command for '{name}'")


def _set_include(name: str, include: bool) -> None:
    store = get_store()

    def _run():
        if store.get_collection(name) is None:
            raise CliError(f"Collection not found: {name}")
        store.set_collection_include_by_default(name, include)

    run_or_exit(_run)
    state = "included in" if include else "excluded from"
    typer.echo(f"✓ Collection '{name}' {state} default queries")


@app.command("include")
def include(name: str = typer.Argument(...)) -> None:
    """Include a collection in default (no -c) search results."""
    _set_include(name, True)


@app.command("exclude")
def exclude(name: str = typer.Argument(...)) -> None:
    """Exclude a collection from default (no -c) search results."""
    _set_include(name, False)
