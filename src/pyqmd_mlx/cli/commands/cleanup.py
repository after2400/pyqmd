"""`pyqmd cleanup` command."""

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._theme import green

app = typer.Typer(help="Clear the LLM cache, purge inactive documents, and vacuum the database.")


@app.callback()
def _cleanup_callback() -> None:
    """Clear the LLM cache, purge inactive documents, and vacuum the database.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["cleanup", ...])`-style invocation (the
    literal "cleanup" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so this callback is not expected to be
    removed later.
    """


@app.command("cleanup")
def cleanup(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be removed, without changing anything."
    ),
) -> None:
    """Clear the LLM cache, purge inactive documents, remove orphaned
    content, and compact/vacuum the database."""
    store = get_store()

    if dry_run:
        typer.echo("Dry run — no changes made.\n")
        cache_count = store.count_llm_cache()
        if cache_count > 0:
            typer.echo(f"Would clear {cache_count} cached API responses")
        inactive_count = store.count_inactive_documents()
        if inactive_count > 0:
            typer.echo(f"Would remove {inactive_count} inactive document record(s)")
        orphaned_count = store.count_orphaned_content()
        if orphaned_count > 0:
            typer.echo(f"Would clean up {orphaned_count} orphaned content hash(es)")
        typer.echo("Would compact FTS and vacuum the database")
        return

    cache_count = store.clear_llm_cache()
    if cache_count > 0:
        typer.echo(f"{green('✓')} Cleared {cache_count} cached API responses")

    inactive_count = store.purge_inactive_documents()
    if inactive_count > 0:
        typer.echo(f"{green('✓')} Removed {inactive_count} inactive document record(s)")

    orphaned_count = store.cleanup_orphaned_content()
    if orphaned_count > 0:
        typer.echo(f"{green('✓')} Cleaned up {orphaned_count} orphaned content hash(es)")

    store.optimize_documents_fts()
    store.vacuum()
    typer.echo(f"{green('✓')} FTS compacted, database vacuumed")
