"""Index-scan reporting shared by `collection add` and `update`, worded to
match Node qmd's indexFiles/updateCollections output (src/cli/qmd.ts)."""

import typer


def echo_index_summary(result) -> None:
    typer.echo(
        f"\nIndexed: {result.indexed} new, {result.updated} updated, "
        f"{result.unchanged} unchanged, {result.removed} removed"
    )
    if result.skipped:
        typer.echo(f"Skipped {result.skipped} file(s):", err=True)
        for skipped_path, reason in result.skipped_files:
            typer.echo(f"  {skipped_path}: {reason}", err=True)
    if result.orphaned_cleaned > 0:
        typer.echo(f"Cleaned up {result.orphaned_cleaned} orphaned content hash(es)")


def echo_embed_hint(store) -> None:
    pending = store.get_status_counts()["pending_embed"]
    if pending > 0:
        typer.echo(
            f"\nRun 'pyqmd embed' to update embeddings ({pending} unique hashes need vectors)"
        )
