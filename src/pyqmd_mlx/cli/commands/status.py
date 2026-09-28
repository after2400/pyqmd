"""`pyqmd status` command."""

import os

import typer

from pyqmd_mlx.cli._db import DEFAULT_DB_PATH, get_store
from pyqmd_mlx.cli._format import format_bytes, format_time_ago
from pyqmd_mlx.cli._theme import bold, cyan, dim, green, yellow
from pyqmd_mlx.store._ast import get_ast_status

app = typer.Typer(help="Show index status.")


@app.callback()
def _status_callback() -> None:
    """Show index status.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["status", ...])`-style invocation (the
    literal "status" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so this callback is not expected to be
    removed later.
    """


def _hf_link(model_id: str) -> str:
    """Turn an `org/repo`-shaped model id into an https://huggingface.co/
    link, matching Node's own hf: link transform (src/cli/qmd.ts). Anything
    else -- an absolute filesystem path, like pyqmd's locally-converted
    expansion model -- is returned unchanged."""
    if model_id.startswith("/") or model_id.count("/") != 1:
        return model_id
    return f"https://huggingface.co/{model_id}"


@app.command("status")
def status() -> None:
    """Show index status and collections."""
    store = get_store()
    db_path = store.db_path if store.db_path != ":memory:" else str(DEFAULT_DB_PATH)
    try:
        size = os.path.getsize(db_path)
    except OSError:
        size = 0

    counts = store.get_status_counts()
    collections = store.list_collections()
    collection_stats = store.get_collection_document_stats()

    typer.echo(bold("pyqmd Status") + "\n")
    typer.echo(f"Index: {db_path}")
    typer.echo(f"Size:  {format_bytes(size)}\n")

    typer.echo(bold("Documents"))
    typer.echo(f"  Total:    {counts['active_documents']} files indexed")
    typer.echo(f"  Vectors:  {counts['embedded_vectors']} embedded")
    orphaned = store.count_orphaned_content()
    if orphaned > 0:
        typer.echo(yellow(f"  Orphaned: {orphaned} content hash(es)") + " — run 'pyqmd cleanup'")
    if counts["pending_embed"] > 0:
        typer.echo(
            yellow(f"  Pending:  {counts['pending_embed']} need embedding") + " (run 'pyqmd embed')"
        )
    if counts["most_recent_modified_at"]:
        typer.echo(f"  Updated:  {format_time_ago(counts['most_recent_modified_at'])}")

    ast_status = get_ast_status()
    typer.echo("\n" + bold("AST Chunking"))
    if ast_status.available:
        available_langs = [lang.language for lang in ast_status.languages if lang.available]
        unavailable_langs = [lang for lang in ast_status.languages if not lang.available]
        typer.echo(f"  Status:    {green('active')}")
        typer.echo(f"  Languages: {', '.join(available_langs)}")
        for lang in unavailable_langs:
            typer.echo(yellow(f"  Unavailable: {lang.language} ({lang.error})"))
    else:
        typer.echo(f"  Status:    {yellow('unavailable')} (falling back to regex chunking)")
        for lang in ast_status.languages:
            if lang.error:
                typer.echo(dim(f"  {lang.language}: {lang.error}"))

    typer.echo("\n" + bold("Collections"))
    if not collections:
        typer.echo("  (none)")

    contexts_by_collection: dict[str, list[dict]] = {}
    for ctx in store.list_all_contexts():
        contexts_by_collection.setdefault(ctx["collection"], []).append(ctx)

    for c in collections:
        name = c["name"]
        stats = collection_stats.get(name, {"count": 0, "latest_modified": None})
        typer.echo(f"  {cyan(name)} {dim(f'(qmd://{name}/)')}")
        typer.echo(f"    {dim('Pattern:')}  {c['pattern']}")
        if stats["latest_modified"]:
            updated = format_time_ago(stats["latest_modified"])
            typer.echo(f"    {dim('Files:')}    {stats['count']} (updated {updated})")
        else:
            typer.echo(f"    {dim('Files:')}    {stats['count']}")

        collection_contexts = contexts_by_collection.get(name, [])
        if collection_contexts:
            typer.echo(f"    {dim('Contexts:')} {len(collection_contexts)}")
            for ctx in collection_contexts:
                path_display = f"/{ctx['path']}" if ctx["path"] else "/"
                preview = ctx["context"]
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                typer.echo(f"      {dim(path_display + ':')} {preview}")

    if collections:
        first = collections[0]["name"]
        typer.echo("\n" + bold("Examples"))
        typer.echo(dim("  # List files in a collection"))
        typer.echo(f"  pyqmd ls {first}")
        typer.echo(dim("  # Get a document"))
        typer.echo(f"  pyqmd get qmd://{first}/path/to/file.md")
        typer.echo(dim("  # Search within a collection"))
        typer.echo(f'  pyqmd search "query" -c {first}')

    typer.echo("\n" + bold("Models"))
    typer.echo(f"  Embedding:   {_hf_link(store._embed_model)}")
    typer.echo(f"  Reranking:   {_hf_link(store._rerank_model)}")
    typer.echo(f"  Generation:  {_hf_link(store._expand_model)}")

    tips: list[str] = []
    without_context = [c["name"] for c in collections if c["name"] not in contexts_by_collection]
    if without_context:
        names = ", ".join(without_context[:3])
        more = f" +{len(without_context) - 3} more" if len(without_context) > 3 else ""
        tips.append(f"Add context to collections for better search results: {names}{more}")
        tips.append(dim('  pyqmd context add qmd://<name>/ "What this collection contains"'))

    if len(collections) > 1:
        without_update = [c["name"] for c in collections if not c["update_command"]]
        if without_update:
            names = ", ".join(without_update[:3])
            more = f" +{len(without_update) - 3} more" if len(without_update) > 3 else ""
            tips.append(f"Add update commands to keep collections fresh: {names}{more}")
            tips.append(
                dim(
                    "  pyqmd collection update-cmd <name> "
                    "'git stash && git pull --rebase --ff-only && git stash pop'"
                )
            )

    if tips:
        typer.echo("\n" + bold("Tips"))
        for tip in tips:
            typer.echo(f"  {tip}")
