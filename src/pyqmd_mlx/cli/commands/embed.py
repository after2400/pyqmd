"""`pyqmd embed` command."""

from time import monotonic

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import CliError, run_or_exit
from pyqmd_mlx.cli._format import format_eta
from pyqmd_mlx.cli._progress import EmbedProgress, final_bar_line
from pyqmd_mlx.cli._theme import bold, dim, green, yellow
from pyqmd_mlx.cli._types import ChunkStrategy

app = typer.Typer(help="Generate embeddings for indexed documents.")


@app.callback()
def _embed_callback() -> None:
    """Generate embeddings for indexed documents.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["embed", ...])`-style invocation (the
    literal "embed" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so unlike documents.py this callback
    is not expected to be removed later.
    """


@app.command("embed")
def embed(
    collection: str = typer.Option(None, "-c", "--collection"),
    force: bool = typer.Option(False, "--force", help="Re-embed even already-current content."),
    chunk_strategy: ChunkStrategy = typer.Option(
        ChunkStrategy.REGEX,
        "--chunk-strategy",
        help='Chunking mode: "regex" (default) or "auto" (AST-aware for supported code files).',
    ),
) -> None:
    """Generate vector embeddings for indexed documents."""
    store = get_store()

    def _check_collection_exists() -> None:
        if collection and store.get_collection(collection) is None:
            raise CliError(f"Collection not found: {collection}")

    run_or_exit(_check_collection_exists)

    if force:
        typer.echo(yellow("Force re-indexing: clearing all vectors..."))
        store.clear_embeddings(collection)
    elif store.count_pending_embed(collection, chunk_strategy=chunk_strategy.value) == 0:
        typer.echo(green("✓ All content hashes already have embeddings."))
        return

    content = store.get_indexable_content(collection)
    if not content:
        typer.echo(green("✓ No non-empty documents to embed."))
        return

    typer.echo(dim(f"Model: {store.embed_model.rsplit('/', 1)[-1]}") + "\n")
    doc_sizes = [len(row["doc"].encode("utf-8")) for row in content]
    start = monotonic()
    docs_processed = 0
    chunks_embedded = 0
    bytes_processed = 0
    with EmbedProgress(total_bytes=sum(doc_sizes)) as progress:
        for row, size in zip(content, doc_sizes, strict=True):
            chunk_count = store.index_content(
                row["hash"], row["doc"], filepath=row["path"], chunk_strategy=chunk_strategy.value
            )
            docs_processed += 1
            chunks_embedded += chunk_count
            bytes_processed += size
            progress.update(bytes_processed, chunks_embedded)
    elapsed = monotonic() - start

    typer.echo(final_bar_line())
    typer.echo(
        f"\n{green('✓ Done!')} Embedded {bold(str(chunks_embedded))} chunks from "
        f"{bold(str(docs_processed))} documents in {bold(format_eta(elapsed))}"
    )
