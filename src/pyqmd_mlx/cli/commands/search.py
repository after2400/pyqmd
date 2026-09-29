"""`pyqmd search`/`vsearch`/`query` commands."""

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli._fullpath import apply_full_path, format_fullpath_warning
from pyqmd_mlx.cli._index_health import check_index_health
from pyqmd_mlx.cli._metadata_filter import parse_cli_metadata_filter
from pyqmd_mlx.cli._output_search import format_search_results
from pyqmd_mlx.cli._types import ChunkStrategy, DisplayResult, OutputFormat

app = typer.Typer(help="Search commands.")

# Shared across search/vsearch/query so the three commands' `-n` defaults
# can never silently drift apart again (query's own default was 10 until
# 2026-09-13, discovered as a real parity bug against Node's actual shared
# default of 20 for these commands).
DEFAULT_SEARCH_LIMIT = 20


def _search_result_to_display(r) -> DisplayResult:
    # r.filepath is already the qmd://-prefixed form (see SearchResult in
    # pyqmd_mlx/store/_types.py, populated by search_fts/search_vec's own SQL) --
    # r.display_path is the bare "collection/path" form Node never actually
    # renders in its own output (confirmed against src/cli/qmd.ts's
    # toQmdPath()). Using filepath here, not display_path, is the fix.
    return DisplayResult(
        docid=r.docid,
        score=r.score,
        display_path=r.filepath,
        title=r.title,
        body=r.body,
        chunk_pos=r.chunk_pos,
        context=r.context,
    )


def _hybrid_result_to_display(r) -> DisplayResult:
    # Same fix as _search_result_to_display above, but HybridQueryResult's
    # already-prefixed field is named `file`, not `filepath`.
    return DisplayResult(
        docid=r.docid,
        score=r.score,
        display_path=r.file,
        title=r.title,
        body=r.body,
        chunk_pos=r.best_chunk_pos,
        context=r.context,
        metadata=r.metadata,
    )


@app.command("search")
def search(
    query: str = typer.Argument(...),
    collection: list[str] = typer.Option(None, "-c", "--collection"),
    limit: int = typer.Option(DEFAULT_SEARCH_LIMIT, "-n"),
    min_score: float = typer.Option(0.0, "--min-score"),
    filter: str = typer.Option(None, "--filter"),
    full_path: bool = typer.Option(False, "--full-path"),
    full: bool = typer.Option(False, "--full"),
    line_numbers: bool = typer.Option(False, "--line-numbers"),
    format: OutputFormat = typer.Option(OutputFormat.CLI, "--format"),
) -> None:
    """Full-text keyword search (BM25, no LLM)."""
    parsed_filter = parse_cli_metadata_filter(filter)
    store = get_store()
    results = store.search_fts(query, limit=limit, collection=collection, filter=parsed_filter)
    results = [r for r in results if r.score >= min_score]
    display = [_search_result_to_display(r) for r in results]
    if full_path:
        unresolved = apply_full_path(display, store)
        if unresolved:
            typer.echo(format_fullpath_warning(unresolved), err=True)
    typer.echo(
        format_search_results(
            display, format.value, query=query, full=full, line_numbers=line_numbers
        )
    )


@app.command("vsearch")
def vsearch(
    query: str = typer.Argument(...),
    collection: list[str] = typer.Option(None, "-c", "--collection"),
    limit: int = typer.Option(DEFAULT_SEARCH_LIMIT, "-n"),
    min_score: float = typer.Option(0.0, "--min-score"),
    filter: str = typer.Option(None, "--filter"),
    full_path: bool = typer.Option(False, "--full-path"),
    full: bool = typer.Option(False, "--full"),
    line_numbers: bool = typer.Option(False, "--line-numbers"),
    format: OutputFormat = typer.Option(OutputFormat.CLI, "--format"),
) -> None:
    """Vector similarity search (no reranking)."""
    parsed_filter = parse_cli_metadata_filter(filter)
    store = get_store()
    check_index_health(store)
    results = store.search_vec(query, limit=limit, collection=collection, filter=parsed_filter)
    results = [r for r in results if r.score >= min_score]
    display = [_search_result_to_display(r) for r in results]
    if full_path:
        unresolved = apply_full_path(display, store)
        if unresolved:
            typer.echo(format_fullpath_warning(unresolved), err=True)
    typer.echo(
        format_search_results(
            display, format.value, query=query, full=full, line_numbers=line_numbers
        )
    )


@app.command("query")
def query(
    query: str = typer.Argument(...),
    collection: list[str] = typer.Option(None, "-c", "--collection"),
    limit: int = typer.Option(DEFAULT_SEARCH_LIMIT, "-n"),
    min_score: float = typer.Option(0.0, "--min-score"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    intent: str = typer.Option(None, "--intent"),
    filter: str = typer.Option(None, "--filter"),
    full_path: bool = typer.Option(False, "--full-path"),
    full: bool = typer.Option(False, "--full"),
    line_numbers: bool = typer.Option(False, "--line-numbers"),
    format: OutputFormat = typer.Option(OutputFormat.CLI, "--format"),
    chunk_strategy: ChunkStrategy = typer.Option(
        ChunkStrategy.REGEX,
        "--chunk-strategy",
        help='Chunking mode for best-chunk selection: "regex" (default) or "auto".',
    ),
) -> None:
    """Search with query expansion and reranking (recommended)."""
    parsed_filter = parse_cli_metadata_filter(filter)
    store = get_store()
    check_index_health(store)
    results = run_or_exit(
        lambda: store.query(
            query,
            limit=limit,
            min_score=min_score,
            collection=collection,
            skip_rerank=no_rerank,
            intent=intent,
            filter=parsed_filter,
            chunk_strategy=chunk_strategy.value,
        )
    )
    display = [_hybrid_result_to_display(r) for r in results]
    if full_path:
        unresolved = apply_full_path(display, store)
        if unresolved:
            typer.echo(format_fullpath_warning(unresolved), err=True)
    typer.echo(
        format_search_results(
            display, format.value, query=query, full=full, line_numbers=line_numbers, intent=intent
        )
    )
