"""`pyqmd get`/`multi-get`/`ls` commands."""

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli._format import format_bytes, format_ls_time
from pyqmd_mlx.cli._fullpath import apply_full_path, format_fullpath_warning, render_full_path
from pyqmd_mlx.cli._lineparse import parse_line_range
from pyqmd_mlx.cli._multiget import resolve_multi_get
from pyqmd_mlx.cli._output_documents import format_documents
from pyqmd_mlx.cli._output_search import add_line_numbers
from pyqmd_mlx.cli._theme import cyan, dim
from pyqmd_mlx.cli._types import DocumentEntry, OutputFormat

app = typer.Typer(help="Fetch and list documents.")


@app.command("get")
def get(
    identifier: str = typer.Argument(
        ..., help="Path, qmd:// URI, or docid (optionally :line or :line:count)."
    ),
    from_line: int = typer.Option(None, "--from", help="Start line (1-indexed)."),
    max_lines: int = typer.Option(None, "-l", help="Maximum number of lines."),
    line_numbers: bool = typer.Option(True, "--line-numbers/--no-line-numbers"),
    full_path: bool = typer.Option(False, "--full-path"),
) -> None:
    """Get a single document by path, qmd:// URI, or docid."""
    store = get_store()
    bare_identifier, from_line, max_lines = parse_line_range(identifier, from_line, max_lines)

    def _run():
        doc = store.find_document_by_identifier(bare_identifier)
        if doc is None:
            raise ValueError(f"Document not found: {identifier}")
        return doc

    doc = run_or_exit(_run)

    resolved = store.resolve_full_path(doc["collection"], doc["path"]) if full_path else None
    if resolved is not None:
        header = render_full_path(resolved)
    else:
        header = (
            f"{dim('qmd://')}{cyan(doc['collection'] + '/' + doc['path'])}"
            f"{dim('  #' + doc['hash'][:6])}"
        )
    typer.echo(header)
    if full_path and resolved is None:
        typer.echo(format_fullpath_warning(1), err=True)
    context = store.get_context_for_path(doc["collection"], doc["path"])
    if context:
        typer.echo(f"{dim('Folder Context:')} {context}")
    typer.echo("---\n")

    body = doc["doc"]
    start_line = from_line or 1
    if from_line is not None or max_lines is not None:
        lines = body.split("\n")
        start = start_line - 1
        end = start + max_lines if max_lines is not None else len(lines)
        body = "\n".join(lines[start:end])

    if line_numbers:
        body = add_line_numbers(body, start_line)

    typer.echo(body)


@app.command("multi-get")
def multi_get(
    pattern: str = typer.Argument(..., help="Comma-separated docids or paths."),
    max_lines: int = typer.Option(None, "-l"),
    max_bytes: int = typer.Option(10 * 1024, "--max-bytes"),
    line_numbers: bool = typer.Option(True, "--line-numbers/--no-line-numbers"),
    full_path: bool = typer.Option(False, "--full-path"),
    format: OutputFormat = typer.Option(OutputFormat.CLI, "--format"),
) -> None:
    """Get multiple documents by glob or comma-separated docid/path list."""
    store = get_store()
    resolved = resolve_multi_get(store, pattern, max_bytes)

    if full_path:
        resolvable = [r for r in resolved if r.not_found is None]
        unresolved = apply_full_path(resolvable, store)
        if unresolved:
            typer.echo(format_fullpath_warning(unresolved), err=True)

    entries: list[DocumentEntry] = []
    for r in resolved:
        if r.not_found:
            typer.echo(f"Not found: {r.not_found}", err=True)
            continue
        if r.skipped:
            entries.append(
                DocumentEntry(
                    display_path=r.display_path,
                    title=r.title,
                    body="",
                    skipped=True,
                    skip_reason=r.skip_reason,
                    docid=r.docid,
                )
            )
            continue

        body = r.body
        if max_lines is not None:
            lines = body.split("\n")
            truncated = len(lines) > max_lines
            body = "\n".join(lines[:max_lines])
            if truncated:
                body += f"\n\n[... truncated {len(lines) - max_lines} more lines]"
        if line_numbers:
            body = add_line_numbers(body)

        entries.append(
            DocumentEntry(display_path=r.display_path, title=r.title, body=body, docid=r.docid)
        )

    if not entries:
        typer.echo("No documents found.", err=True)
        raise typer.Exit(1)

    typer.echo(format_documents(entries, format.value))


@app.command("ls")
def ls(
    arg: str = typer.Argument(None, help="Optional 'collection' or 'collection/path-prefix'."),
) -> None:
    """List collections, or files within a collection."""
    store = get_store()

    if not arg:
        collections = store.list_collections()
        if not collections:
            typer.echo("No collections.")
            return
        stats = store.get_collection_document_stats()
        for c in collections:
            count = stats.get(c["name"], {"count": 0})["count"]
            typer.echo(f"{dim('qmd://')}{cyan(c['name'] + '/')}  {dim(f'({count} files)')}")
        return

    if "/" in arg:
        collection, prefix = arg.split("/", 1)
    else:
        collection, prefix = arg, None

    def _run():
        if store.get_collection(collection) is None:
            raise ValueError(f"No such collection: {collection}")
        docs = store.get_active_documents_with_size(collection)
        if prefix:
            normalized_prefix = prefix.rstrip("/") + "/"
            docs = [
                d for d in docs if d["path"] == prefix or d["path"].startswith(normalized_prefix)
            ]
        return docs

    docs = run_or_exit(_run)
    if not docs:
        typer.echo("No documents.")
        return

    sizes = [format_bytes(d["size"]) for d in docs]
    max_size = max(len(s) for s in sizes)
    for doc, size_str in zip(docs, sizes, strict=True):
        time_str = format_ls_time(doc["modified_at"])
        prefix_str = dim(f"qmd://{collection}/")
        typer.echo(f"{size_str.rjust(max_size)}  {time_str}  {prefix_str}{cyan(doc['path'])}")
