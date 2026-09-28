"""`pyqmd context add/list/remove` commands."""

import os

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import CliError, run_or_exit
from pyqmd_mlx.cli._theme import bold, cyan, dim, green, yellow

app = typer.Typer(help="Manage per-path context descriptions attached to get/status/search output.")


def _resolve_collection_and_prefix(store, path: str, *, status_hint: bool) -> tuple[str, str]:
    """Resolve a context command's <path> argument to (collection,
    path_prefix). Accepts 'qmd://collection/path' (path_prefix "" for the
    collection root) or a filesystem path -- '.'/'./' -> cwd, '~/' expands,
    other relative paths resolve against cwd -- matched against an indexed
    collection's directory via Store.detect_collection_for_path. Raises
    CliError with Node's wording on either failure mode; `status_hint`
    adds context add's second "Run 'pyqmd status'..." line, which Node's
    context remove omits."""
    if path.startswith("qmd://"):
        without_scheme = path[len("qmd://") :]
        if "/" in without_scheme:
            collection, prefix = without_scheme.split("/", 1)
        else:
            collection, prefix = without_scheme, ""
        prefix = prefix.rstrip("/")
        if store.get_collection(collection) is None:
            raise CliError(yellow(f"Collection not found: {collection}"))
        return collection, prefix

    if path in (".", "./"):
        fs_path = os.getcwd()
    elif path.startswith("~/"):
        fs_path = os.path.expanduser(path)
    elif os.path.isabs(path):
        fs_path = path
    else:
        fs_path = os.path.abspath(os.path.join(os.getcwd(), path))

    detected = store.detect_collection_for_path(fs_path)
    if detected is None:
        lines = [yellow(f"Path is not in any indexed collection: {fs_path}")]
        if status_hint:
            lines.append(dim("Run 'pyqmd status' to see indexed collections"))
        raise CliError(*lines)
    return detected


@app.command("add")
def add(
    path: str = typer.Argument(
        ...,
        help=(
            "qmd://collection/path, or a filesystem path (., ~/, relative, "
            "or absolute) inside an indexed collection."
        ),
    ),
    text: list[str] = typer.Argument(..., help="Context description."),
) -> None:
    """Add or update the context for a path prefix."""
    store = get_store()
    context_text = " ".join(text).strip()

    def _run():
        collection, prefix = _resolve_collection_and_prefix(store, path, status_hint=True)
        store.add_context(collection, prefix, context_text)
        return collection, prefix

    collection, prefix = run_or_exit(_run)
    if prefix:
        display = f"qmd://{collection}/{prefix}"
    elif path.startswith("qmd://"):
        display = f"qmd://{collection}/ (collection root)"
    else:
        display = f"qmd://{collection}/"
    typer.echo(f"{green('✓')} Added context for: {display}")
    typer.echo(dim(f"Context: {context_text}"))


@app.command("list")
def list_contexts() -> None:
    """List all configured contexts, grouped by collection."""
    store = get_store()
    contexts = store.list_all_contexts()
    if not contexts:
        typer.echo(dim("No contexts configured. Use 'pyqmd context add' to add one."))
        return

    typer.echo(f"\n{bold('Configured Contexts')}\n")
    last_collection = None
    for ctx in contexts:
        if ctx["collection"] != last_collection:
            typer.echo(cyan(ctx["collection"]))
            last_collection = ctx["collection"]
        typer.echo(f"  {ctx['path']}" if ctx["path"] else "  / (root)")
        typer.echo(f"    {dim(ctx['context'])}")


@app.command("remove")
def remove(path: str = typer.Argument(...)) -> None:
    """Remove the context for a path prefix."""
    store = get_store()

    def _run():
        collection, prefix = _resolve_collection_and_prefix(store, path, status_hint=False)
        shown = path if path.startswith("qmd://") else f"qmd://{collection}/{prefix}"
        if not store.remove_context(collection, prefix):
            raise CliError(yellow(f"No context found for: {shown}"))
        return shown

    shown = run_or_exit(_run)
    typer.echo(f"{green('✓')} Removed context for: {shown}")
