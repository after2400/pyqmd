"""`pyqmd update` command."""

import subprocess
from pathlib import Path

import typer

from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli._index_report import echo_embed_hint, echo_index_summary
from pyqmd_mlx.cli._theme import bold, cyan, dim, green, yellow
from pyqmd_mlx.store._indexing import scan_and_register_collection, split_glob_mask

app = typer.Typer(help="Re-scan existing collections for new/changed/removed files.")


@app.callback()
def _update_callback() -> None:
    """Re-scan existing collections for new/changed/removed files.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["update", ...])`-style invocation (the
    literal "update" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so this callback is not expected to be
    removed later.
    """


def _pull_collection(collection: dict) -> bool:
    """Run `git pull --ff-only` in a collection's directory, if it's a git
    working copy. Returns True if this collection should still be scanned
    this run (not a git repo, or a successful pull), False if a real git
    repo's pull failed and this collection should be skipped for this run
    rather than aborting the whole `update`.

    Node's own --pull flag is dead code: declared in the CLI arg parser
    and documented in --help, but never actually read anywhere in
    updateCollections() -- there is no existing behavior to match here.
    --ff-only (never produce a surprise merge commit in a collection the
    user didn't ask update to modify beyond its content) and skipping
    quietly when a collection isn't a git repo at all (most collections
    won't be) are pyqmd's own choices, not a port.

    Skip-this-collection-only (rather than abort the whole run, which is
    what a failing update_command hook does) is deliberate: unlike a
    hook, which is user-authored and opt-in, a git pull failure (a
    diverged local branch, no configured remote, a transient network
    error) is a common, often-transient, per-collection condition that
    shouldn't block updating every other collection in the same run. A
    missing git binary is neither: it warns and proceeds to the scan --
    a re-scan of the directory's current state is still the truthful
    thing to do.

    The .git-is-a-directory check is deliberately simple: git worktrees
    (.git is a *file*, not a directory) and collections pointing at a
    subdirectory of a repo are treated as non-repos and their pull is
    skipped quietly.
    """
    path = collection["path"]
    if not (Path(path) / ".git").is_dir():
        return True
    typer.echo("    Pulling latest changes...")
    try:
        proc = subprocess.run(
            ["git", "pull", "--ff-only"], cwd=path, capture_output=True, text=True
        )
    except OSError as err:
        typer.echo(f"    git unavailable, continuing without pull: {err}", err=True)
        return True
    if proc.stdout.strip():
        typer.echo(proc.stdout.strip())
    if proc.returncode != 0:
        typer.echo(
            f"    git pull failed, skipping this collection: {proc.stderr.strip()}", err=True
        )
        return False
    return True


def _run_update_hook(collection: dict) -> None:
    """Run a collection's update_command, if set, before re-scanning it.
    Exits the process on failure -- matching Node's behavior of stopping
    the whole `update` run rather than skipping to the next collection.

    If the collection's path no longer exists on disk, subprocess raises
    (FileNotFoundError/NotADirectoryError); catch it and raise ValueError
    so `run_or_exit` prints a clean
    `Error: Collection path no longer exists: <path>` and exits 1, the
    whole command stopping -- same as a failing hook.

    Extension point: when a future sub-project adds project-local/checked-
    in config (`init`), collections may start arriving from a file that
    travels with `git clone` rather than a command the user typed
    themselves -- at that point this call needs a trust gate in front of
    it (see the design spec's "Security note"). Until that lands, every
    update_command in pyqmd's index only ever got there via an explicit
    `pyqmd collection add --update-cmd` / `update-cmd` the user ran
    themselves, so no gate is needed yet.
    """
    cmd = collection["update_command"]
    if not cmd:
        return
    typer.echo(dim(f"    Running update command: {cmd}"))
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd], cwd=collection["path"], capture_output=True, text=True
        )
    except OSError as err:
        raise ValueError(f"Collection path no longer exists: {collection['path']} ({err})") from err
    for stream_text in (proc.stdout, proc.stderr):
        if stream_text.strip():
            typer.echo("\n".join(f"    {line}" for line in stream_text.strip().split("\n")))
    if proc.returncode != 0:
        typer.echo(yellow(f"✗ Update command failed with exit code {proc.returncode}"))
        raise typer.Exit(proc.returncode)


@app.command("update")
def update(
    collection: list[str] = typer.Option(None, "-c", "--collection"),
    pull: bool = typer.Option(False, "--pull", help="git pull each collection's directory first"),
) -> None:
    """Re-index every (or a named subset of) collection's directory."""
    store = get_store()

    def _resolve_targets() -> list[dict]:
        all_collections = {c["name"]: c for c in store.list_collections()}
        if not collection:
            return list(all_collections.values())
        targets = []
        for name in collection:
            if name not in all_collections:
                raise ValueError(f"No such collection: {name}")
            targets.append(all_collections[name])
        return targets

    targets = run_or_exit(_resolve_targets)

    if not targets:
        typer.echo(
            dim("No collections found. Run 'pyqmd collection add .' to index markdown files.")
        )
        return

    typer.echo(bold(f"Updating {len(targets)} collection(s)...") + "\n")
    skipped = 0

    for i, coll in enumerate(targets, start=1):
        position = cyan(f"[{i}/{len(targets)}]")
        pattern = coll["pattern"]
        typer.echo(f"{position} {bold(coll['name'])} {dim(f'({pattern})')}")

        if pull and not _pull_collection(coll):
            skipped += 1
            continue

        def _run(coll=coll):
            _run_update_hook(coll)
            typer.echo(f"Collection: {coll['path']} ({coll['pattern']})")
            ignore_patterns = (
                split_glob_mask(coll["ignore_patterns"]) if coll["ignore_patterns"] else None
            )
            return scan_and_register_collection(
                store, coll["path"], coll["pattern"], coll["name"], ignore_patterns=ignore_patterns
            )

        result = run_or_exit(_run)
        echo_index_summary(result)
        typer.echo("")

    if skipped:
        typer.echo(
            green(
                f"✓ Updated {len(targets) - skipped} of {len(targets)} "
                f"collection(s) ({skipped} skipped)."
            )
        )
    else:
        typer.echo(green("✓ All collections updated."))

    echo_embed_hint(store)
