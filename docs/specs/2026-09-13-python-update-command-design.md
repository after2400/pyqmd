# pyqmd `update` command — design

**Status:** Implemented (2026-09-14)

**Sub-project #8** of the qmd Python/MLX rewrite. Roadmap:
`2026-09-10-python-mlx-rewrite-roadmap.md`, item #8.

## Problem

Today, the only way to pick up new/changed/removed files in an existing
pyqmd collection is destructive: `collection remove` (deletes all
embeddings and metadata) + `collection add` + a full `embed` re-run.
`collection add` on an existing name raises a raw `sqlite3.IntegrityError`
rather than failing cleanly. There is no way to re-scan a collection in
place, even though the underlying logic
(`scan_and_register_collection`'s indexed/updated/unchanged/removed
tracking) already supports it — it's simply unreachable from the CLI.

This has been a recurring parked finding since sub-project #3's final
review, referenced again in #4 and #7 without ever being scheduled. This
spec gives it a real numbered slot and closes it out.

## Reference implementation

Node's `qmd update` (`src/cli/qmd.ts`'s `updateCollections()`) and
`reindexCollection` (`src/store.ts`). pyqmd's `update` intentionally does
**not** port Node's YAML-config/trust-gating machinery — see "Security
note" below for why that's safe to skip today.

## Decisions made during brainstorming (2026-09-13)

1. **`update` scoping**: matches Node's "update all collections by
   default" behavior, plus a `-c/--collection` option — repeatable, like
   every other pyqmd command that supports it (`search`/`vsearch`/`query`/
   `embed`) — to scope to specific collections instead of all of them.
2. **`update_command` hook**: included in this sub-project. `Store`
   already has an `update_command` column and constructor parameter on
   `add_collection` that nothing sets via the CLI — this finishes exposing
   it rather than leaving it as dead plumbing.
3. **Orphaned-content cleanup**: included. Node's `reindexCollection`
   cleans up orphaned `content`/`content_vectors`/`vectors_vec` rows
   inline after every scan; pyqmd's `scan_and_register_collection`
   currently doesn't. Left as-is, every `update` on a changing collection
   would leak storage that Node doesn't.
4. **Auto-embed**: explicitly decoupled, matching Node. `update` re-scans
   only and reports a "N hashes need embedding, run `pyqmd embed`" hint;
   it never runs embedding itself.
5. **The parked idempotent-`collection add` question** (sub-project #3):
   resolved by checking Node's own behavior — Node's `collection add` on a
   duplicate name, or a duplicate `(path, pattern)` pair under a different
   name, is **also** a clean error pointing the user at `qmd update` or
   `qmd collection remove`, not an idempotent re-scan. pyqmd should match
   this exactly: `add` stays strict. The only real gap is presentation —
   pyqmd's raw `IntegrityError` needs to become the same kind of clean,
   helpful error Node prints.
6. **`cleanup` command** (VACUUM/FTS-compaction/purge-deactivated-docs,
   Node's separate `qmd cleanup [--dry-run]`): out of scope for this
   sub-project. Tracked as new roadmap item #9 instead of being folded in
   — re-scanning and vacuuming are different operations with different
   edge cases, and this sub-project's scope is already full.
7. **`--pull`** (added after spec review, 2026-09-13): included. Node's own
   `--pull` turned out to be **dead code** — declared in the CLI arg parser
   (`pull: { type: "boolean" }`) and mentioned in `--help`, but
   `updateCollections()` never reads the value anywhere; there is no real
   git-pull behavior in Node to port. Cheap enough to build for real once
   `update`'s hook-running plumbing already exists (same subprocess/print
   shape as `_run_update_hook`), so it's folded in now rather than left as
   a speculative future add-on. See "Git pull (--pull)" below for the
   (pyqmd's own, not ported) behavior this implies.
8. **Output framing** (added after spec review, 2026-09-13): `update`
   prints a `Updating N collection(s)...` header, a `[i/N] name (pattern)`
   line per collection, and an honest trailer — matching Node's bookends
   (`src/cli/qmd.ts:920-937`, `:1013`) so multi-collection runs are
   unambiguous. The trailer is `✓ All collections updated.` when every
   collection was scanned; when some were skipped (failed `--pull`) it is
   `✓ Updated N of M collection(s) (K skipped).` — "all updated" would be
   a lie otherwise (Node never hits this: its `--pull` is dead code). An
   empty index prints `No collections found. Run 'pyqmd collection add
<path> --name <name>' to index markdown files.` and exits 0 (Node
   `:920-923`).
9. **Orphan-line presentation**: the counts line stays byte-identical to
   `collection add`'s existing format; the new information is a _separate_
   line, `Cleaned up N orphaned content hash(es)`, printed **only when
   N > 0**, in both `update` and `collection add` — mirroring Node exactly
   (`src/cli/qmd.ts:1001-1003`). A fresh `collection add` never orphans
   anything (a new name cannot deactivate documents), so `add`'s output is
   unchanged in practice.
10. **`update-cmd` argument shape**: the command is a _variadic_ argument,
    joined with spaces — matching Node's `cli.args.slice(2).join(' ')`
    (`src/cli/qmd.ts:4589`), so an unquoted `pyqmd collection update-cmd
   brain git pull` works. An empty or whitespace-only command normalizes
    to `None` (clears the hook), matching Node's `|| null`.
11. **Subprocess failure modes**:

- `git` binary missing under `--pull` → warn
  (`git unavailable, continuing without pull`) and proceed to the scan;
  a missing binary must not skip or abort collections.
- Collection path deleted from disk + a hook set → the hook's
  `subprocess.run` raises; catch it and surface a clean
  `Error: Collection path no longer exists: <path>` with exit 1, the
  whole command stopping — same as a failing hook, since a vanished
  path is a stronger failure.
- The `.git`-is-a-directory check stays deliberately simple: git
  worktrees (`.git` is a _file_, not a directory) and collections
  pointing at a subdirectory of a repo are treated as non-repos and
  their pull is skipped quietly. Documented and pinned by test, not
  "fixed" (YAGNI — most collections are plain directories).

12. **Extraction guard**: the orphan-cleanup block inside
    `remove_collection` runs only when the collection had document rows
    (`if doc_ids:`, `store.py:208`). The extraction keeps that guard, so
    `remove_collection` behavior is unchanged;
    `scan_and_register_collection` calls the method unconditionally (a
    re-scan is exactly when orphans arise).
13. **Deleted collection directory**: `update` on a path that no longer
    exists re-scans nothing and deactivates every active document in that
    collection (all counted `Removed`) — Node parity (an empty glob means
    "all documents gone"). Pinned by test; deliberately not turned into an
    early error.
14. **`collection show` prints the hook**: when `update_command` is set,
    `show` gains an `Update: <cmd>` line (Node `src/cli/qmd.ts:4651-4653`)
    — the only way to inspect a hook without running `update` (which may
    have side effects).
15. **`--exclude` persistence**: `collection add` already accepts repeatable
    `--exclude` globs and passes them to its scan, but never persists them —
    the `ignore_patterns` column exists for exactly this and is never
    written. A `update` re-scan that didn't receive the same globs would
    index the files `add` deliberately skipped. So: `add` stores
    `",".join(exclude)` (same comma-join convention as the existing
    `pattern` column; a pattern containing a top-level comma is ambiguous
    exactly the way `--mask` already is) and keeps passing the raw list to
    its own scan (no behavior change); `update` reads the column back
    through `split_glob_mask` (so braces/brackets survive, same as masks)
    and passes it as the re-scan's `ignore_patterns`.

## Design

### New command: `pyqmd update`

New module `qmd/cli/commands/update.py`, mounted as a top-level command in
`qmd/cli/app.py`, mirroring the existing single-command module pattern
used by `embed.py` and `status.py` (a no-op `@app.callback()` plus one
`@app.command("update")`).

```
pyqmd update [-c/--collection NAME]... [--pull]
```

```python
collection: list[str] = typer.Option(None, "-c", "--collection")
pull: bool = typer.Option(False, "--pull", help="git pull each collection's directory first")
```

(`collection` matches `search.py`'s existing `-c/--collection` signature
exactly — `typer.Option(None, ...)`, not `typer.Option([], ...)`, for
consistency with the rest of the CLI.)

- No `-c`: update every collection, in `list_collections()`'s order.
- One or more `-c NAME`: update only the named collection(s); error
  cleanly (via `run_or_exit`) if any named collection doesn't exist —
  checked up front, before any collection is touched, so a typo in the
  second of three names doesn't leave the first partially updated.
- An empty index (no collections at all) prints `No collections found. Run
'pyqmd collection add <path> --name <name>' to index markdown files.`
  and exits 0.

Framing (decision 8): before the loop, print
`Updating N collection(s)...` (N = number of targeted collections). Per
collection, print the header line `[i/N] name (pattern)` first. After the
loop, print the honest trailer: `✓ All collections updated.` if no
collection was skipped, else `✓ Updated N of M collection(s) (K skipped).`

Per-collection flow (one iteration of a loop over the resolved
collections, after its header line):

1. If `--pull` is set, run `_pull_collection(collection)` (see "Git pull
   (--pull)" below). If it returns `False` (a real git repo whose pull
   failed), skip the rest of this collection's flow — no hook, no scan —
   and move to the next collection; this one is reported as skipped, not
   counted in the indexed/updated/unchanged/removed totals.
2. If the collection's `update_command` is set, run it via
   `_run_update_hook(collection)` (see "Hook execution" below). A nonzero
   exit stops the whole command immediately (`raise typer.Exit(code)`) —
   matching Node's `process.exit(exitCode)` — before this collection (or
   any later one) is scanned. A vanished collection path surfaces as a
   clean `Error: Collection path no longer exists: <path>` exit 1 (the
   hook and the scan are wrapped in `run_or_exit` together).
3. Call `scan_and_register_collection(...)` (now extended per "Orphan
   cleanup" below), passing the collection's persisted exclude globs back
   (decision 15).
4. Print the counts line — byte-identical to `collection add`'s existing
   format, `Indexed: N  Updated: N  Unchanged: N  Removed: N` — then, only
   when `orphaned_cleaned > 0`, the separate line
   `Cleaned up N orphaned content hash(es)` (decision 9).
5. Print skipped files the same way `collection add` already does
   (`Skipped N file(s):` to stderr, one `path: reason` line per file).

After every targeted collection has been processed (and the trailer): call
`store.get_status_counts()` (already exists, already used by `status`) and,
if `pending_embed > 0`, print
`Run 'pyqmd embed' to update embeddings (N hashes need embedding)`.
`pending_embed` is an index-wide count (all collections, not just the ones
updated this run) — matching Node's index-wide `getHashesNeedingEmbedding`;
a bare `pyqmd embed` covers everything it refers to.

### Git pull (`--pull`)

```python
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
    missing git binary is neither: it warns and proceeds to the scan
    (decision 11) -- a re-scan of the directory's current state is still
    the truthful thing to do.

    The .git-is-a-directory check is deliberately simple (decision 11):
    git worktrees (.git is a *file*, not a directory) and collections
    pointing at a subdirectory of a repo are treated as non-repos and
    their pull is skipped quietly.
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
```

### Hook execution

```python
def _run_update_hook(collection: dict) -> None:
    """Run a collection's update_command, if set, before re-scanning it.
    Exits the process on failure -- matching Node's behavior of stopping
    the whole `update` run rather than skipping to the next collection.

    If the collection's path no longer exists on disk, subprocess raises
    (FileNotFoundError/NotADirectoryError); catch it and raise ValueError
    so `run_or_exit` prints a clean
    `Error: Collection path no longer exists: <path>` and exits 1, the
    whole command stopping -- same as a failing hook (decision 11).

    Extension point: when a future sub-project adds project-local/checked-
    in config (`init`), collections may start arriving from a file that
    travels with `git clone` rather than a command the user typed
    themselves -- at that point this call needs a trust gate in front of
    it (see this file's "Security note"), the same way Node's
    resolveLocalConfigTrust() gates its own equivalent call. Until that
    lands, every update_command in pyqmd's index only ever got there via
    an explicit `pyqmd collection add --update-cmd` / `update-cmd` the
    user ran themselves, so no gate is needed yet.
    """
    cmd = collection["update_command"]
    if not cmd:
        return
    typer.echo(f"    Running update command: {cmd}")
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd], cwd=collection["path"], capture_output=True, text=True
        )
    except OSError as err:
        raise ValueError(
            f"Collection path no longer exists: {collection['path']} ({err})"
        ) from err
    if proc.stdout.strip():
        typer.echo(proc.stdout.strip())
    if proc.stderr.strip():
        typer.echo(proc.stderr.strip())
    if proc.returncode != 0:
        typer.echo(f"Update command failed with exit code {proc.returncode}", err=True)
        raise typer.Exit(proc.returncode)
```

Isolating this in its own function is the one piece of "scaffolding" this
design deliberately includes — a normal decomposition seam, not a guess at
`init`'s eventual design. See the "Scaffolding" discussion below.

### Orphan cleanup

Extract `remove_collection`'s existing orphan-detection/deletion logic
(the `content`/`content_vectors`/`vectors_vec` cleanup for hashes no
longer referenced by any active document — `store.py` around lines
220-247) into a new, reusable method:

```python
def cleanup_orphaned_content(self) -> int:
    """Delete content/content_vectors/vectors_vec rows for hashes no
    longer referenced by any active document. Global (not collection-
    scoped) -- content is addressed by hash, shared across collections.
    Returns the number of orphaned hashes cleaned. Extracted from
    remove_collection so update's re-scan path can reuse it (matching
    Node's reindexCollection, which calls the same cleanup inline after
    every scan -- see store.ts:1762)."""
```

`remove_collection` calls this new method in place of its inline logic,
**inside its existing `if doc_ids:` block** — a collection removed with
zero document rows orphans nothing, so the guard travels with the code and
`remove_collection`'s behavior is unchanged (decision 12; the method
commits its own transaction, since in the scan path it is the last
operation and its deletes must not be left uncommitted). `ReindexResult`
gains an `orphaned_cleaned: int = 0` field. `scan_and_register_collection`
calls `store.cleanup_orphaned_content()` once, at the very end (after the
removed-document deactivation loop), storing the count in
`result.orphaned_cleaned` — matching where Node calls
`cleanupOrphanedContent(db)` inside `reindexCollection` itself, not in the
outer CLI loop. `collection add` (which already calls
`scan_and_register_collection`) thus picks up the cleanup for free; the
only change to `add` is the conditional orphan line (decision 9), which a
fresh add never prints — a new name cannot deactivate documents — so its
summary output stays byte-identical in practice. The stale `_indexing.py`
module docstring (which still claims orphan cleanup was dropped from the
port) is corrected in the same change.

### Clean duplicate-collection error

In `collection.py`'s `add` command, before calling `store.add_collection`:

```python
def _check_no_duplicate(name: str, path: str, mask: str) -> None:
    existing = store.get_collection(name)
    if existing is not None:
        raise ValueError(
            f"Collection '{name}' already exists. Use 'pyqmd update' to "
            f"re-index it, or remove it first with 'pyqmd collection remove {name}'."
        )
    for c in store.list_collections():
        if c["path"] == path and c["pattern"] == mask:
            raise ValueError(
                f"A collection already exists for this path and pattern: "
                f"'{c['name']}'. Use 'pyqmd update' to re-index it, or "
                f"remove it first with 'pyqmd collection remove {c['name']}'."
            )
```

This raises a plain `ValueError`, already caught by `run_or_exit`'s
existing `EXPECTED_EXCEPTIONS` tuple — no changes needed to `_errors.py`.
The message text mirrors Node's (`src/cli/qmd.ts:1839`, `:1849-1852`)
closely enough to be recognizable, not verbatim.

One inherited nuance, documented and pinned by test rather than "fixed":
the `(path, pattern)` check compares the raw stored string, exactly like
Node (`src/cli/qmd.ts:1846`) — `./notes` and `notes` are distinct values,
so two spellings of the same directory are _not_ deduplicated.

### `update_command` hook surface

- `collection add` gains `--update-cmd <cmd>` (`typer.Option(None,
"--update-cmd")`), passed straight through to `add_collection`'s
  existing `update_command` parameter. `add` also persists its `--exclude`
  globs into the `ignore_patterns` column (decision 15).
- New subcommand `collection update-cmd <name> [command...]`:
  - `command` is variadic (decision 10): `typer.Argument(None)` on a
    `list[str]`, joined with `" "`, `.strip()`ed, then
    `"..." or None` — so an unquoted multi-word command works and an
    empty/whitespace-only one normalizes to `None` (clears the hook),
    matching Node's `|| null`.
  - No `command` given → clears the hook (`update_command = NULL`).
  - Prints `✓ Set update command for '<name>': <cmd>` or
    `✓ Cleared update command for '<name>'`, matching Node's text
    (`src/cli/qmd.ts:4602-4609`).
  - Backed by a new `Store` method (pure setter; the existence check
    lives in the CLI, matching the `rename`/`remove` pattern):
    ```python
    def set_collection_update_command(self, name: str, command: str | None) -> None:
        self.conn.execute(
            "UPDATE collections SET update_command = ? WHERE name = ?", (command, name)
        )
        self.conn.commit()
    ```
  - Errors cleanly (`ValueError`, via `run_or_exit`) if `name` doesn't
    exist, checked up front like `rename`/`remove`.
- `collection show` gains an `Update: <cmd>` line when `update_command`
  is set (decision 14; Node `src/cli/qmd.ts:4651-4653`).

### Excluded patterns (`--exclude`)

`collection add` already accepts repeatable `--exclude` globs and passes
them to its scan, but never persists them — the `ignore_patterns` column
exists for exactly this and is never written (the second dead-plumbing
column, like `update_command`). A `update` re-scan that didn't receive the
same globs would index the files `add` deliberately skipped. So (decision
15): `add` stores `",".join(exclude)` in the column — the same comma-join
convention as the existing `pattern` column, split back with
`split_glob_mask` at use time so braces/brackets survive — while keeping
the raw list for its own scan (no behavior change). `update` reads the
column back through `split_glob_mask` and passes it as the re-scan's
`ignore_patterns`.

### Security note (not a gap to fix here — a documented boundary)

Node's `update` gates hook execution (and out-of-project paths, and custom
model URIs) behind a "trust" system, because Node collections can be
defined in a project-local `.qmd/index.yml` that travels with `git clone`
— so a hook can arrive on your machine from someone else's checked-in
config, not from a command you typed. pyqmd has no project-local/checked-
in config feature at all (no `init`). Every collection, and every
`--update-cmd` value, only ever enters pyqmd's index via a CLI command the
user ran themselves — there is no path for a third party's config to
inject a hook without the user's own explicit command doing it. So no
trust-gating is needed today.

**This changes the moment pyqmd gets a project-local/checked-in config**
(most likely alongside a future `init` command). At that point, hook
execution needs the same kind of trust gate Node has, inserted at the
`_run_update_hook` call site (see its docstring above). This is a
documented tradeoff, not an oversight — recorded here so a future `init`
sub-project's brainstorm starts from this note instead of rediscovering
the threat model from scratch.

### Scaffolding (what's deliberately _not_ built now)

Discussed explicitly during brainstorming: no trust-system code, storage,
or command surface is being added in this sub-project. Building any of it
now would mean guessing at a future `init` sub-project's design (where a
checked-in config would live, what fields besides hooks it would cover,
what a trust-record/approval store would look like) before that design
exists — the same premature-abstraction risk this project has avoided
elsewhere (e.g. #7 left "a future sub-project builds `update`" as a
documented gap rather than guessing at it). The only concession to the
future is structural: `_run_update_hook` is its own function (normal
decomposition, not scaffolding) with a docstring pointing at this file, so
the insertion point for a future trust gate is obvious rather than
requiring Node's source to be re-read from scratch.

## Out of scope

- `cleanup` command (VACUUM, FTS compaction, purge-deactivated-docs) —
  tracked separately as roadmap item #9.
- An `--embed` flag to auto-run embedding after `update` — decided against;
  decoupled matches Node and today's `collection add` + `embed` workflow.
- Any trust/config-file system — see "Scaffolding" above.

## Testing approach

Standard pyqmd CLI-testing pattern (matching `tests/cli/test_collection*.py`
and `test_embed_command.py`): `CliRunner` against an in-memory `Store`,
monkeypatching `get_store`. Cases to cover in the implementation plan:

- `update` with no collections: `No collections found. ...` message,
  exit 0.
- `update` with no `-c`: re-scans all collections; `Updating N
collection(s)...` header, per-collection `[i/N] name (pattern)` headers,
  per-collection counts, trailer; pending-embed hint when nonzero and
  omitted at zero (pre-embed every hash in the test to hit zero).
- `update -c <name>` (single and repeated): scopes correctly (only the
  named collection's header appears); errors cleanly on an unknown name,
  before touching any real collection (assert a modified file in a
  _named, valid_ collection was NOT re-scanned when a _later_ name is
  misspelled).
- A collection with `update_command` set: hook runs before the scan (prove
  by order — the hook generates a file the scan then indexes), its
  stdout is printed, a nonzero exit stops the whole command (that exit
  code is the command's, later collections untouched).
- A collection with no `update_command`: no hook output, scan runs
  directly.
- Collection path deleted from disk + hook set: clean `Error: Collection
path no longer exists: ...` exit 1, no traceback.
- Collection path deleted from disk, no hook: re-scan deactivates every
  active document in that collection (all `Removed`), exit 0 (decision 13).
- `update --pull` on a collection whose directory has a `.git`: a real
  `git pull --ff-only` runs before the hook/scan (use a real local git
  repo fixture with a second repo as its remote, not a mock, so a genuine
  fast-forward is exercised — a new commit made on the remote side must
  appear as `indexed` after the update); git-fixture tests skip when `git`
  is not on PATH.
- `update --pull` on a collection with no `.git` directory: pull is
  skipped quietly (no `Pulling latest changes...` line), hook/scan proceed.
- `update --pull` where the pull fails (a diverged local branch): that
  collection is skipped (no hook run, no scan — a modified file there
  stays stale in the index), a warning is printed, the command continues
  to the next collection, and the trailer reports
  `✓ Updated N of M collection(s) (K skipped).` (decision 8).
- `update --pull` where `git` is missing from PATH (gutted `PATH` env):
  warning `git unavailable, continuing without pull`, scan proceeds,
  exit 0 (decision 11).
- `.git` present as a _file_ (worktree-style): treated as non-repo, pull
  skipped quietly (decision 11).
- `update` without `--pull`: no git commands run at all, even for
  collections that are git repos (assert via a recording
  `subprocess.run` substitute in the update module).
- `update` on a collection with persisted `ignore_patterns` (from
  `add --exclude`): a file matching the exclude glob is not indexed by the
  re-scan (decision 15); end-to-end via the `collection add` CLI so the
  persistence round-trip is what's tested.
- Orphan line: a document's content hash changes across an update → the
  separate `Cleaned up N orphaned content hash(es)` line appears; a
  no-change re-scan → the line is absent (decision 9). Store-level: direct
  unit tests of `cleanup_orphaned_content` (orphaned hashes deleted from
  `content`/`content_vectors`/`vectors_vec`, shared content preserved,
  return count) in a new `tests/store/test_orphan_cleanup.py`, plus a
  scan-level test (hash change across two scans: old hash's rows gone,
  `orphaned_cleaned == 1`) in `tests/store/test_indexing_scan.py`. The
  existing `remove_collection` orphan test in
  `tests/store/test_cli_capabilities.py` stays untouched and green — the
  extraction must not change it.
- `collection add` on an existing name: clean `ValueError` message
  mentioning `update` and `remove`, not a raw `IntegrityError` (the
  pre-existing raw-error test keeps passing: exit code 1 either way).
- `collection add` on an existing `(path, pattern)` under a different
  name: same clean-error treatment; and the raw-string nuance (two
  spellings of the same directory are _not_ deduplicated) pinned by test.
- `collection add --update-cmd <cmd>`: stored and retrievable via
  `get_collection`; `collection show` prints the `Update:` line (decision 14) and omits it when unset.
- `collection update-cmd <name> git pull` (unquoted multi-word, variadic)
  and `collection update-cmd <name>` (no command): set the joined command /
  clear respectively (an empty-string command also clears); errors cleanly
  on an unknown collection name. Store-level
  `set_collection_update_command` tests (set/update/clear, ValueError on
  unknown name) in `tests/store/test_collections.py`.
- App-level: `update` appears in `pyqmd --help`'s command list, and the
  existing callback-warning regression test (`tests/cli/test_app.py`) is
  extended to cover `update` (it follows the same no-op-callback
  single-command-module pattern as `embed`/`status`).

Parity-suite note: `update`/`collection update-cmd` are new CLI surface
not currently in `parity/scenarios/cli_scenarios.py`. Adding parity
scenarios for them is out of scope for this sub-project (the parity
suite's own scope was closed out separately). The feature inventory that
tracks this is the parity _design spec's_ "Feature inventory" table
(`2026-09-12-python-node-parity-suite-design.md` — `parity/README.md` has
no such tables, only a "Known gaps" section); its `update [--pull]` and
`collection set-update`/`update-cmd` rows (currently `not-yet-built`)
must be updated as part of this sub-project, and the `collection add`
row's note de-staled, so the inventory doesn't silently lie.
