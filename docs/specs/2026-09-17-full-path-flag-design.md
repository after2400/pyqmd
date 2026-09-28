# `--full-path` output flag — design

**Status:** Implemented (2026-09-18)

Roadmap: `2026-09-10-python-mlx-rewrite-roadmap.md`, "Backlog / housekeeping"
section, `--full-path output flag` entry. Not a numbered sub-project — a
scoped output-formatting gap, already tracked as a genuine parity gap
(unlike the `pull` stub, which had no Node analog to build against) in
`COMMAND_STATUS.md`.

## Problem

Node's `get`/`multi-get`/`search`/`vsearch`/`query` all share a
`--full-path` boolean ("show on-disk paths instead of `qmd://` + docid").
pyqmd has no equivalent on any of the five — `src/qmd/cli/_output_search.py`
even documents this explicitly as a known deferral: `"it does NOT include
... --full-path resolution, which are deferred (see design spec)"`. This
is that spec.

## Reference implementation (Node)

- One shared flag, no per-command duplication: `"full-path": { type:
"boolean" }` in the single `parseArgs` options object (`qmd.ts:3147`,
  help text `qmd.ts:3737`). For `search`/`vsearch`/`query` it's threaded
  into the shared `OutputOptions.fullPath` field consumed by the single
  shared formatter `outputResults()` (`qmd.ts:2484`); for `get`/`multi-get`
  it's a plain parameter into `getDocument(..., fullPath)`
  (`qmd.ts:1260`) / `multiGet(..., fullPath)` (`qmd.ts:1362`).
- Path construction, `renderFullPath()` (`qmd.ts:1228-1238`): given an
  already-resolved absolute fs path, `realpathSync` it (fallback to the
  raw absolute path on failure); `"./"` if it equals cwd; a `./`-prefixed
  relative path if it's a cwd subpath; otherwise the absolute realpath.
  The absolute path itself comes from `resolveVirtualPath()`
  (`store.ts:809-819`): parse the `qmd://collection/path` URI, look up
  the collection's stored root path, `resolve(collection.pwd,
relativePath)`, and reject (return `null`) if the result escapes
  `collection.pwd` (`isPathInsideDir`).
- **Effect on output is not just a string swap — the docid is dropped**
  for every row that resolves; a row that falls back keeps its docid so
  it stays addressable. This is format-agnostic: Node's `multiGet()` and
  `outputResults()` both compute a resolutions map once, then every
  format branch (json/csv/files/md/xml/cli) consults it via small
  `identOf()`/`docidOf()` (multi-get) or `displayPathFor()`/`showDocid()`
  (search) helpers — not re-resolved per format.
- **Fallback**: if resolution fails (moved/deleted file since indexing)
  or the file no longer exists on disk, that row falls back to the normal
  `qmd://collection/path` + docid rendering, and a single aggregated
  count is reported via `warnUnresolvedFullPaths()` (`qmd.ts:1248-1258`)
  to stderr after all output — never mixed into stdout, so machine
  formats stay parseable.
- No interaction with other flags (`--filter`, `--min-score`, etc.); works
  identically across every `--format`. Node's own tests
  (`test/cli.test.ts:1268-2438`, `test/full-path-fallback.test.ts`,
  `test/path-fidelity.test.ts:245`) confirm: `./`-prefix under cwd,
  absolute realpath otherwise, docid dropped only when resolved, graceful
  fallback + warning when the file is gone.

## Design

### New Store method: `resolve_full_path`

`src/qmd/store/store.py`, placed immediately after `detect_collection_for_path`
(`store.py:457-479`) — the reverse direction of that existing method
(fs path → collection there; collection+path → fs path here), same
realpath/style conventions:

```python
def resolve_full_path(self, collection: str, path: str) -> str | None:
    """The reverse of detect_collection_for_path: given a collection name
    and a document's relative path (as stored in the `documents` table),
    return its realpath'd on-disk location, or None if the collection
    doesn't exist, the resolved path escapes the collection's root
    (mirrors Node's resolveVirtualPath's isPathInsideDir check), or the
    file no longer exists on disk (moved/deleted since indexing)."""
    coll = self.get_collection(collection)
    if coll is None:
        return None
    real_root = os.path.realpath(coll["path"])
    candidate = os.path.realpath(os.path.join(coll["path"], path))
    if candidate != real_root and not candidate.startswith(real_root + os.sep):
        return None
    if not os.path.exists(candidate):
        return None
    return candidate
```

### New CLI module: `src/qmd/cli/_fullpath.py`

Pure rendering (no Store access) plus one shared "apply to a list of
display rows" helper reused by every list-based command:

```python
"""--full-path resolution shared by get/multi-get/search/vsearch/query.
Node's own outputResults()/multiGet() compute this once per invocation,
then every format branch consults the result -- not re-resolved per
format. render_full_path mirrors Node's renderFullPath (qmd.ts:1228-1238);
apply_full_path mirrors the resolutions-map-then-consult pattern,
generalized across DisplayResult and DocumentEntry/MultiGetEntry since
both already expose mutable display_path/docid fields."""

import os


def render_full_path(fs_path: str) -> str:
    cwd = os.path.realpath(os.getcwd())
    if fs_path == cwd:
        return "./"
    if fs_path.startswith(cwd + os.sep):
        return "./" + fs_path[len(cwd) + 1 :]
    return fs_path


def apply_full_path(rows, store) -> int:
    """Mutates each row's display_path/docid in place when its qmd://
    path resolves to a real on-disk file; leaves it untouched (still
    addressable via its original docid) otherwise. Returns the number of
    rows that could not be resolved, for the caller to warn about."""
    unresolved = 0
    for row in rows:
        without_scheme = row.display_path.removeprefix("qmd://")
        collection, _, path = without_scheme.partition("/")
        resolved = store.resolve_full_path(collection, path) if path else None
        if resolved is None:
            unresolved += 1
            continue
        row.display_path = render_full_path(resolved)
        row.docid = None
    return unresolved


def format_fullpath_warning(unresolved: int) -> str:
    noun = "document" if unresolved == 1 else "documents"
    return (
        f"Warning: {unresolved} {noun} could not be resolved to a full "
        "path (file may have moved); showing qmd:// links instead."
    )
```

### `DisplayResult.docid` becomes optional

`src/qmd/cli/_types.py`: `docid: str` → `docid: str | None = None` (matches
the convention `DocumentEntry`/`MultiGetEntry` already use). `_output_search.py`'s
6 formatter functions each get the same truthy-gate `_output_documents.py`
already has for `MultiGetEntry`/`DocumentEntry`:

- `search_results_to_json` (line 79): `"docid": f"#{row.docid}"` unconditional
  → only set the `docid` dict key `if row.docid`.
- `search_results_to_csv` (line 116): `f"#{row.docid}"` → `f"#{row.docid}" if row.docid else ""`.
- `search_results_to_files` (line 133): same conditional-empty-string treatment.
- `search_results_to_markdown` (line 159): drop the `**docid:** ...` line entirely when falsy.
- `search_results_to_xml` (line 189): drop the `docid="..."` attribute entirely when falsy.
- `search_results_to_cli` (line 210): drop the `#{row.docid}` suffix when falsy.

`_output_documents.py`'s formatters need **no changes** — they already
gate on `e.docid` truthiness everywhere.

### Threading through the 5 commands

- **`search`/`vsearch`/`query`** (`src/qmd/cli/commands/search.py`): add
  `full_path: bool = typer.Option(False, "--full-path")` to each of the
  three command signatures. After building the `display: list[DisplayResult]`
  list (already happens today via `_search_result_to_display`/
  `_hybrid_result_to_display`), if `full_path`: `unresolved =
apply_full_path(display, store)`, then `if unresolved: typer.echo(format_fullpath_warning(unresolved), err=True)`.
- **`multi-get`** (`src/qmd/cli/commands/documents.py`): add the same
  `--full-path` option. `resolve_multi_get`'s returned `MultiGetEntry`
  list already has the right shape for `apply_full_path`, but `not_found`
  entries must be filtered out first — their `display_path` is the raw
  unmatched token (no `qmd://` scheme), and `apply_full_path`'s
  empty-`path` guard would otherwise miscount them as unresolved paths,
  double-warning alongside their existing separate "Not found: ..."
  message. Call `apply_full_path([r for r in resolved if r.not_found is
None], store)` before the existing per-entry loop that builds
  `DocumentEntry`s (that loop already special-cases `r.not_found`
  separately and is unaffected).
- **`get`** (`documents.py`): add `--full-path`. `get` doesn't go through
  either dataclass — it has its own bespoke header line (`documents.py:39-43`).
  When `--full-path` is set: `resolved = store.resolve_full_path(doc["collection"], doc["path"])`;
  if resolved, header becomes just `render_full_path(resolved)` (no docid
  suffix); otherwise keep today's header and print the same
  `format_fullpath_warning(1)` to stderr.

### Out of scope

- Node's OSC-8 clickable-hyperlink terminal integration — `_output_search.py`'s
  module docstring already notes this was deferred independently of
  `--full-path`, and stays deferred; unrelated to this flag.
- Any change to how `collection add`/`update` store a collection's root
  path (`coll["path"]` is used as-is, exactly like `detect_collection_for_path`
  already does).

## Testing plan

1. `tests/store/test_collections.py` (or a new `test_resolve_full_path.py`
   alongside it): `resolve_full_path` returns the realpath'd fs path for a
   real indexed file; returns `None` for a path that escapes the
   collection root (e.g. `path="../../etc/passwd"`); returns `None` when
   the file has been deleted since indexing (create, index, delete, then
   resolve).
2. `tests/cli/` — a new `test_fullpath.py` for the pure helpers:
   `render_full_path` returns `"./"` for cwd itself, a `./`-prefixed
   relative path for a cwd subpath, and the untouched absolute path
   otherwise; `apply_full_path` mutates matching rows and returns the
   correct unresolved count for a mixed resolved/unresolved batch.
3. Per-command CLI tests (extending `test_search_commands.py`,
   `test_collection_command.py`'s sibling `test_documents_command.py` or
   equivalent) for `get`, `multi-get`, `search`, `vsearch`, `query`:
   `--full-path` swaps the displayed path and omits the docid on a
   resolved row; a row whose file was deleted after indexing falls back
   to `qmd://...#docid` and the stderr warning appears; at least one
   non-default `--format` (e.g. `json`) tested for `multi-get`/`search`
   to confirm docid omission is format-aware, not CLI-only.
4. Parity suite: Node's and pyqmd's captures index the scifact corpus
   from different absolute filesystem locations (isolated capture
   directories on each side), so an exact byte-for-byte path comparison
   isn't meaningful here the way `--format json`'s file-set checks are.
   Add `--full-path` variants to existing `get`/`search`/`multi-get`
   scenarios in `parity/scenarios/cli_scenarios.py` with an
   invariant-based shape check (output isn't `qmd://`-prefixed, docid is
   absent, path ends with the expected relative suffix) — matching this
   repo's existing precedent for `vsearch`/`query` ("verified invariants
   only") rather than claiming exact-match parity.
5. `just test-fast`, `just test-parity`, `just lint`.
6. Update `COMMAND_STATUS.md`'s `search`/`vsearch`/`query`/`get`/`multi-get`
   rows' notes to mention `--full-path` support, and mark this backlog
   entry done in the roadmap doc.
