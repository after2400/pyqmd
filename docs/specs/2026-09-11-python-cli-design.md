# Python CLI (sub-project #3) Design Spec

**Status:** Implemented (2026-09-11)

## Purpose

Build `pyqmd`, a Python CLI command surface on top of sub-project #2's
`qmd.store.Store`, porting the semantics (not the Bun-specific plumbing) of
the existing Node/TypeScript CLI at `src/cli/qmd.ts`. This is the third
sub-project in the Python/MLX rewrite (`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`),
building on sub-project #1 (`qmd.llm`, MLX embed/rerank/expand) and
sub-project #2 (`qmd.store`, SQLite FTS5 + sqlite-vec storage and hybrid
search, with `Store.query()` as its stable public API).

## Scope (deliberate v1 MVP)

**In scope**, a "core read/write loop":

- `collection add`, `collection list`, `collection show`, `collection remove`,
  `collection rename`
- `get`, `multi-get` (comma-separated docids/paths only — see Deferred)
- `ls`
- `search` (FTS), `vsearch` (vector), `query` (hybrid)
- `embed` (a scoped-down, explicit indexing step — see below)
- `status`

**Explicitly deferred**, matching or extending sub-project #2's own
deferral list (none of these block v1; they are candidates for later
sub-projects or fast-follows once real usage shows which matter):

- `init` (project-local `.qmd/index.yml` + `.qmd/index.sqlite`) — this is
  exactly the "file-based project config" sub-project #2 already deferred.
  v1 uses a single global DB; project-local indexes are a future sub-project
  once file-based config is in scope.
- `trust` / hooks / checked-in `.qmd` config trust model — depends on the
  same file-based project config as `init`.
- `skills` / `skill` — unrelated subsystem, no dependency on storage/CLI
  work, can be ported independently whenever needed.
- `doctor`, `bench`, `pull`, `context`, `update` — secondary utilities;
  `update` in particular is really "re-run `collection add` + `embed` for
  every registered collection," trivial to add once both exist, not needed
  for v1's core loop.
- `cleanup` — depends on sub-project #2's deferred cache/cleanup/vacuum
  utilities (`countOrphanedVectors`, `runCleanup`, etc.), which `Store`
  does not implement.
- `mcp` — this is sub-project #4 in its entirety.
- Glob-pattern matching in `multi-get` (`matchFilesByGlob`) — matches sub-project
  #2's deferred "comma-list glob resolution." v1's `multi-get` accepts only
  comma-separated docids or literal paths.
- Metadata filtering (`--filter`) on `search`/`query` — sub-project #2 never
  implemented this on `Store`; out of scope here too.
- AST-aware chunking (`--chunk-strategy auto`) — sub-project #5.
- All 6 output formats (`cli`, `json`, `csv`, `md`, `xml`, `files`) ARE in
  scope for v1 (confirmed explicitly — not deferred), since each is a
  fairly mechanical transform once the result data model exists.

## Naming: package, command, and DB location

Three separate names, deliberately different, to avoid any collision with
the live Node install while this rewrite is a parallel, coexisting tool:

- **Python package/import name**: stays `qmd` (`qmd.llm`, `qmd.store`,
  new `qmd.cli`) — unchanged from sub-project #2. This is a _rewrite_
  intended to eventually replace the Node tool, not a permanently separate
  product, so the import name doesn't change now only to change back later
  once the Node version is retired.
- **Installed console-script command**: `pyqmd`, distinct from the live
  Node `qmd` binary already on `PATH`. `pyproject.toml`'s
  `[project.scripts]` maps `pyqmd = "qmd.cli.app:main"`.
- **Cache/DB directory**: `~/.cache/pyqmd/index.sqlite`, matching the
  `pyqmd` command name, cleanly separate from the Node CLI's
  `~/.cache/qmd/index.sqlite`. Zero risk of two active processes fighting
  over one file, and the two tools can coexist during the transition.

## Package structure

```
qmd/
  cli/
    __init__.py
    app.py            # Typer app instance, command registration, entry point (main())
    _db.py            # DB path resolution (~/.cache/pyqmd/index.sqlite), Store lifecycle
    _output.py        # --format cli/json/csv/md/xml/files rendering
    commands/
      collection.py   # add/list/remove/rename/show
      documents.py    # get/multi-get/ls
      search.py       # search/vsearch/query
      embed.py        # embed
      status.py       # status
  store/
    _indexing.py      # NEW: scan_and_register_collection() -- free function,
                       # mirrors store.ts's own reindexCollection (also a free
                       # function taking a Store, not a Store method)
    store.py           # +2 new methods (find_document_by_identifier, clear_embeddings)
                       # +1 behavior fix (remove_collection now cascades, see below)
```

**Dependency addition**: `typer` (pulls in `click` + `rich`), chosen over
stdlib `argparse` for nicer decorator-based command registration and
built-in help formatting. This is the CLI sub-project's one new runtime
dependency; `qmd.llm`/`qmd.store` are unaffected.

## New capabilities required in `qmd.store` (small, expected additions)

Sub-project #2 is "done" in the sense that its own plan is complete and
reviewed, but integration work commonly surfaces small, legitimate gaps —
this is normal, not scope creep back into sub-project #2's territory.

1. **`Store.find_document_by_identifier(identifier: str) -> dict | None`**
   — resolves either a docid (a hash prefix, e.g. `abc123` or `#abc123`) or
   a `collection/path`-style string (optionally prefixed `qmd://`) to the
   underlying active document + content row. Needed by `get`/`multi-get`;
   `Store.find_active_document()` only resolves by exact `(collection, path)`
   today, not by docid.
2. **`qmd/store/_indexing.py`'s `scan_and_register_collection(store, path,
pattern, collection_name, ignore_patterns=None) -> ReindexResult`** — a
   free function (not a `Store` method, matching `store.ts`'s own boundary:
   `reindexCollection` is a free function there too), porting the
   directory-walk / glob-match / hash-and-register / deactivate-missing
   loop from `store.ts:1635-1764`. Explicitly drops two pieces of that
   original logic that depend on deferred features: metadata-extraction
   sync (`syncDocumentMetadata`) and orphaned-content cleanup
   (`cleanupOrphanedContent`) — neither exists on the Python `Store`.
   Returns a small dataclass (`ReindexResult`: `indexed`, `updated`,
   `unchanged`, `removed`, `skipped`, `skipped_files` — no `metadata_errors`
   or `orphaned_cleaned` fields, since those don't apply).
3. **`Store.remove_collection()` behavior change (a correction, verified
   against `store.ts:3698-3715`):** sub-project #2's current implementation
   only deletes the `collections` table row, leaving that collection's
   `documents`/`content_vectors`/`documents_fts` rows orphaned in the DB
   forever. `store.ts`'s `removeCollection` actually hard-deletes
   `documents WHERE collection = ?` and cleans up now-orphaned `content`
   rows (`WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)`).
   Update `Store.remove_collection()` to match this destructive behavior
   (also delete the now-orphaned `content_vectors`/`vectors_vec` rows for
   those hashes, and re-sync `documents_fts`, since `store.ts` doesn't have
   an FTS-triggers-only equivalent to worry about but Python's explicit-sync
   design does — those FTS rows must be deleted too, not left dangling).
   Sub-project #2's existing `test_remove_collection` test only checked
   non-destructive behavior; it needs updating alongside this change to
   assert documents are actually gone. The CLI layer prompts for
   confirmation (`Are you sure you want to remove 'NAME' and its N
documents? [y/N]`) before calling this — Typer's `typer.confirm()` — since
   this is now a real, irreversible data-loss operation.
4. **`Store.clear_embeddings(collection: str | None = None) -> int`** — new
   method deleting `content_vectors`/`vectors_vec` rows for hashes belonging
   to active documents in the given collection (or all collections if
   `None`), returning the count cleared. Used by `embed --force` to make
   already-embedded hashes eligible for re-embedding again (`index_content`'s
   own no-op check would otherwise skip them).

## Command behavior

| Command                                                                         | Behavior                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `collection add PATH --name N [--mask GLOB] [--exclude GLOB...]`                | `Store.add_collection()` then `scan_and_register_collection()`. Prints indexed/updated/unchanged/removed summary.                                                                                                                                                                                                                                                                                                                                                                                    |
| `collection list`                                                               | `Store.list_collections()`, formatted as a table (`cli`) or per `--format`.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `collection show NAME`                                                          | `Store.get_collection()` plus live doc/vector counts for that collection.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `collection remove NAME`                                                        | Prompts for confirmation (`typer.confirm()`), then calls the updated (destructive) `Store.remove_collection()`, which now cascade-deletes that collection's `documents`/`content_vectors`/`vectors_vec`/orphaned `content`/`documents_fts` rows, matching `store.ts`'s actual behavior (see "New capabilities," item 3).                                                                                                                                                                             |
| `collection rename OLD NEW`                                                     | `Store.rename_collection()` (already cascades to `documents.collection`, verified in sub-project #2).                                                                                                                                                                                                                                                                                                                                                                                                |
| `get FILE_OR_DOCID[:from[:count]]`                                              | `find_document_by_identifier()`, then slice by line range if given, apply line numbers unless `--no-line-numbers`.                                                                                                                                                                                                                                                                                                                                                                                   |
| `multi-get "a,b,c"`                                                             | Comma-split, `find_document_by_identifier()` per token, respecting `--max-bytes`/`-l`.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ls [collection[/path]]`                                                        | No arg: `Store.list_collections()`. With arg: `Store.get_active_document_paths()` filtered by the given path prefix.                                                                                                                                                                                                                                                                                                                                                                                 |
| `search QUERY [-c collection] [-n limit]`                                       | `Store.search_fts()` directly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `vsearch QUERY [-c collection] [-n limit]`                                      | `Store.search_vec()` directly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `query QUERY [-c collection] [-n limit] [--min-score] [--no-rerank] [--intent]` | `Store.query()` directly. `--intent` is implemented properly inside `Store.query()`/`_retrieve_and_fuse()` (disables the BM25 strong-signal bypass, weights chunk selection via `INTENT_WEIGHT_CHUNK`, prepends intent to the reranker's input text only) — decided during implementation planning to supersede the weaker verbatim-concatenation design originally sketched here; see the implementation plan's Global Constraints and Task 3.                                                      |
| `embed [-c collection] [--force]`                                               | Query distinct `(hash, doc)` pairs for active documents (scoped if `-c` given). If `--force`, call the new `Store.clear_embeddings(collection)` first. Then call `Store.index_content(hash, doc, model)` per hash.                                                                                                                                                                                                                                                                                   |
| `status`                                                                        | DB path/size (`os.path.getsize`), `SELECT COUNT(*) FROM documents WHERE active=1`, `SELECT COUNT(*) FROM content_vectors`, a pending-embed count (documents whose hash has no `content_vectors` row for the current embed model), most recent `modified_at`, collections list with their collection-level `context` field (no per-path context — deferred). No MCP daemon check (mcp is sub-project #4), no orphaned-vector check (cleanup deferred), no metadata-pending check (metadata deferred). |

## Output formats

Port all 6 (`cli`, `json`, `csv`, `md`, `xml`, `files`) from
`src/cli/formatter.ts`, for both search-result lists and get/multi-get
document lists. `cli` (human-readable, colorized where useful) is the
default. `_output.py` mirrors `formatter.ts`'s function-per-format-per-shape
structure as plain Python functions, not a class — consistent with
`qmd.store`'s existing "pure functions for stateless transforms" pattern.

## Error handling

- Typer's built-in argument-parsing errors (missing/malformed args) are
  used as-is — no custom wrapping needed.
- `Store`-raised exceptions expected at the CLI boundary (`sqlite3.IntegrityError`
  on duplicate collection names, `ValueError` on embedding dimension
  mismatch, a `find_document_by_identifier()` miss) are caught in each
  command function and re-raised as `typer.Exit(1)` after printing a clean,
  one-line message to stderr via `typer.echo(..., err=True)`. No raw
  tracebacks for these expected, user-facing failure modes.
- Unexpected exceptions (a real bug, a corrupted DB, an MLX model load
  failure) are NOT swallowed — they propagate normally so the real
  traceback is visible for debugging, matching the original CLI's
  `exitWithError` pattern (which prints the error but still exits non-zero,
  not a silent catch-all).

## Testing strategy

Matches sub-project #2's pattern:

- Unit tests per command module, using an in-memory `Store(":memory:", embed_fn=fake, rerank_fn=fake, expand_fn=fake)` and Typer's `CliRunner` (bundled with Typer/Click) to invoke commands in-process — no subprocess shelling.
- One `@pytest.mark.slow` end-to-end test: a real temp directory with a
  couple of markdown files, `collection add` → `embed` → `query`, using
  real MLX models, asserting a real relevant result comes back.
- `scan_and_register_collection()` gets its own focused unit tests (glob
  matching, hidden-file/dir exclusion, path-escape-the-collection-root
  rejection, re-add with changed/unchanged/removed files) independent of
  any CLI command, matching how sub-project #2 tested chunking/RRF/FTS-query
  as pure/isolated units before wiring them into `Store`.
