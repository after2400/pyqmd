# pyqmd `cleanup` command — design

**Status:** Implemented (2026-09-16)

**Sub-project #9** of the qmd Python/MLX rewrite. Roadmap:
`2026-09-10-python-mlx-rewrite-roadmap.md`, item #9.

## Problem

pyqmd has no way to reclaim space or compact its index. `documents` rows
that fall out of a collection (a file deleted, or a collection re-scanned
via `update`) are only ever soft-deleted (`active = 0`) — they stay in
the database forever, along with their `documents_fts` and
`document_metadata`/`document_metadata_values` rows. Content/vector rows
that become unreferenced (`Store.cleanup_orphaned_content()`, built for
`update`) get cleaned up automatically on every re-scan, but nothing ever
runs `VACUUM` or compacts the `documents_fts` b-trees, so deleted rows'
disk space is never reclaimed. There is also no way to hard-purge the
inactive document tombstones themselves.

This was explicitly deferred out of sub-project #8 (`update`)'s scope —
see that spec's decision 6 — and given its own roadmap slot rather than
folded in, since re-scanning and vacuuming are different operations with
different edge cases.

## Reference implementation

Node's `qmd cleanup [--dry-run]` (`src/cli/qmd.ts`'s `case "cleanup"`)
and `runCleanup`/`previewCleanup` (`src/store.ts`).

## Decisions made during brainstorming (2026-09-16)

1. **Stats granularity**: pyqmd reports **one** number for orphaned
   content, not Node's two (`orphanedVectors` + `orphanedContent`). Node's
   split is a byproduct of its own two-step implementation order (vectors
   and content are computed/deleted as genuinely separate passes there);
   pyqmd's existing `cleanup_orphaned_content()` already does both in one
   atomic pass and returns one count. At pyqmd's scale (~300 files, a
   handful of chunks per file) the two numbers would essentially always
   move together, so the extra number doesn't earn its keep. Reuses
   `update`'s existing wording: `Cleaned up N orphaned content hash(es)`.
2. **`llm_cache` clearing**: included, even though nothing in pyqmd
   currently writes to that table (no query-expansion/rerank response
   caching is wired up yet, unlike Node). Always reports 0 today; costs
   nothing to keep and needs no follow-up if/when pyqmd adds LLM response
   caching.
3. **No `-c`/`--collection` scoping**: matches Node (no such flag).
   Considered and rejected: of cleanup's four steps, only "purge inactive
   document rows" is even meaningfully scopable to one collection — the
   `llm_cache` clear, orphan-content cleanup, and VACUUM/FTS-optimize are
   inherently index-wide. Orphan-content cleanup in particular **must**
   stay index-wide for correctness, not just simplicity: content is
   hash-addressed and can be shared across collections, so "orphaned
   relative to collection X's active docs" could delete a content row
   still referenced by an active document in collection Y. A `-c` flag
   that only narrowed one of four steps while silently running the other
   three index-wide would be actively misleading.
4. **`status`'s orphan-count hint**: out of scope here. Node's `status`
   shows `Orphaned: N embedding chunks (X%) — run 'qmd cleanup'`; pyqmd's
   doesn't have this yet. Left to roadmap #10 (CLI output
   formatting/parity polish), which already owns `status`'s other
   missing sections — keeps `status` changes in one place instead of
   splitting across two sub-projects.

## Design

### Store layer (`src/qmd/store/store.py`)

New methods, placed near `cleanup_orphaned_content`:

- **`purge_inactive_documents(self) -> int`** — hard-deletes every
  `documents` row with `active = 0`, globally (no collection filter).
  Follows `remove_collection`'s established explicit-cleanup pattern
  (delete `documents_fts` rows by rowid, then
  `document_metadata_values`, then `document_metadata`, then the
  `documents` rows themselves) rather than relying on `ON DELETE CASCADE`
  — per this repo's own documented constraint that `PRAGMA foreign_keys`
  isn't reliably re-enabled on reopening an existing on-disk DB. Commits
  its own transaction as one discrete unit (FTS-optimize and VACUUM still
  run afterward, as separate steps). Returns the count of rows purged.
- **`clear_llm_cache(self) -> int`** — `DELETE FROM llm_cache`, commits,
  returns `cursor.rowcount`.
- **`count_inactive_documents(self) -> int`**,
  **`count_llm_cache(self) -> int`**,
  **`count_orphaned_content(self) -> int`** — read-only `COUNT(*)`
  helpers for `--dry-run`, mirroring the mutating methods' `WHERE`
  clauses exactly (`count_orphaned_content` mirrors
  `cleanup_orphaned_content`'s existing orphan-detection query).
- **`vacuum(self) -> None`** — `VACUUM`.
- **`optimize_documents_fts(self) -> None`** — `INSERT INTO
documents_fts(documents_fts) VALUES('optimize')`, guarded the same way
  `cleanup_orphaned_content` already guards `vectors_vec` access: skip
  quietly (no error) if `documents_fts` doesn't exist yet (a brand new,
  never-indexed database).

### CLI layer (`src/qmd/cli/commands/cleanup.py`)

New single-command Typer app, same shape as `update.py`/`status.py`
(a no-op `@app.callback()` so Typer doesn't collapse the single
`@app.command("cleanup")` into a bare top-level command — see
`update.py`'s callback docstring for why this matters). Mounted flat on
the root app in `qmd/cli/app.py` alongside `embed`/`status`/`update`.

```
pyqmd cleanup [--dry-run]
```

Order of operations (real run):

1. `store.clear_llm_cache()`
2. `store.purge_inactive_documents()`
3. `store.cleanup_orphaned_content()` (existing method, unchanged)
4. `store.optimize_documents_fts()`
5. `store.vacuum()`

Steps 2 and 3's relative order doesn't affect correctness here (unlike
Node, where `cleanupOrphanedContent` must run _after_ inactive documents
are hard-deleted, because its query checks "referenced by any document
row" rather than "referenced by an active one). pyqmd's
`cleanup_orphaned_content` already filters by `active = 1` directly, so
it produces the same result regardless of whether inactive rows have
been purged yet. This order is kept anyway because it reads naturally:
remove the tombstones, then clean up whatever they leave unreferenced.

Output convention matches this codebase's existing pattern from
`update`/`collection add`: **only print a stat line when its count is
nonzero** — no "0 items" filler lines, no color (color scheme is
roadmap #10's scope, not this one). The FTS-optimize/VACUUM line always
prints, since that step always runs regardless of what else there was to
clean:

```
✓ Cleared 3 cached API responses
✓ Removed 12 inactive document record(s)
✓ Cleaned up 5 orphaned content hash(es)
✓ FTS compacted, database vacuumed
```

(Each of the first three lines is independently omitted when its count
is 0; an all-zero index prints only the last line.)

`--dry-run`: prints `Dry run — no changes made.` first, then the same
conditional lines with "Would" phrasing, sourced from the three
`count_*` helpers instead of the mutating methods:

```
Dry run — no changes made.

Would clear 3 cached API responses
Would remove 12 inactive document record(s)
Would clean up 5 orphaned content hash(es)
Would compact FTS and vacuum the database
```

No writes happen in `--dry-run` mode at all — not even inside a
rolled-back transaction; the dry-run path calls only the `count_*`
helpers and never touches the mutating methods.

### Error handling

No new exception types, and no new entries needed in `_errors.py`'s
`EXPECTED_EXCEPTIONS`. `VACUUM`/FTS-optimize failures (e.g. a locked
database from a concurrent connection) are not specially caught —
matching the conservative stance `update`'s spec already took: only
catch what's anticipated; let genuinely unexpected `sqlite3` errors
surface as an uncaught traceback rather than inventing a clean-error
path for a failure mode with no realistic single-user trigger today.

## Testing strategy

Store-level (new `tests/store/test_cleanup.py`, following the existing
per-feature test-file convention, e.g. `test_orphan_cleanup.py`):

- `purge_inactive_documents`: deletes only `active = 0` rows (an active
  document in the same or a different collection survives untouched);
  removes the purged document's `documents_fts` row (searchable-by-FTS
  check before/after) and its `document_metadata`/
  `document_metadata_values` rows; returns the correct count; returns 0
  on an index with no inactive documents.
- `clear_llm_cache`: removes all rows, returns the count, returns 0 when
  already empty.
- `count_inactive_documents`/`count_llm_cache`/`count_orphaned_content`:
  each matches what its mutating counterpart would have deleted, without
  actually deleting anything (call the count method, then the mutating
  method, assert the mutating method's return value equals the earlier
  count and equals 0 on a second call).
- `vacuum`/`optimize_documents_fts`: smoke tests — run without raising,
  including `optimize_documents_fts` on a fresh `:memory:` store that has
  never indexed anything (exercises the "table doesn't exist yet" guard).

CLI-level (new `tests/cli/test_cleanup_command.py`, `CliRunner` against
an in-memory `Store`, matching `test_update_command.py`'s pattern):

- Real run with something in every category (llm_cache row, an inactive
  document with metadata, an orphaned content hash): all four checkmark
  lines appear with correct counts, and a follow-up query against the
  store confirms every targeted row is actually gone (not just that the
  right text was printed).
- Real run on a clean index (nothing to clean anywhere): only the
  `✓ FTS compacted, database vacuumed` line appears; the other three are
  absent.
- `--dry-run` with something in every category: the `Dry run — no
changes made.` header plus all four "Would ..." lines appear with
  correct counts, and a follow-up query against the store confirms
  **nothing** was actually removed (`llm_cache` row still present,
  inactive document row still present with its metadata and FTS entry
  intact, orphaned content row still present).
- `--dry-run` on a clean index: only `Would compact FTS and vacuum the
database` appears.

## Out of scope

- `status`'s orphan-count hint (decision 4) — roadmap #10.
- `-c`/`--collection` scoping (decision 3) — rejected outright, not
  deferred; see decision 3's correctness argument.
- Any change to `cleanup_orphaned_content` itself — reused as-is.
- Node's colored (`c.green`/`c.dim`) output — roadmap #10's scope.
