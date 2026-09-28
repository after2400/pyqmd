# `collection include`/`exclude` — design

**Status:** Implemented (2026-09-17)

Roadmap: `2026-09-10-python-mlx-rewrite-roadmap.md`, "Next step" section,
pickup-order item 3. Not a numbered sub-project — a scoped CLI/Store
feature gap, already tracked in `COMMAND_STATUS.md`'s `collection
include` / `exclude` row (currently ❌ "Not started, not on the
roadmap") and in that file's "Not added to the roadmap this round"
note.

## Problem

`collections.include_by_default` (`src/qmd/store/_schema.py:53`,
`INTEGER NOT NULL DEFAULT 1`) already exists and is already _displayed_
— `collection list` shows a yellow `[excluded]` tag when it's false
(`collection.py:83`) — but nothing sets it after collection creation,
and nothing reads it to scope default search results. Two gaps:

1. No CLI command to flip the flag on an existing collection.
2. `search`/`vsearch`/`query` (CLI and MCP) treat "no `-c` given" as
   "search every collection," ignoring `include_by_default` entirely —
   the toggle is currently display-only.

## Reference implementation (Node)

- `qmd collection include <name>` / `qmd collection exclude <name>`
  (`src/cli/qmd.ts:4614-4632`, help text `4661-4680`): one positional
  arg, no flags. Whole-collection membership in default search scope —
  unrelated to the `--mask`/`--exclude` glob flags on `collection add`,
  which control what gets indexed at all. Validates the collection
  exists, then `updateCollectionSettings(name, { includeByDefault:
include })` (`src/collections.ts:283-310`), printing `✓ Collection
'<name>' included in/excluded from default queries`.
- Default-scope resolution: `resolveCollectionFilter(raw, useDefaults)`
  (`qmd.ts:2717-2737`) — if no `-c` was passed _and_ `useDefaults` is
  true, returns `getDefaultCollectionNames()`
  (`index.ts:513-516`/`collections.ts:266-278`, collections filtered by
  `includeByDefault !== false`). If `-c` _was_ passed, every named
  collection is used as given — **no exclusion check applies to an
  explicitly named collection**. `search()`, `vectorSearch()`, and
  `querySearch()` (`qmd.ts:2871`, `2921`, `2966`) all call this
  identically; the MCP server (`src/mcp/server.ts:206`, `904`) has its
  own copy of the same call.
- Node's `collection show` prints an `Include: yes (default)` /
  effectively-no line (`qmd.ts:4650`).

Node's dual-backend complication (a legacy YAML config store kept in
sync with the SQLite `collections` table) doesn't apply to pyqmd —
`include_by_default` on the `collections` table is pyqmd's sole source
of truth already.

## Design

### Store layer

`src/qmd/store/store.py`:

- `set_collection_include_by_default(self, name: str, include: bool) ->
None` — modeled on `set_collection_update_command` (`store.py:
361-365`): `UPDATE collections SET include_by_default = ? WHERE name
= ?`, commit. Existence-checking stays in the CLI layer, matching
  every other `collection` subcommand's pattern (`update_cmd`, `rename`,
  `remove` all check via `store.get_collection(name)` before calling
  their mutator).
- `_normalize_collections` (`store.py:121-130`, currently a module-level
  function with exactly two callers, both inside `Store` methods:
  `search_fts` at `store.py:1131`, `search_vec` at `store.py:1239` — no
  other internal or test code imports it directly) becomes a `Store`
  method, `self._normalize_collections`, so it can resolve the "nothing
  given" case against the DB. Both call sites change from
  `_normalize_collections(collection)` to
  `self._normalize_collections(collection)`. New behavior: when
  `collection` is `None`, instead of returning `None` unconditionally,
  it calls the new `get_default_collection_names()` below and returns
  that instead. A bare string or explicit list is still normalized and
  returned as-is (no exclusion check applied) — this preserves Node's
  override rule that naming a collection with `-c` always searches it
  regardless of `include_by_default`.
- `get_default_collection_names(self) -> list[str] | None`: returns
  `None` if every collection has `include_by_default = 1` (the common
  case — no restriction needed, preserving today's fast path exactly:
  no `IN (...)` clause, no `fts_limit` overfetch-window widening), or
  the explicit list of included collection names otherwise. Concretely:
  ```python
  def get_default_collection_names(self) -> list[str] | None:
      excluded = self.conn.execute(
          "SELECT 1 FROM collections WHERE include_by_default = 0 LIMIT 1"
      ).fetchone()
      if excluded is None:
          return None
      rows = self.conn.execute(
          "SELECT name FROM collections WHERE include_by_default = 1"
      ).fetchall()
      return [r["name"] for r in rows]
  ```

This is the single choke point: `search_fts` (`store.py:1092`) and
`search_vec` (`store.py:1217`) are the only two callers of
`_normalize_collections`, and `query()` (`store.py:1469`) delegates to
both of them via `_retrieve_and_fuse` (`store.py:1387`), passing the
same raw `collection` argument through unchanged — it does no
normalization of its own. So fixing `_normalize_collections` once fixes
all three CLI commands _and_ the MCP `query` tool (`qmd/mcp/server.py`'s
`_query_impl` calls `store.query(...)` directly) with **no changes
needed in `search.py` or `mcp/server.py` at all**. This is the approved
"Approach A" from brainstorming: resolve inside `Store`, not duplicated
across call sites the way Node's `resolveCollectionFilter` is.

### CLI layer

`src/qmd/cli/commands/collection.py`, new `include`/`exclude` commands
following the `update_cmd` pattern exactly:

```python
def _set_include(name: str, include: bool) -> None:
    store = get_store()

    def _run():
        if store.get_collection(name) is None:
            raise ValueError(f"No such collection: {name}")
        store.set_collection_include_by_default(name, include)

    run_or_exit(_run)
    state = "included in" if include else "excluded from"
    typer.echo(f"{green('✓')} Collection '{cyan(name)}' {state} default queries")


@app.command("include")
def include(name: str = typer.Argument(...)) -> None:
    """Include a collection in default (no -c) search results."""
    _set_include(name, True)


@app.command("exclude")
def exclude(name: str = typer.Argument(...)) -> None:
    """Exclude a collection from default (no -c) search results."""
    _set_include(name, False)
```

Idempotent by design (matching Node): calling `include` on an
already-included collection just re-sets the same value and prints the
same confirmation — no special-cased no-op branch.

`collection show` gains one line, reusing the row it already fetches:

```python
typer.echo(f"{dim('Include:')} {'yes (default)' if collection['include_by_default'] else 'no'}")
```

Placed after the existing `Update:` line (before `Documents:`), matching
Node's field ordering.

### Out of scope

- Any change to `update`/`embed`/`cleanup` — exclusion only affects
  default _search_ scope, matching Node exactly. Excluded collections
  still get scanned/re-indexed/embedded normally.
- `collection list`'s `[excluded]` tag — already correct, unaffected by
  this work.
- File-level include/exclude glob patterns — that's `collection add
--mask`/`--exclude`, a different, already-implemented feature; naming
  overlap with Node's command names is coincidental (Node's own
  `include`/`exclude` are whole-collection toggles too, per the
  reference section above).

## Testing plan

1. `tests/store/test_collections.py`: `get_default_collection_names`
   returns `None` when all collections default-included; returns the
   included-only subset once one collection is excluded;
   `set_collection_include_by_default` flips the column and is
   idempotent.
2. `tests/cli/test_collection_command.py`: `collection include`/
   `exclude` success message and exit code; error on a nonexistent
   collection; `collection show`'s new `Include:` line reflecting both
   states.
3. Behavioral test proving the actual fix, against a `Store` with two
   collections (one excluded) each containing a document matching the
   same query: a plain `store.search_fts`/`search_vec`/`query` call (no
   `collection` arg) returns only the included collection's hits, while
   passing the excluded collection's name explicitly via `collection=`
   still returns it (override). Exercising `store.query()` directly is
   sufficient to also cover the MCP path, since `_query_impl` is a thin
   wrapper with no scoping logic of its own — no separate MCP-layer test
   needed.
4. Extend the parity suite's `collection_lifecycle` flow
   (`parity/scenarios/cli_flow_scenarios.py`) with `exclude`/`include`
   steps (using the existing `_ok_shape` helper, matching the depth of
   its other confirmation-only steps like `set_update_cmd`) plus a
   `list`/`show` step afterward to confirm the tag/line landed — same
   shallow structural-parity depth this flow already uses elsewhere, not
   a new byte-exact shape parser.
5. `just test-fast`, `just test-parity`, `just lint`.
6. Update `COMMAND_STATUS.md`'s `collection include` / `exclude` row to
   ✅, remove the "Not added to the roadmap this round" note (superseded
   by this spec), and mark roadmap pickup-order item 3 done.
