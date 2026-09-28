# `multi-get` glob-pattern support — design

**Status:** Implemented (2026-09-17)

Roadmap: `2026-09-10-python-mlx-rewrite-roadmap.md`, "Next step" section,
pickup-order item 2. Not a numbered sub-project — a scoped feature gap,
already tracked as a known limitation before this work
(`COMMAND_STATUS.md`'s `multi-get` row, `parity/README.md`'s "Known
gaps" section, `parity/test_structural.py`'s
`multi_get_glob_pattern_not_supported` xfail).

## Problem

pyqmd's `multi-get <pattern>` (`src/qmd/cli/_multiget.py`'s
`resolve_multi_get`) only ever comma-splits `pattern` and resolves each
token via `Store.find_document_by_identifier` (docid, `qmd://`, or bare
path). A glob pattern like `17*.md` is treated as one literal,
non-matching token. `multi-get`'s own CLI help text already says "Get
multiple documents by glob or comma-separated docid/path list" — the
glob half was never built. `_multiget.py` backs both the CLI's
`multi-get` command (`src/qmd/cli/commands/documents.py`) and the MCP
`multi_get` tool (`qmd/mcp/server.py`), so fixing it fixes both surfaces
at once.

This is a real, already-snapshotted parity gap, not new scope discovery:
`parity/node_ref/scifact/cli/multi_get_glob_pattern_not_supported.json`
already holds Node's real, correct captured output
(`{"exit_code": 0, "line_count": 9}`) for a glob-pattern scenario. No new
capture run is needed — fixing pyqmd's behavior should make the existing
`xfail(strict=True)` in `parity/test_structural.py` start passing, which
requires removing the marker (the suite's own convention: an unexpected
XPASS fails loudly until it's removed).

## Reference implementation

Node's `multiGet()` dispatch (`src/cli/qmd.ts:1362-1407`) and
`matchFilesByGlob` (`src/store.ts:3485-3505`), using the `picomatch`
npm package. Real, tested Node behaviors this must reproduce
(`test/store.test.ts:3313-3377`):
`journals/*.md` matches only files directly under `journals/`, not
deeper nesting (segment-aware `*`); `{readme,changelog}.md` brace
expansion; `<collection>/*.md` collection-scoped patterns.

## Findings from investigating this (avoid re-deriving them)

- **Python's stdlib `fnmatch` is not a drop-in for `picomatch`.**
  Verified against Node's own tests: `fnmatch.fnmatch()`'s `*` matches
  across `/` (no segment awareness — `fnmatch.fnmatch("sub/dir/file.md",
"*.md")` is `True` in Python, `False` under picomatch's default), and
  `fnmatch` has no brace-expansion support at all (`{a,b}` would be
  matched as a literal string, never expanded). Both are real, tested
  Node behaviors (`test/store.test.ts`'s `journals/*.md` and
  `{readme,changelog}.md` cases), not obscure edge cases.
- **`wcmatch.glob.globmatch()` (with `flags=glob.BRACE`) reproduces both
  correctly**, verified directly:
  `globmatch("sub/dir/file.md", "*.md")` → `False` (segment-aware, like
  picomatch); `globmatch("journals/2024-01.md", "journals/*.md")` →
  `True`; `globmatch("readme.md", "{readme,changelog}.md", flags=BRACE)`
  → `True`, `globmatch("license.md", "{readme,changelog}.md",
flags=BRACE)` → `False`; case-sensitive by default, matching
  picomatch. New dependency, pinned `wcmatch>=11.0,<12.0` (11.0.1
  confirmed working during investigation; the constraint tracks the
  same major line rather than trusting the maintainer's own semver
  going forward, matching this repo's other floor-pinned dependencies).
- **Node's own comma-vs-glob dispatch has a real, tested quirk this
  spec ports as-is, not "improves on"**: `isCommaSeparated` (`qmd.ts:
1366`) is `pattern.includes(',') && !pattern.includes('*') &&
!pattern.includes('?') && !pattern.includes('{')`. A pattern with
  both a comma AND any of `*`/`?`/`{` is **not** treated as a comma
  list — the _entire_ string (comma included) becomes one glob pattern,
  which in practice matches nothing (globs don't treat a bare `,`
  outside `{}` specially). Node does not support mixing a comma-list
  with glob syntax in one call. Porting this exactly, rather than
  "fixing" it, keeps pyqmd's behavior identical to Node's for every
  input, including this one.
- **Zero-match exit behavior already matches** and needs no new code:
  pyqmd's `multi_get` CLI command (`documents.py:104-106`) already
  exits 1 with a message when `resolve_multi_get` returns no entries
  (today this only happens via all-comma-tokens-not-found; once the
  glob branch exists, zero glob matches naturally hits the same path) —
  matching Node's `"No files matched pattern"` / exit 1 for an empty
  glob result. No change needed here; noted so it isn't mistaken for a
  gap during implementation.

## Design

### New dependency

`pyproject.toml`: add `"wcmatch>=11.0,<12.0"` to `[project]
dependencies`.

### `Store.find_documents_by_glob` (`src/qmd/store/store.py`)

New method, placed near `find_document_by_identifier`:

```python
def find_documents_by_glob(self, pattern: str) -> list[dict]:
    """Active documents (all collections) whose qmd://collection/path,
    bare path, or collection/path matches `pattern`. Mirrors Node's
    matchFilesByGlob (store.ts) -- same three-candidate matching, same
    brace-expansion/segment-aware-wildcard semantics (wcmatch.glob with
    the BRACE flag matches picomatch's defaults, verified directly; see
    the design spec's findings). Sorted by (collection, path) for a
    stable, deterministic order -- Node's own SQL has no ORDER BY, but
    the parity check only compares line count, not row order, so this
    is a pyqmd-side improvement, not a divergence anything depends on.
    Returns the same row shape as find_document_by_identifier (full
    `documents` columns plus `doc`), so callers can treat both sources
    uniformly."""
    from wcmatch import glob as wcglob

    rows = self.conn.execute(
        """
        SELECT d.*, c.doc AS doc
        FROM documents d JOIN content c ON c.hash = d.hash
        WHERE d.active = 1
        """
    ).fetchall()
    matched = []
    for row in rows:
        doc = dict(row)
        candidates = (
            f"qmd://{doc['collection']}/{doc['path']}",
            doc["path"],
            f"{doc['collection']}/{doc['path']}",
        )
        if any(wcglob.globmatch(c, pattern, flags=wcglob.BRACE) for c in candidates):
            matched.append(doc)
    matched.sort(key=lambda d: (d["collection"], d["path"]))
    return matched
```

### `_multiget.py` dispatch + refactor

Add `_is_glob_pattern` (the ported Node heuristic) and extract the
existing per-document → `MultiGetEntry` conversion (currently inlined
in the comma-token loop) into a shared `_entry_from_doc` helper, so the
new glob branch and the existing comma/docid branch build entries
identically:

```python
_GLOB_CHARS = ("*", "?", "{")


def _is_glob_pattern(pattern: str) -> bool:
    """Mirrors Node's isCommaSeparated exclusion check (qmd.ts's
    multiGet): a comma-separated list is only treated as such when it
    contains none of these characters; otherwise (including a bare
    pattern with no comma at all) the whole string is one glob pattern.
    See the design spec's findings for why this exact heuristic, not a
    "smarter" one, is what gets ported."""
    return any(c in pattern for c in _GLOB_CHARS)


def _entry_from_doc(doc: dict, max_bytes: int) -> MultiGetEntry:
    display_path = f"qmd://{doc['collection']}/{doc['path']}"
    docid = doc["hash"][:6]
    body_length = len(doc["doc"].encode("utf-8"))
    if body_length > max_bytes:
        return MultiGetEntry(
            display_path=display_path,
            title=doc["title"],
            body="",
            skipped=True,
            skip_reason=(
                f"File too large ({body_length // 1024}KB > {max_bytes // 1024}KB). "
                f"Use 'pyqmd get {display_path}' to retrieve."
            ),
            docid=docid,
        )
    return MultiGetEntry(
        display_path=display_path, title=doc["title"], body=doc["doc"], docid=docid
    )


def resolve_multi_get(store: Store, pattern: str, max_bytes: int) -> list[MultiGetEntry]:
    if _is_glob_pattern(pattern):
        return [_entry_from_doc(doc, max_bytes) for doc in store.find_documents_by_glob(pattern)]

    names = [s.strip() for s in pattern.split(",") if s.strip()]
    entries: list[MultiGetEntry] = []
    for name in names:
        doc = store.find_document_by_identifier(name)
        if doc is None:
            entries.append(MultiGetEntry(display_path=name, title="", body="", not_found=name))
            continue
        entries.append(_entry_from_doc(doc, max_bytes))
    return entries
```

No changes needed to `documents.py` (the CLI command) or `qmd/mcp/
server.py` (the MCP tool) — both already call `resolve_multi_get` and
consume `MultiGetEntry` uniformly regardless of which branch produced
it.

### Removing the known-gap marker

Once the fix is verified against the real Node snapshot:

- `parity/test_structural.py`: remove the
  `"multi_get_glob_pattern_not_supported"` entry from `_CLI_KNOWN_GAPS`.
- `parity/README.md`: remove the corresponding bullet from "Known
  gaps".
- `COMMAND_STATUS.md`: update the `multi-get` row's "Matching parity"
  and notes to reflect glob support, same as the mutating-commands
  work's Task 8 pattern.
- Roadmap doc: mark pickup-order item 2 done.

## Testing plan

1. `tests/cli/test_multiget_shared.py` (existing file, already tests
   `resolve_multi_get` directly against an in-memory `Store` with no
   MLX/embedding involved): add cases mirroring Node's own tested
   behaviors —
   - a glob matching multiple documents in one collection (`*.md`)
   - a `<collection>/*.md`-scoped pattern
   - a segment-aware case proving `*` does not cross `/` (a doc at
     `journals/2024-01.md` vs. a pattern `*.md` should not match it;
     `journals/*.md` should)
   - brace expansion (`{a,b}.md` matching two specific docs, not a
     third)
   - the comma+glob-chars-together case matching nothing (Node's
     ported quirk, Findings above)
   - confirm zero matches still produces an empty list (the existing
     `multi_get` CLI command's already-correct "No documents found." /
     exit 1 handles the rest, no new test needed there)
2. Run the existing full `tests/cli/test_multiget_shared.py`,
   `tests/cli/test_multi_get_command.py`, and
   `tests/mcp/test_server_multi_get.py` suites to confirm no
   regressions to the comma/docid path.
3. Remove the `_CLI_KNOWN_GAPS` entry and re-run
   `uv run pytest parity/test_structural.py -k multi_get` — expect the
   previously-`xfail` scenario to pass for real (no new capture run
   needed; the Node snapshot is already correct and already committed).
4. `just test-fast`, `just test-parity`, `just lint`.
5. Update `parity/README.md`, `COMMAND_STATUS.md`, and the roadmap doc.

## Out of scope

- Any glob syntax picomatch/wcmatch support that Node's own code and
  tests don't exercise (extglob, globstar `**`, negation) — not
  requested, not tested on the Node side, so there's nothing to port
  parity against.
- `--full-path` (Node has it, pyqmd's `multi-get`/`get` don't yet) and
  any other `multi-get`/`get` flag gap — unrelated to glob-matching.
  (`--format` is not one of these: pyqmd's `multi-get` already has it,
  independent of this work.) `--full-path` is now tracked separately as
  its own roadmap backlog item, added alongside this spec.
