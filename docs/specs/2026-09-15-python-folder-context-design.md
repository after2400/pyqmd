# pyqmd folder-context feature — design

**Status:** Implemented (2026-09-15)

Not a numbered sub-project on its own — a gap discovered while comparing
pyqmd's real-world output against Node qmd side by side on a live
collection. Sub-project #3 ("CLI command
surface (collection/context management...)") claimed context management
was done; in fact only the DB column (`collections.context`) exists —
there has never been a CLI command to write it, and nothing reads it back
into `get`/`status`/search output. This spec closes that actual gap.

## Problem

Node qmd has a folder-context system: `qmd context add/list/rm` stores
free-text descriptions keyed by path-prefix per collection (root context,
per-subfolder context, etc.), resolved hierarchically and attached to
`get`, `status`, and every search/query result **as display-only
metadata** — confirmed by reading `store.ts`: `context: getContextForFile(...)`
is assigned to result rows _after_ FTS/vector scoring and reranking are
already final, never fed into the ranking itself. This matches the user's
own observation comparing pyqmd against Node: same documents found, just
missing the human-readable annotation.

pyqmd already has the column (`collections.context TEXT`, `_schema.py`)
and, more surprisingly, most of the _consumer_ plumbing: `DisplayResult`
and `HybridQueryResult` both already have a `context: str | None` field,
and `qmd/cli/_output_search.py` (all of json/csv/md/xml/cli formats) and
the MCP server's `_formatting.py` already render `.context` when present.
None of it has ever had real data — `HybridQueryResult(..., context=None)`
is hardcoded in `store.py`'s `query()`, `SearchResult` (used by
`search`/`vsearch`) has no `context` field at all yet, and `get`/`status`
never call anything that would resolve one. No CLI command has ever
written to the `context` column either — `collection add` has no
`--context` flag, so it is always `NULL` in every real pyqmd database
today.

## Reference implementation

Node's `src/collections.ts` (`addContext`/`removeContext`/
`listAllContexts`, YAML-config-backed — used by the `context` CLI
subcommand's error/detail messages) and `src/store.ts`
(`getContextForPath`/`getContextForFile`/`getStoreContexts`,
DB-synced-from-YAML — used by the actual runtime `get`/search code paths).
The CLI switch case lives at `src/cli/qmd.ts:4415-4490` (dispatch),
`:1058-1122` (`contextAdd`), `:1124-1150` (`contextList`),
`:1152-1187+` (`contextRemove`).

pyqmd has no project-local/checked-in config and is SQLite-only per this
repo's `CLAUDE.md` — there is no YAML side to sync, so pyqmd implements
**one** resolution path (matching `store.ts`'s DB-side semantics, which is
what actually appears in Node's `get`/status/search output) rather than
porting Node's parallel YAML-side `findContextForPath` (a different,
"most-specific-prefix-only" algorithm that is not what's user-visible in
the CLI output this spec is trying to match — confirmed by working
backward from the exact two-paragraph "Folder Context: ..." text in
Node's real output, which is a hierarchical join, not a single
most-specific match).

## Decisions made during brainstorming (2026-09-15)

1. **Storage model**: reinterpret the existing, always-`NULL`,
   never-written `collections.context` column as a JSON-encoded
   `{path_prefix: text}` map — exactly what Node's own `context` column
   holds internally (also JSON-in-TEXT, not a separate table). Zero schema
   migration, zero data-migration risk (nothing has ever written a
   non-JSON value there). Considered and rejected: a dedicated
   `collection_contexts` table (more relational, but buys nothing since
   contexts are never filtered/joined by SQL — just walked as a handful of
   prefix strings per lookup — and costs a schema migration); a YAML
   sidecar config mirroring Node's actual source of truth (rejected
   outright — contradicts `CLAUDE.md`'s SQLite-only architecture, where
   `Store` is the sole thing that touches `conn`).
2. **Global (`/`-wide) context**: out of scope for this pass. Node's
   `context add /` sets a context applied to every collection via a
   separate `store_config` key-value table pyqmd doesn't have. The
   side-by-side comparison only exercised per-collection contexts; add
   global context later as a small, separate follow-up if actually wanted.
3. **Resolution semantics**: `get_context_for_path` collects **every**
   matching path-prefix context, sorts general→specific (root first, then
   more specific ones), and joins with `"\n\n"` — matching
   `store.ts`'s `getContextForPath` (store.ts:3521-3559) exactly. This is
   deliberately not Node's _other_ `findContextForPath`
   (collections.ts:484-516, "most specific match wins, no join"), which is
   YAML-side code not reachable from the actual CLI output paths this spec
   targets.
4. **CLI addressing**: `context add`/`context remove` accept either a
   `qmd://collection/path` URI or a filesystem path. A filesystem path is
   resolved relative to `$PWD` (`.`/`./` → cwd, `~/` expands, relative
   paths resolve against cwd), then matched against each collection's
   stored `path` column by longest-prefix match
   (`detect_collection_for_path`) — full parity with Node's cwd-relative
   ergonomics (`src/cli/qmd.ts:1071-1079`, `detectCollectionFromPath`).
   Explicitly **not** ported: Node's single-arg shorthand where omitting
   the path argument entirely infers "context text only, path defaults to
   cwd" from argument count (`cli.args.length` sniffing) — Typer's
   positional-argument model doesn't need that ambiguity; `pyqmd context
add . "text"` already covers the same "current directory" case
   explicitly. Also not ported: Node's `/`-means-global special case (no
   global context, decision 2) and the `rm` alias (pyqmd's `collection`
   command group already drops Node's `rm`/`mv` aliases in favor of
   `remove`/`rename` only — this follows the same established
   convention).
5. **Path normalization for `detect_collection_for_path`**: realpath-
   normalize both the candidate filesystem path and each collection's
   stored `path` value _only inside this comparison function_ —
   `collection add` itself keeps storing whatever raw path string the user
   typed (unlike Node, which realpaths at add-time). Changing how
   `collection add` stores paths is a separate, pre-existing,
   unrelated gap and out of scope here.
6. **Output style**: plain text, no ANSI color, matching `status.py`'s and
   `collection.py`'s existing style. Checked whether to port Node's color
   scheme for the specific lines this feature touches, and found Node
   itself isn't consistent: `qmd get`'s header/`Folder Context:` line is
   plain (`src/cli/qmd.ts:1352-1356`, confirmed against source, not just
   the pasted transcript — no `c.*` color codes there at all), while `qmd
context add/list/rm` and `status`'s `Contexts:` section do use color.
   Porting a half-colored reference piecemeal, one feature at a time,
   would leave pyqmd with the same inconsistency rather than a considered
   scheme. Decision: keep this feature plain like the rest of pyqmd's CLI
   today, and design one consistent (and better than Node's) color scheme
   across the whole CLI as its own later pass — see "Out of scope" below.
   `✓`-prefixed confirmation lines follow the existing `collection
update-cmd` precedent (`✓ Set update command for '<name>': <cmd>` /
   `✓ Cleared update command for '<name>'`).

## Design

### Storage & Store API (`qmd/store/store.py`)

New methods, direct semantic ports of Node's `store.ts` functions, all
operating on the existing `collections.context` column parsed as JSON:

```python
def add_context(self, collection: str, path_prefix: str, text: str) -> bool:
    """Add or overwrite the context for one path prefix within a
    collection. Returns False if the collection doesn't exist."""

def remove_context(self, collection: str, path_prefix: str) -> bool:
    """Remove one path prefix's context. Returns False if the collection
    doesn't exist, or that exact prefix has no context set. Clears the
    column back to NULL once the map becomes empty (matches Node's
    removeStoreContext, store.ts:1425-1437)."""

def list_all_contexts(self) -> list[dict]:
    """[{"collection": str, "path": str, "context": str}, ...] across every
    collection that has any context set, ordered by collection name (SQL
    ORDER BY) then by insertion order within each collection's JSON map
    (Python's json.loads preserves key order, matching JS object insertion
    order — same as Node's getStoreContexts, store.ts:1356-1375)."""

def get_context_for_path(self, collection: str, path: str) -> str | None:
    """Hierarchical join: every stored path-prefix whose normalized form
    (leading '/') is a prefix of the normalized `path`, sorted shortest
    (most general) to longest (most specific) prefix, joined with '\\n\\n'.
    None if the collection doesn't exist, has no context map, or nothing
    matches. Matches store.ts's getContextForPath (store.ts:3521-3559)."""

def get_context_for_file(self, filepath: str) -> str | None:
    """Accepts 'qmd://collection/path' or bare 'collection/path' (same
    strip-scheme-then-split-on-first-slash shape already used inline by
    _docid_for_file and find_document_by_identifier); delegates to
    get_context_for_path. None on any parse failure (empty path after the
    collection name resolves the collection's root context, path_prefix
    "")."""

def detect_collection_for_path(self, fs_path: str) -> tuple[str, str] | None:
    """Realpath-normalize fs_path and every collection's stored `path`,
    then longest-prefix match. Returns (collection_name, relative_path)
    — relative_path is "" when fs_path IS the collection root — or None
    if fs_path isn't under any collection's indexed directory. Matches
    Node's detectCollectionFromPath (src/cli/qmd.ts, referenced from
    contextAdd/contextRemove)."""
```

`SearchResult` (`qmd/store/_types.py`) gains a `context: str | None = None`
field (it currently has none — `HybridQueryResult` already has one, unused
until now).

Three existing call sites get wired up:

- `search_fts` (store.py ~849) and `search_vec` (store.py ~1036): add
  `context=self.get_context_for_file(row["filepath"])` to each
  `SearchResult(...)` construction — mirrors Node's `getContextForFile`
  call inside both its `searchFts`/`searchVec` (store.ts:4183, :4395).
- `query`'s `HybridQueryResult(...)` construction (store.py:1234): replace
  the hardcoded `context=None` with
  `context=self.get_context_for_file(ranked.file)`.

### CLI command surface (`qmd/cli/commands/context.py`, new)

Mounted as a named sub-group in `app.py`, exactly like `collection`:
`app.add_typer(context.app, name="context")` (not the flat/no-op-callback
pattern used by `embed`/`status`/`update`/`mcp`, since this module has
real subcommands from the start).

```
pyqmd context add <path> <text...>
pyqmd context list
pyqmd context remove <path>
```

```python
@app.command("add")
def add(
    path: str = typer.Argument(
        ..., help="qmd://collection/path, or a filesystem path (., ~/, relative, or absolute) inside an indexed collection."
    ),
    text: list[str] = typer.Argument(..., help="Context description."),
) -> None:
    """Add or update the context for a path prefix."""
    # text is variadic + space-joined -- same convention as
    # `collection update-cmd`'s command argument.

@app.command("list")
def list_contexts() -> None: ...

@app.command("remove")
def remove(path: str = typer.Argument(...)) -> None: ...
```

Path resolution (shared by `add` and `remove`, a small local helper —
not a new shared module, matching how `find_document_by_identifier`
inlines its own qmd:// splitting rather than factoring one out):

1. Strip a `qmd://` scheme if present → split on the first `/` into
   `(collection, path_prefix)` (path_prefix `""` = collection root).
   Verify the collection exists (`store.get_collection`); error
   `Collection not found: <name>` (exit 1) if not.
2. Otherwise, treat as a filesystem path: `.`/`./` → `os.getcwd()`,
   `~/` → `os.path.expanduser`, other relative paths →
   `os.path.abspath` against cwd. Call
   `store.detect_collection_for_path(fs_path)`; on `None`, error
   `Path is not in any indexed collection: <fs_path>` +
   `Run 'pyqmd status' to see indexed collections` (exit 1).

`add` output: `✓ Added context for: qmd://<collection>/<path>` (or
`qmd://<collection>/ (collection root)` when `path_prefix` is `""`),
then `Context: <text>`.

`remove` output: `✓ Removed context for: qmd://<collection>/<path>` on
success; `No context found for: <path>` (exit 1) when `remove_context`
returns `False` for a collection that DOES exist (existing-but-unset vs.
collection-not-found get the two distinct messages above).

`list` output — grouped by collection, matching Node's `contextList`
(qmd.ts:1124-1150) minus color:

```
reference
  / (root)
    Stable reference notes: core topics and ongoing projects
  People
    Notes about people and contacts
```

(No leading `/` on the non-root path — `contextList` prints the stored
prefix bare, e.g. `People` from `parseVirtualPath`'s own `path: match[2]`,
store.ts:772. This is a different convention from `status`'s own
per-collection preview section below, which _does_ add a leading `/`
(qmd.ts:641) — two genuinely different Node code paths, not a typo; each
is matched exactly as its own reference does.)

Empty state: `No contexts configured. Use 'pyqmd context add' to add one.`

### `get` command (`qmd/cli/commands/documents.py`)

After resolving `doc` (which already carries `collection` and `path`
columns from `find_document_by_identifier`'s `d.*` select), before
printing `---`:

```python
context = store.get_context_for_path(doc["collection"], doc["path"])
if context:
    typer.echo(f"Folder Context: {context}")
```

Matches Node's `get` output exactly (`src/cli/qmd.ts:1352-1356`) — this is
the specific missing line from the side-by-side comparison.

### `status` command (`qmd/cli/commands/status.py`)

Call `store.list_all_contexts()` once, group into a
`dict[str, list[dict]]` keyed by collection name (plain Python grouping,
mirroring Node's `contextsByCollection` map, qmd.ts:587-599). Remove the
existing `if c["context"]: typer.echo(f"    Context: {c['context']}")`
branch (it read the old flat-string shape, which never had real data
since nothing ever wrote one), replacing it with:

```
  <name>  <path>
    Contexts: <N>
      /: <first 60 chars of root context>...
      /People: <first 60 chars>...
```

matching Node's per-collection context preview (qmd.ts:637-647) — path
display normalizes `""`/`"/"` to `/`, everything else gets a leading `/`;
text over 60 chars gets `...`-truncated.

## Out of scope

- Global (`/`-wide) context and its `store_config` table (decision 2).
- Realpath-normalizing `collection add`'s stored path column (decision 5)
  — a separate, pre-existing gap unrelated to this feature.
- A single "formatting" follow-up subtask, bundling three things that
  don't belong in this spec: `status`'s other missing sections
  (per-collection `Pattern:`/`Files:`, `Examples`, `Models`, `Tips`),
  `ls`'s missing size/date/`qmd://` formatting, and a considered,
  consistent ANSI color scheme across the whole CLI (`status`, `get`,
  `collection`, `context`, `ls`) — designed once, deliberately, rather
  than ported piecemeal from Node's own inconsistent coloring (decision
  6). Bounded work; doesn't need its own architectural spec.
- MCP-specific changes: `_formatting.py` already renders `.context` from
  whatever `HybridQueryResult`/`SearchResult` carries — no MCP server code
  needs to change once the Store-level wiring above lands.

## Testing approach

Store-level (new `tests/store/test_context.py`, following the existing
per-feature test-file convention, e.g. `test_orphan_cleanup.py`):

- `add_context`/`remove_context`: set on an empty map, overwrite an
  existing prefix, remove the only entry (column becomes `NULL`), remove
  one of several (map survives with the rest), `False` return for an
  unknown collection (both add and remove) and for removing an unset
  prefix on a real collection.
- `list_all_contexts`: empty index → `[]`; multiple collections/prefixes →
  correct grouping and field values; a collection with no context set is
  simply absent from the list (not an empty-string entry).
- `get_context_for_path`: no match → `None`; single root match; multiple
  matching prefixes joined general→specific with `"\n\n"`; a prefix that
  is NOT an ancestor of the queried path is correctly excluded (e.g.
  `/Topics` context must not leak into a `/People/wife.md` lookup).
- `get_context_for_file`: `qmd://` and bare `collection/path` forms both
  resolve; unknown collection → `None`.
- `detect_collection_for_path`: exact collection root, a nested
  subdirectory, an unrelated path (→ `None`), and a trailing-slash
  variant of the collection's stored path.

CLI-level (new `tests/cli/test_context_command.py`, `CliRunner` +
in-memory `Store`, matching `test_collection_command.py`'s pattern):

- `context add qmd://<name>/<path> some text here`: stored (verify via
  `store.get_collection`/`list_all_contexts`), success message format,
  root-path message variant (`(collection root)`).
- `context add <fs-path> text` with a monkeypatched cwd: resolves via
  `detect_collection_for_path`; error message when the path isn't inside
  any collection.
- `context add` targeting an unknown `qmd://` collection: clean error, no
  partial write.
- `context list`: empty-state message; populated multi-collection,
  multi-prefix output grouping.
- `context remove`: success message; "no context found" vs. "collection
  not found" get their distinct error messages/exit codes.
- `get` on a document under a collection with root + subfolder contexts
  set: `Folder Context: <joined text>` line appears in the right place
  (before `---`), correctly joined; a document in a collection with no
  context configured: line is absent entirely (no `Folder Context:` with
  empty text).
- `status`: a collection with contexts shows `Contexts: N` and truncated
  previews; a collection with none shows neither line (regression check
  against the removed single-line `Context: ...` branch).
- `search`/`vsearch`/`query`: at least one test per command confirming a
  populated `context` value round-trips into `--format json` output
  (`"context": "..."` present) — not re-testing every formatter, since
  `_output_search.py`'s per-format rendering of an existing `context`
  field already has coverage; this only proves the value now actually
  gets there.

Parity-suite note: `context add/list/remove` is new CLI surface not
currently in `parity/scenarios/cli_scenarios.py`. Adding parity scenarios
for it is out of scope for this spec (matching the precedent set by the
`update` command spec) — the parity suite's own scope was closed out
separately. The feature inventory
(`2026-09-12-python-node-parity-suite-design.md`) should still gain rows
for `context add/list/remove` and get its `get`/`status` rows' notes
de-staled once this lands, so the inventory doesn't silently lie.
