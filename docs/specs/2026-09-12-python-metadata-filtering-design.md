# qmd Python/MLX rewrite — sub-project #7: metadata filtering — design

**Status:** Implemented (2026-09-12)
Date: 2026-09-12

## Purpose

Add document metadata to `qmd.store.Store`: documents opt in via a `qmd.metadata`
namespace in their YAML frontmatter, and every result-producing search surface
(`search`, `vsearch`, `query` on the CLI; the MCP `query` tool, over both stdio
and HTTP) can be constrained by a recursive filter AST (`and`/`or`/`not`,
`eq`/`ne`/`gt`/`gte`/`lt`/`lte`, `in`/`nin`/`all`, `exists`).

This closes an explicitly-tracked gap that surfaced repeatedly across
sub-projects #2, #3, and #4 without ever being scheduled — see
`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md` item #7.

This is a faithful port of the Node reference's three-module implementation:
`src/metadata.ts` (extraction), `src/metadata-store.ts` (schema/persistence),
`src/metadata-filter.ts` (filter AST parse + SQL compile), and their call
sites in `src/store.ts` and `src/mcp/server.ts`. The "less rework for future
parity" principle governing this whole rewrite favors matching the reference
exactly rather than inventing a different shape — every deviation below is
called out explicitly and justified.

## Scope

**In scope:**

- Frontmatter-based metadata extraction (`.md`/`.markdown`/`.mdx` only).
- Metadata storage: one row per document recording extraction state, plus one
  row per scalar value for indexed filtering.
- A recursive filter AST: parser (strict validation, defensive limits),
  compiler (parameterized SQL), and evaluator (via SQL `EXISTS`/`NOT EXISTS`
  subqueries — no in-Python evaluation).
- `--filter <json>` on the CLI's `search`, `vsearch`, and `query` commands.
- `filter` parameter on the MCP `query` tool (works identically over stdio and
  HTTP — sub-project #4 already built both transports on one shared tool
  implementation, so there is no transport-specific work here).
- Metadata extraction wired into `scan_and_register_collection` (`collection
add`).
- Real population of `HybridQueryResult.metadata` (already a field, always
  `{}` today — see sub-project #4's memory note).
- `query` output metadata display in the CLI's `json`, `md`, `xml`, and `cli`
  formats (an improvement over the Node reference, which only renders it in
  `json` — see "CLI metadata display" below), and in the MCP `query` tool's
  structured result output (matching the reference exactly there).

**Deferred / explicitly out of scope:**

- Non-frontmatter metadata sources. `extract_document_metadata`'s result
  shape stays source-agnostic (matching the reference's own stated design
  intent), but nothing beyond frontmatter is built.
- Metadata display in `search`/`vsearch` output. Filtering works identically
  for all three CLI commands; only `query`'s output renders metadata values.
  A future follow-up if wanted — sized in this spec's history as "bigger than
  query's case" because `SearchResult`/`search_fts`/`search_vec` have no
  post-fusion attachment point the way `query()` does.
- `csv`/`files` output formats never render metadata (matches the Node
  reference; these formats are deliberately terse).
- Re-extraction/staleness repair outside `collection add`. Metadata is
  (re-)extracted only when `collection add` runs, exactly matching how
  indexing itself already works today (`collection add` on an existing name
  still raises `IntegrityError` — no re-scan path exists yet). Staleness is
  an existing, already-documented gap (sub-project #3's parked "re-indexing
  is unreachable" finding) that this sub-project does not attempt to solve;
  full re-extraction naturally arrives with sub-project #8 (`update`
  command — see the roadmap doc), added alongside this spec specifically to
  give that recurring gap a real, numbered home instead of an implied one.
- Typed sub-queries (`searches` array) on the MCP `query` tool — already
  deferred from sub-project #4, unrelated to this work.
- A schema version bump. See "Schema" below.

## Naming

No new naming conventions. Reuses this project's existing terms: `Store`,
`collection`, `document`. The Node reference's `MetadataFilter`/
`MetadataCondition`/`MetadataFilterGroup`/`MetadataFilterNegation` type names
carry over as Python dataclass/type-alias names in `qmd/store/_metadata_filter.py`.

## File structure

New files in `qmd/store/`:

- **`_metadata.py`** — pure functions, no `self`: `extract_document_metadata(content,
path) -> MetadataExtractionResult`, frontmatter slicing, key/value
  normalization, `METADATA_LIMITS`. Port of `metadata.ts`. Zero SQL.
- **`_metadata_filter.py`** — pure functions/dataclasses, no `self`: the
  `MetadataFilter` type union, `parse_metadata_filter(input) -> MetadataFilter`
  (strict validation against `METADATA_FILTER_LIMITS`), `compile_metadata_filter(filter,
alias) -> CompiledMetadataFilter` (parameterized SQL: `EXISTS`/`NOT EXISTS`
  subqueries against `document_metadata_values`), `MetadataFilterError`. Port
  of `metadata-filter.ts`. These stay standalone functions, matching this
  codebase's existing pattern for `_rrf.py`/`_fts_query.py` — logic with zero
  `self` references doesn't belong on `Store`.

Modified files:

- **`qmd/store/_schema.py`** — `_create_schema_v1` gains the two new tables
  and three indexes (see "Schema" below). `CURRENT_SCHEMA_VERSION` stays `1`.
- **`qmd/store/store.py`** — new `Store` methods (persistence, unlike the
  reference's free functions — matching this codebase's established "`Store`
  holds its own logic" convention from sub-project #2):
  - `sync_document_metadata(document_id, content, path, *, only_if_stale=False)
-> MetadataExtractionResult | None`
  - `replace_document_metadata(document_id, extraction) -> None`
  - `get_metadata_by_filepath(filepaths) -> dict[str, dict]`
  - `count_documents_pending_metadata() -> int`

  `search_fts`, `search_vec`, `_retrieve_and_fuse`, and `query` all gain a
  `filter: MetadataFilter | None` parameter.

- **`qmd/store/_indexing.py`** — `scan_and_register_collection` calls
  `store.sync_document_metadata(...)` for every scanned file.
- **`qmd/store/_types.py`** — no new fields; `HybridQueryResult.metadata`
  already exists and starts getting real values.
- **`qmd/cli/commands/search.py`** — `--filter <json>` option on `search`,
  `vsearch`, `query`; a shared parse helper (new, small — likely lives beside
  the command module or in a new `qmd/cli/_metadata_filter.py`) mirrors
  `parseCliMetadataFilter`'s clean-exit behavior.
- **`qmd/cli/_types.py`** — `DisplayResult` gains `metadata: dict = field(default_factory=dict)`.
- **`qmd/cli/_output_search.py`** — `json`/`md`/`xml`/`cli` formatters render
  `metadata` when non-empty; `csv`/`files` untouched.
- **`qmd/mcp/server.py`** — the `query` tool's implementation gains a
  `filter: dict | None` parameter, validated the same way, returning an
  `isError=True` `CallToolResult` on failure (never a raised exception).

## Schema

The two tables below are added directly into `_create_schema_v1` (not a new
migration step). `CURRENT_SCHEMA_VERSION` stays `1`.

**Why no version bump:** this project has never been released, has exactly
one user (the person driving this rewrite), and has no real document
collection indexed yet. A `PRAGMA user_version`-driven migration exists to
carry _real, already-collected data_ forward across a schema change; there is
none to carry. Sub-project #2's migration scaffolding (`_MIGRATIONS` dict,
`migrate()`) stays in place and ready for the _first_ schema change that
lands after real data exists — this is a deliberate, reasoned exception to
"always version schema changes," not a precedent for skipping versioning
going forward. Practical consequence: anyone with a dev DB from an earlier
sub-project must delete `~/.cache/pyqmd/index.sqlite` before using this
feature, since `CREATE TABLE IF NOT EXISTS` inside `_create_schema_v1` never
re-runs against a database already at `user_version = 1`. No compatibility
code is built for this — it's a `rm` and a re-run of `collection add`.

Ported verbatim from `metadata-store.ts`'s `initializeMetadataSchema`
(field/column names, types, and `CHECK` constraints unchanged):

```sql
CREATE TABLE document_metadata (
    document_id INTEGER PRIMARY KEY,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    extraction_version INTEGER NOT NULL,
    extraction_error TEXT,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE document_metadata_values (
    document_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    value_type TEXT NOT NULL,
    text_value TEXT,
    number_value REAL,
    boolean_value INTEGER,
    PRIMARY KEY (document_id, key, ordinal),
    FOREIGN KEY (document_id) REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    CHECK (value_type IN ('string', 'number', 'boolean')),
    CHECK (
        (value_type = 'string' AND text_value IS NOT NULL AND number_value IS NULL AND boolean_value IS NULL)
        OR (value_type = 'number' AND number_value IS NOT NULL AND text_value IS NULL AND boolean_value IS NULL)
        OR (value_type = 'boolean' AND boolean_value IN (0, 1) AND text_value IS NULL AND number_value IS NULL)
    )
);

CREATE INDEX idx_metadata_text_lookup
    ON document_metadata_values(key, text_value, document_id) WHERE value_type = 'string';
CREATE INDEX idx_metadata_number_lookup
    ON document_metadata_values(key, number_value, document_id) WHERE value_type = 'number';
CREATE INDEX idx_metadata_boolean_lookup
    ON document_metadata_values(key, boolean_value, document_id) WHERE value_type = 'boolean';
```

`document_metadata` records extraction state per document — including
successful-but-empty extraction — so filtered search can distinguish
"extracted with no metadata," "not yet extracted," and "extraction failed."
`document_metadata_values` holds one indexed row per scalar value (arrays
expand to one row per element, `ordinal`-numbered) for filtering.

## Extraction

`extract_document_metadata(content, path) -> MetadataExtractionResult`
(ported from `metadata.ts`) reads `qmd:\n  metadata:\n    ...` from a
document's leading YAML frontmatter and normalizes it to
`DocumentMetadata = dict[str, MetadataValue]`, where `MetadataValue` is a
scalar (`str | int | float | bool`) or a flat, homogeneous array of one
scalar type. Never raises: a document without frontmatter, without the `qmd`
namespace, or with a non-frontmatter extension yields empty metadata with no
error; invalid frontmatter or invalid metadata yields empty metadata plus a
bounded (`METADATA_LIMITS.max_error_length`-truncated) extraction error
string. Extraction is all-or-nothing per document — one bad key throws
internally and the whole document's extraction is recorded as failed, so
stale partial metadata can never persist.

Limits (ported verbatim from `METADATA_LIMITS`): `max_frontmatter_bytes =
64*1024`, `max_keys = 64`, `max_key_bytes = 128`, `max_string_length = 1024`,
`max_array_length = 128`, `max_yaml_alias_count = 100`, `max_error_length =
200`.

**YAML alias-bomb defense — no direct library equivalent.** `pyyaml`
(already an indirect dependency today per `uv.lock`; this sub-project
promotes it to a direct one in `pyproject.toml`) has no built-in option
equivalent to the JS `yaml` package's `maxAliasCount` — checked directly
against the installed `pyyaml==6.0.3`. `max_frontmatter_bytes` alone is not
sufficient defense: a classic alias-expansion ("billion laughs") bomb
achieves exponential blowup from a small byte string well under 64KB.
Implementation needs a small custom `yaml.SafeLoader` subclass that counts
alias resolutions during composition and raises once
`max_yaml_alias_count` is exceeded (e.g. overriding `compose_node` to
increment a counter on alias nodes) — a well-known, small (~15-20 line)
pattern, not a new dependency. This detail is called out here so the
implementation plan doesn't waste a task assuming a keyword argument that
doesn't exist.

**Wiring:** `scan_and_register_collection` (`qmd/store/_indexing.py`) calls
`store.sync_document_metadata(document_id, content, relative_posix,
only_if_stale=...)` for every scanned file, mirroring `store.ts:1744-1745`
exactly:

- New or content-changed document → `only_if_stale=False` (always
  re-extract).
- Unchanged document → `only_if_stale=True` (skip re-extraction if a
  current-version extraction row already exists; backfills missing/stale
  extraction cheaply otherwise — relevant today mainly for a document whose
  extraction previously failed transiently, or after this feature's own
  rollout against a freshly-reset dev DB).

`sync_document_metadata` calls `extract_document_metadata` then
`replace_document_metadata`, which atomically (one SQLite transaction)
upserts the `document_metadata` row and replaces all
`document_metadata_values` rows for that document — exactly matching
`metadata-store.ts`'s `replaceDocumentMetadata`.

## Filter AST and SQL compilation

Ported verbatim from `metadata-filter.ts` into `qmd/store/_metadata_filter.py`:

```python
MetadataScalar = str | int | float | bool
MetadataFilter = MetadataFilterGroup | MetadataFilterNegation | MetadataCondition
# MetadataFilterGroup:     {"operator": "and" | "or", "operands": [...]}
# MetadataFilterNegation:  {"operator": "not", "operand": {...}}
# MetadataCondition:       {"key": str, "operator": ..., "value": ...}
#   eq/ne            -> value: MetadataScalar
#   gt/gte/lt/lte     -> value: str | int | float (not bool)
#   in/nin/all       -> value: homogeneous list of one scalar type
#   exists           -> value: bool
```

`parse_metadata_filter(input: object) -> MetadataFilter` strictly validates
an untrusted value: rejects unknown operators, unknown properties,
operator-incompatible value types, and anything exceeding
`METADATA_FILTER_LIMITS` (`max_depth=16`, `max_nodes=256`,
`max_group_operands=32`, `max_membership_values=64`, plus the same
`max_key_bytes`/`max_string_length` as extraction). Membership arrays
(`in`/`nin`/`all`) are canonicalized by de-duplicating while preserving
order. Raises `MetadataFilterError(path, message)` — the JSON path of the
failing node — never a generic exception, so callers can surface an
actionable location.

`compile_metadata_filter(filter, alias) -> CompiledMetadataFilter` compiles a
validated filter into one parameterized SQL boolean expression correlated
against a documents-table alias, e.g. `d`. Every user-supplied key and value
is a bound parameter — never interpolated into SQL. Each condition compiles
to an `EXISTS (SELECT 1 FROM document_metadata_values mv WHERE
mv.document_id = {alias}.id AND ...)` (or `NOT EXISTS`), following the exact
per-operator logic in `metadata-filter.ts:284-356` (including `ne`'s
"key must have a same-type value present, and none may equal the operand"
semantics, and `nin`'s equivalent "present but not a member" semantics —
both distinguish a genuinely-absent key from "no match," which matters for
correct negation).

**Store-layer integration:** `search_fts`/`search_vec` each gain the guard
`AND dm.extraction_version = 1 AND dm.extraction_error IS NULL AND
{compiled.sql}` alongside a `LEFT JOIN document_metadata dm ON dm.document_id
= d.id`, appended only when a `filter` is given — ported from
`store.ts:4153-4159`/`:4300`/`:4360`. This means an unextracted or
extraction-failed document never satisfies any filter, not even `exists:
false` — a document must have gone through extraction to participate in
filtered search at all. `_retrieve_and_fuse`/`query` thread `filter` through
to both.

## CLI surface

`--filter <json>` added to `search`, `vsearch`, and `query` (all three,
matching the Node reference's help text: `--filter <json> - Metadata filter
(recursive JSON AST; search/vsearch/query)`). A shared parse helper mirrors
`parseCliMetadataFilter`:

1. `json.loads` the raw string. On failure: print `Invalid --filter JSON:
<error>` plus an example (`--filter
'{"key":"status","operator":"eq","value":"published"}'`), exit non-zero.
2. `parse_metadata_filter(...)` the parsed value. On a `MetadataFilterError`:
   print its message, exit non-zero.

Both paths go through this project's existing clean-exit convention (the
`run_or_exit` pattern already used by every other command) — never a raw
Python traceback.

No `--filter` given → `filter=None` threaded through unchanged; behavior is
identical to today (regression-tested).

## MCP surface

The `query` tool's implementation function gains `filter: dict | None`.
Validation reuses the exact same `parse_metadata_filter` (no separate MCP
grammar) against the raw dict the SDK hands the tool — mirroring
`validateFilterArgument`'s contract exactly:

- `filter is None` → no filtering, no error.
- Invalid (fails `parse_metadata_filter`) → `CallToolResult(is_error=True,
content=[TextContent(type="text", text=f"Error: {message}")])`. Never a
  raised exception — an MCP client must see a clean tool-level error, not a
  transport-level fault.
- Valid → passed to `store.query(..., filter=parsed)`.

Because sub-project #4 already implemented both stdio and Streamable HTTP
transports on one shared `_query_impl` function, this validation and
behavior is identical on both transports with zero transport-specific code.

**MCP result formatting (corrected during this spec's own self-review):**
the Node MCP `query` tool already includes `metadata` in each result's
structured output — `...(Object.keys(r.metadata).length > 0 ? { metadata:
r.metadata } : {})` (`src/mcp/server.ts:421`), the same non-empty-only guard
used by the CLI's `json` formatter. The first draft of this spec scoped
metadata _display_ to the CLI only and missed this — an oversight caught
before implementation, not a deliberate deferral, since the reference
already does it and this rewrite's "less rework for parity" principle
applies directly. `qmd/mcp/_formatting.py::format_query_result` gains the
same non-empty-only `metadata` key, alongside the existing
`docid`/`file`/`title`/`score`/`context`/`line`/`snippet` fields, on both
transports (again free — one shared formatter).

## Result metadata attachment and CLI display

`HybridQueryResult.metadata` (already a field on the dataclass, always `{}`
prior to this sub-project — see sub-project #4's memory note) gets a real
batch `get_metadata_by_filepath` call inside `query()`, run once after RRF
fusion, best-chunk selection, and reranking are complete — matching
`store.ts`'s `attachResultMetadata` timing exactly (`store.ts:5510-5511`:
"Runs after RRF/reranking so metadata is never duplicated through the
intermediate ranked lists").

**CLI display (an intentional improvement over the Node reference):** the
Node CLI threads `metadata` through `search`/`vsearch`/`query`'s output rows,
but only the `json` format ever renders it — `files`/`cli`/`md`/`xml` never
reference it at all (verified directly against `src/cli/qmd.ts` and
`src/cli/formatter.ts`). This rewrite does better for `query` specifically,
since the data is already being attached as part of this sub-project:

- `DisplayResult` (`qmd/cli/_types.py`) gains `metadata: dict =
field(default_factory=dict)`.
- `search.py`'s `_hybrid_result_to_display` (the `query` command's mapper)
  populates it; `_search_result_to_display` (`search`/`vsearch`'s mapper)
  does not — those two commands keep today's behavior exactly, matching the
  earlier decision to defer their metadata display.
- In `_output_search.py`: `json` (matching the reference's own
  non-empty-only guard: `Object.keys(row.metadata).length > 0`), `md`, `xml`,
  and `cli` formatters render `metadata` when non-empty, each following its
  existing idiom (e.g. `cli`'s dim-colored auxiliary-line style used for
  `Context:`). `csv` and `files` stay untouched, matching the reference —
  both formats are deliberately terse and metadata as an arbitrary
  JSON-shaped dict doesn't fit either's flat-row/single-line contract.
- Exact per-format rendering (e.g. compact-JSON-in-one-line vs. one line per
  key for `cli`; child elements vs. an attribute for `xml`) is decided with
  concrete code and tests in the implementation plan, not fixed further in
  prose here — none of these choices affect any other part of the design.

## Error handling summary

| Layer    | Bad JSON                      | Bad AST                         | Unextracted doc + filter         |
| -------- | ----------------------------- | ------------------------------- | -------------------------------- |
| CLI      | clean exit, message + example | clean exit, validator's message | silently excluded (not an error) |
| MCP tool | n/a (already a dict)          | `isError=True` `CallToolResult` | silently excluded (not an error) |
| `Store`  | n/a                           | raises `MetadataFilterError`    | `WHERE` guard excludes the row   |

`MetadataFilterError(path, message)` is the one exception type
`parse_metadata_filter`/`compile_metadata_filter` ever raise; both the CLI
and MCP layers catch it and translate to their surface's native error shape.

## Testing strategy

Following this project's established pattern throughout sub-projects #1-#4:
a real in-memory `Store` (`:memory:`) with fakes for `qmd.llm`, never mocked
SQL.

- **Extraction (`_metadata.py`):** no frontmatter; frontmatter without a
  `qmd` namespace; valid metadata (scalars, arrays, mixed types); each limit
  individually (`max_keys`, `max_key_bytes`, `max_string_length`,
  `max_array_length`, oversized frontmatter, YAML alias bomb); invalid YAML;
  non-frontmatter extensions (`.txt`); `null` values (rejected); empty arrays
  (rejected); mixed-type arrays (rejected); nested arrays (rejected); control
  characters in keys (rejected); a `__proto__`-shaped key (must round-trip
  like any other key — Python has no prototype-pollution equivalent, but the
  key must not be silently dropped or mishandled); UTF-8 BOM and CRLF
  handling in frontmatter detection.
- **Filter parse/validate (`_metadata_filter.py`):** every operator accepted
  with valid shapes; every rejection path (unknown operator, unknown
  property, wrong value type per operator, empty/oversized
  `operands`/membership arrays, depth limit, node-count limit,
  non-homogeneous membership arrays); membership-value dedup while preserving
  order.
- **Filter SQL compile + execute:** seed a real in-memory `Store` with
  documents carrying varied metadata (including some with no metadata, some
  with `extraction_error` set), assert each operator's actual query results
  end-to-end, plus the "unextracted document never matches, even `exists:
false`" guarantee, plus `and`/`or`/`not` nesting, plus parameter binding
  safety (a metadata value containing SQL-special characters must not affect
  query structure).
- **Integration:** `search_fts`/`search_vec`/`query` with `filter` combined
  with `collection` scoping; all three `scan_and_register_collection`
  extraction paths (new doc, changed doc, unchanged doc with
  `only_if_stale`); `count_documents_pending_metadata`.
- **CLI:** valid `--filter` on all three commands; malformed JSON exits
  cleanly with the example message; a filter AST that fails validation exits
  cleanly with the validator's message; no `--filter` behaves exactly as
  before (regression guard); `query`'s `metadata` rendering across
  `json`/`md`/`xml`/`cli` (non-empty and empty cases) and its absence from
  `csv`/`files`.
- **MCP:** extends `tests/mcp/test_real_dispatch.py`'s real-dispatch
  pattern: `query` tool with a valid `filter` dict, an invalid one
  (`isError=True`, no exception, verified through genuine SDK dispatch, not
  a direct `_query_impl` call), and one thin HTTP-transport assertion
  confirming the same behavior holds there too (not a duplicate suite —
  sub-project #4 already proved the two transports share one code path).
- **Result attachment:** `HybridQueryResult.metadata` reflects real stored
  values after a real fusion+rerank run, not `{}`.
