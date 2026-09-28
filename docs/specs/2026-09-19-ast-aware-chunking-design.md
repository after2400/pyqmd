# AST-aware chunking (roadmap #5) — design

**Status:** Implemented (2026-09-20)

Sub-project #5 from `2026-09-10-python-mlx-rewrite-roadmap.md`,
the last open numbered item after #11/#12 shipped.

## Problem

`src/qmd/store/_chunking.py` splits long documents into ~900-token chunks
using regex-scored break points: markdown headings, code fences, blank
lines, list markers. That works for prose but has no signal for source
code — a `.py`/`.ts`/`.go`/`.rs` file has none of those markers, so the
chunker falls back to its weakest pattern (a bare newline) and can split a
chunk boundary in the middle of a function body. A search hit against that
function then only surfaces half of it.

Node's `qmd` already solved this (`src/ast.ts` + wiring in `src/store.ts`/
`src/cli/qmd.ts`): for supported code file types, parse the real syntax
tree with tree-sitter, extract break points at function/class/etc.
boundaries, merge them with the existing regex break points, and let the
existing scored-cutoff algorithm prefer them when close to the target
chunk size. It's opt-in via `--chunk-strategy auto` (default stays
`regex`) and degrades silently to regex-only on any parse or grammar
failure. This spec ports that behavior to pyqmd.

**Out of scope:** symbol extraction (Node's `extractSymbols` is itself a
stubbed Phase 2 placeholder — nothing to port yet), any change to the
regex chunker's own behavior, MCP tool changes (Node's MCP server doesn't
expose `chunkStrategy` either — verified, no hits in `src/mcp*`).

## Language & binding choice

Match Node's 6 languages: TypeScript, TSX, JavaScript, Python, Go, Rust.
Same `EXTENSION_MAP` (`.ts`/`.mts`/`.cts`→typescript, `.tsx`/`.jsx`→tsx,
`.js`/`.mjs`/`.cjs`→javascript, `.py`→python, `.go`→go, `.rs`→rust).

Binding: `py-tree-sitter` (the `tree_sitter` package) plus one PyPI grammar
package per language (`tree-sitter-python`, `tree-sitter-typescript`
covers both `.ts`/`.tsx`, `tree-sitter-javascript`, `tree-sitter-go`,
`tree-sitter-rust`), added as regular (not optional) dependencies in
`pyproject.toml`. Verified against the current published versions
(`tree-sitter==0.26.0`, `tree-sitter-python==0.25.0`): construction is

```python
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python as tspython

lang = Language(tspython.language())
parser = Parser(lang)
tree = parser.parse(content_bytes)
query = Query(lang, query_source)
captures = QueryCursor(query).captures(tree.root_node)  # dict[str, list[Node]]
```

This is meaningfully different from Node's `web-tree-sitter` (no WASM
runtime, no async init, `captures()` groups by capture name into a dict
instead of returning a flat list) — the port adapts to this shape rather
than mimicking Node's exact function signatures.

**Byte vs. char offsets:** tree-sitter parses bytes and reports
`node.start_byte` as a UTF-8 byte offset, not a character index.
`_chunking.py`'s `pos` values are character offsets into the Python
`str` (consistent with the file's existing UTF-16-surrogate note for the
regex path). For non-ASCII source files these differ. Conversion: encode
`content` to UTF-8 once, collect all needed byte offsets, sort them, and
walk the encoded bytes a single time decoding incrementally to build a
byte→char lookup — O(n) total, not O(n) per break point. Pure-ASCII
content (the common case) short-circuits to an identity mapping.

## Module: `src/qmd/store/_ast.py`

Ported from `ast.ts`, adapted to the sync `py-tree-sitter` API (no
`asyncio` needed anywhere in this module or its callers — parsing is
synchronous and fast enough that pyqmd's existing sync `chunk_document`
call sites don't need to become async):

- `SupportedLanguage` — a `Literal`/enum of the 6 languages.
- `detect_language(filepath: str) -> SupportedLanguage | None` — extension
  lookup, same map as Node's.
- `LANGUAGE_QUERIES: dict[SupportedLanguage, str]` — the same
  S-expression query source per language, ported verbatim from `ast.ts`
  (e.g. Python: `(class_definition) @class`, `(function_definition) @func`,
  `(decorated_definition) @decorated`, `(import_statement) @import`,
  `(import_from_statement) @import`).
- `SCORE_MAP: dict[str, int]` — identical scores to Node's
  (class/iface/struct/trait/impl/mod=100, export/func/method/decorated=90,
  type/enum=80, import=60), so a ported break point competes with the
  regex scale (h1=100 … newline=1) exactly the way Node's does.
- Grammar loading is a straight `import tree_sitter_python as ...`-style
  per-language import inside a `GRAMMAR_LOADERS` dict, cached in a
  module-level `dict[SupportedLanguage, Language]`; a failed import (e.g.
  the module lacks a Query bridge for some future grammar version) is
  caught once and remembered so it isn't retried every call — mirrors
  Node's `failedLanguages`/`grammarLoadErrors`.
- `get_ast_break_points(content: str, filepath: str) -> list[BreakPoint]`
  — resolves language, loads the grammar (returns `[]` on any failure,
  never raises), parses, runs the language's query, converts byte offsets
  to char offsets, dedupes by position keeping the highest score (a
  `class` and its wrapping `decorated_definition` can start at the same
  position — keep the higher-scoring capture, mirroring Node's
  export-wrapper dedup), returns sorted by position. A parse exception is
  caught and swallowed, falling back to `[]` — silently, with no warning
  printed. This is a deliberate divergence from Node's `console.warn` on
  every failure: pyqmd's `qmd/store/` layer has no user-facing-output
  convention today (confirmed — every `typer.echo(..., err=True)` call in
  the codebase lives under `qmd/cli/`, never `qmd/store/`), and the store
  layer doesn't depend on `typer`. Per-language grammar availability is
  already surfaced through `get_ast_status()` / `status`'s new section; a
  single file's rare parse failure just falls back to regex chunking for
  that file without additional plumbing.
- `merge_break_points(regex_points, ast_points) -> list[BreakPoint]` — a
  straightforward merge-and-sort by position (AST points are additional
  candidates, not replacements; if both land on the same position, keep
  the higher score — same dedup rule as within a single source).
- `get_ast_status() -> ASTStatus` — per-language availability (dataclass
  with `available: bool` and `languages: list[LanguageStatus]`, each
  carrying `language`, `available`, and an optional `error` string), used
  by `status`.

## `_chunking.py` changes

`chunk_document` gains two new parameters:

```python
def chunk_document(
    content: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS,
    filepath: str | None = None,
    chunk_strategy: Literal["regex", "auto"] = "regex",
) -> list[tuple[str, int]]:
```

When `chunk_strategy == "auto"` and `filepath` is given: compute
`scan_break_points(content)` as today, then also call
`get_ast_break_points(content, filepath)`; if that returns anything,
merge via `merge_break_points`. Everything downstream (`find_best_cutoff`,
the chunking loop, code-fence handling) is unchanged — AST break points
are just higher-quality candidates fed into the same scoring pass. When
`chunk_strategy == "regex"` (the default) or `filepath` is `None`,
behavior is byte-for-byte identical to today — no import of `_ast.py`
happens on that path, so users who never pass `--chunk-strategy auto`
never pay the tree-sitter import/parse cost.

## Store-layer wiring

Two call sites use `chunk_document`, and both need a filepath threaded in
— this is the one piece of real plumbing beyond a straight port, since
pyqmd's schema doesn't already carry a "the" filepath for chunking
purposes:

**1. `Store.index_content` (embedding/indexing path, `store.py:835`).**
Content is deduplicated by hash — `get_indexable_content` currently
returns distinct `(hash, doc)` pairs with no path, because the same
content hash can back multiple documents at different paths. Node faces
the identical problem in `getPendingEmbeddingDocs` (`store.ts:1925`) and
resolves it with `SELECT d.hash, MIN(d.path) as path, ... GROUP BY
d.hash` — an arbitrary but deterministic representative path, good enough
since it's only used for extension-based language detection, not stored.
Port the same fix: `get_indexable_content` adds `MIN(d.path) AS path` to
its query and returns it in each row; `index_content` gains a
`filepath: str | None = None` parameter and a `chunk_strategy` parameter,
passed straight to `chunk_document`.

**2. `Store.query` (search-time best-chunk selection, `store.py:1515`,
the `chunk_document(ranked.body)` call at line 1545).** `RankedResult`
already carries `file: str` — no schema change needed here, just pass
`filepath=ranked.file, chunk_strategy=chunk_strategy` through. `query`
gains a `chunk_strategy: Literal["regex", "auto"] = "regex"` parameter.

`update` does **not** need this — confirmed Node's `update` command
(`updateCollections`, `qmd.ts:907`) does no chunking/embedding at all,
only re-scan/FTS-index; embedding is `embed`'s job exclusively. The
command-parity table in the project `CLAUDE.md`/`COMMAND_STATUS.md` is
already consistent with this (embed and update are listed separately).

## CLI wiring

`--chunk-strategy <auto|regex>` (default `regex`) added to exactly two
commands, matching Node. Validation follows pyqmd's own established
convention for enum-like CLI options (`OutputFormat` in
`src/qmd/cli/_types.py`, a `str, Enum` subclass) rather than porting
Node's manual `parseChunkStrategy` string check — add a
`ChunkStrategy(str, Enum)` with `REGEX = "regex"` / `AUTO = "auto"` to the
same module, so Typer validates the value itself and lists choices in
`--help`, matching how `--format` is already handled.

The store layer never imports from `qmd.cli` (confirmed: no
`qmd/store/*.py` imports `qmd.cli` anything today, keeping a strict
one-way cli→store dependency), so `chunk_document`/`index_content`/
`query`'s `chunk_strategy` parameter is a plain
`Literal["regex", "auto"]` string, not the CLI enum. The CLI passes
`chunk_strategy.value` across the boundary — exactly the existing
`format.value` pattern used everywhere `OutputFormat` reaches a
store/formatting function (e.g. `search.py:78`,
`documents.py:126`).

- **`embed`** (`src/qmd/cli/commands/embed.py`) — new `chunk_strategy:
ChunkStrategy = typer.Option(ChunkStrategy.REGEX, "--chunk-strategy")`
  option, threaded into the `store.index_content(row["hash"], row["doc"],
filepath=row["path"], chunk_strategy=chunk_strategy.value)` call in the
  embed loop.
- **`query`** (`src/qmd/cli/commands/search.py`) — same option, threaded
  into `store.query(..., chunk_strategy=chunk_strategy.value)`.

`search`/`vsearch` don't chunk at query time in pyqmd (no best-chunk
selection — that's `query`-only, confirmed by reading `search.py`), so
they don't get the flag, matching Node where only `query`'s
`chunkStrategy` actually affects behavior even though the flag is
parsed into a shared CLI options struct.

MCP is untouched: no `chunk_strategy` parameter on the MCP `query` tool,
matching Node (verified no `chunkStrategy` anywhere under Node's MCP
server sources).

## `status` reporting

New "AST Chunking" section in `src/qmd/cli/commands/status.py`, ported
from Node's block in `cli/qmd.ts` (~line 600): calls `get_ast_status()`,
prints `active`/`unavailable` overall status, the list of available
languages, and per-language error detail for any that failed to load —
same three-state shape Node uses (all available / some available / none
available).

## Testing

- Unit tests for `_ast.py`: one fixture snippet per language (a small
  file with a class and a function) asserting break points land at the
  right byte→char-converted positions with the right scores; a
  non-ASCII-content case to exercise the byte/char conversion path; an
  unsupported-extension case returning `[]`; a deliberately-broken/empty
  grammar case (or a monkeypatched import failure) returning `[]` without
  raising.
- Unit tests for `_chunking.py`: `chunk_document` with
  `chunk_strategy="auto"` on a code fixture produces different (better —
  boundary-aligned) chunk splits than `chunk_strategy="regex"` on the same
  input; `chunk_strategy="regex"` output is byte-identical to today's
  behavior (regression guard).
- CLI tests for `embed --chunk-strategy` and `query --chunk-strategy`
  validation (bad value rejected with the right message) and wiring
  (mock/spy that the value reaches the store call).
- Parity-suite coverage: a new scenario exercising `--chunk-strategy auto`
  on `embed` against a small synthetic code-file fixture, per
  `parity/README.md`'s existing pattern — closes the `not-yet-built` row
  in `2026-09-12-python-node-parity-suite-design.md`'s table for
  `--chunk-strategy <auto|regex>`.
- Update `COMMAND_STATUS.md`'s "Roadmap tie-back" section (#5 line) and
  the `status` command's row once the AST Chunking section ships.

## Dependencies

`pyproject.toml` `[project] dependencies` gains:

```
"tree-sitter>=0.26.0",
"tree-sitter-python>=0.25.0",
"tree-sitter-typescript>=0.23",
"tree-sitter-javascript>=0.23",
"tree-sitter-go>=0.23",
"tree-sitter-rust>=0.24",
```

(exact floors to be pinned to whatever's current at implementation time,
consistent with the rest of the dependency list's loose lower-bound
style). These are small pure grammar packages (no multi-MB WASM
download concern Node's own doc-comment in `ast.ts` flags for itself).
