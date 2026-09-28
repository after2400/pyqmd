# Python MCP + HTTP Server (sub-project #4) — Design

**Status:** Implemented (2026-09-12)

## Purpose

Build `qmd.mcp`, a Model Context Protocol server on top of the existing `qmd.store.Store`,
exposing search and document retrieval as MCP tools/resources to AI-assistant clients (Claude
Code, Claude Desktop, etc.). This is sub-project #4 of the qmd Python/MLX rewrite, per the
roadmap doc (`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`), which names it
"MCP + HTTP servers" — both transports are in scope for this one sub-project, sequenced.

Reference implementation: `src/mcp/server.ts` (1187 lines) and `src/mcp/origin-guard.ts` — port
tool/resource _semantics_, not Node-specific plumbing (the official Python `mcp` SDK already
provides equivalents for most of the low-level HTTP/security machinery the Node version had to
hand-build).

Prerequisites (both complete): sub-project #2 (`qmd.store.Store` — the stable hybrid-search API)
and sub-project #3 (`qmd.cli` — the `pyqmd` CLI, whose output-formatting primitives this
sub-project reuses rather than duplicates).

## Scope

**In scope:**

- stdio transport (phase 1) — the primary way local MCP clients talk to a tool server.
- Streamable HTTP transport (phase 2, same sub-project) — reuses phase 1's tool registration.
- Tools: `query`, `get`, `multi_get`, `status`.
- Resource: `qmd://{path}` (read-only, `resources/read` only, no `list`).
- Dynamic server instructions (simplified: status-based, no global-context paragraph).
- CLI integration: `pyqmd mcp [--http] [--port N]`.
- Two `Store`-level fixes, motivated directly by this sub-project's concurrency profile (see
  "Concurrency" below): multi-collection filtering (`collection` params become list-accepting)
  and WAL mode + configurable `busy_timeout`.
- Closing a pre-existing CLI parity gap while touching the same code: `-c`/`--collection` on
  `search`/`vsearch`/`query` becomes a repeatable option (it was documented as repeatable in
  `CLAUDE.md` but the sub-project #3 implementation only ever accepted a single value).
- Extracting the CLI's `multi-get` resolution logic (comma-split → per-token
  `find_document_by_identifier` → `--max-bytes` skip-with-reason) into a function shared by both
  the CLI command and the new `multi_get` MCP tool, rather than duplicating it.

**Deferred (explicitly out of scope for this sub-project):**

- Typed sub-queries on the `query` tool (`searches: [{type: 'lex'|'vec'|'hyde', query}]`) — the
  reference's alternative to plain-text query, giving the calling LLM direct control over
  retrieval strategy. `Store.query()` doesn't support this dispatch mode today. Purely additive
  if added later — doesn't require reworking anything built here.
- `multi_get` glob-pattern matching (e.g. `journals/2025-05*.md`) — comma-separated docids/paths
  only, matching the CLI's sub-project #3 precedent. Also purely additive later.
- Metadata filtering (the reference's recursive filter AST: `and`/`or`/`not`, `eq`/`ne`/`gt`/`lt`,
  `in`/`nin`/`all`, `exists`). This was never actually a numbered item in the original 6-sub-project
  roadmap — it kept surfacing as "deferred" in #2 and #3 without ever being tracked. Adding an
  explicit roadmap item for it (see "Roadmap update" below) rather than deferring it again with no
  tracking.
- Daemon mode (`--daemon`/`stop`, PID-file management) for the HTTP transport. v1 runs in the
  foreground only.
- "Did you mean" similar-file suggestions on `get`'s not-found path (a reference-CLI-only
  nice-to-have, not backed by any existing `Store` capability).
- Global context in dynamic instructions (`Store.get_global_context()` doesn't exist yet —
  deferred since sub-project #2).

## Naming

Follows sub-project #3's established convention: package stays `qmd` (`qmd.mcp` alongside
`qmd.llm`/`qmd.store`/`qmd.cli`); no new console-script — the MCP server starts via the existing
`pyqmd` command's new `mcp` subcommand.

## Package structure

- **`qmd/mcp/__init__.py`** — empty.
- **`qmd/mcp/server.py`** — `build_server(store: Store) -> MCPServer` factory (mirrors the
  reference's `createMcpServer`): registers the `qmd://{path}` resource and the four tools.
  Shared by both transports. Also: `run_stdio(db_path: str | None) -> None`,
  `run_http(host: str, port: int, db_path: str | None, allowed_origins: list[str] | None,
allowed_hosts: list[str] | None) -> None` — the two entry points `pyqmd mcp` calls. Exact source
  of `allowed_origins`/`allowed_hosts` (env vars matching the reference's `QMD_ALLOWED_ORIGINS`/
  `QMD_ALLOWED_HOSTS` convention, vs. new `pyqmd mcp --http` flags) is a plan-writing-time decision,
  not pinned here — the SDK's `TransportSecuritySettings` is what actually enforces them either way.
- **`qmd/mcp/_instructions.py`** — `build_instructions(store: Store) -> str`, the simplified
  dynamic-instructions builder (doc counts, collection names, embedding-coverage note, tool usage
  tips — no global-context paragraph).
- **`qmd/mcp/_formatting.py`** — MCP-shaped result formatting: short docid (`#abc123`), per-result
  snippet extraction (reusing `qmd.cli._snippet.extract_snippet`, not duplicating it), line-numbered
  bodies (reusing `qmd.cli._output_search.add_line_numbers`). Different shape than the CLI's 6
  output formats — MCP tools return `structuredContent` + a text summary — but built on the same
  underlying primitives.
- **`qmd/mcp/_errors.py`** — `tool_result_or_error(fn) -> ToolResult`-style helper: MCP's own error
  convention (`isError: true` content block) is not the CLI's `run_or_exit`/`typer.Exit(1)`
  convention (an MCP server can't exit the process on one bad tool call — it must keep serving
  other requests). Catches the same exception set `run_or_exit` catches
  (`sqlite3.IntegrityError`/`ValueError`/`FileNotFoundError`/`NotADirectoryError`).
- **`qmd/cli/commands/mcp.py`** — new command module: `pyqmd mcp [--http] [--port N]`, calling
  `qmd.mcp.server.run_stdio`/`run_http`.

Modified (existing files):

- **`qmd/store/store.py`** — `__init__` gains WAL mode + `busy_timeout` setup (see
  "Concurrency"). `search_fts`/`search_vec`/`_retrieve_and_fuse`/`query`'s `collection` parameter
  becomes list-accepting.
- **`qmd/cli/commands/search.py`** — `-c`/`--collection` becomes a repeatable Typer option on
  `search`/`vsearch`/`query`.
- **`qmd/cli/commands/documents.py`** — `multi-get`'s comma-split/resolution/skip-with-reason logic
  extracted into a shared function (new home: `qmd/cli/_multiget.py`, or inline in `documents.py`
  as a plain function the Typer command calls — task-writing time decides the exact file).

## New `Store` capabilities

1. **Multi-collection filtering.** `search_fts(query, limit, collection: str | list[str] | None)`,
   `search_vec(query, limit, collection: str | list[str] | None, model=None)`,
   `_retrieve_and_fuse(query, collection, candidate_limit, intent)`, and
   `query(query, ..., collection: str | list[str] | None, ...)` all accept either a single
   collection name (existing behavior, unchanged) or a list (new). SQL changes from
   `AND d.collection = ?` to `AND d.collection IN (...)` with the right number of placeholders.
   `search_vec`'s eligible-set exact-scan query (the sub-project #2 small-collection-starvation
   fix) needs the same widening: `WHERE d.collection = ?` → `WHERE d.collection IN (...)` in the
   eligible-hash-seq computation, plus the same change in its second (document-fetch) query.
   `_exact_vec_scan_by_hash_seq` itself is unaffected — it already operates on a pre-computed
   hash_seq list, collection-agnostic.

   This is _why_ it's in this sub-project rather than left deferred: the MCP `query` tool's
   `collections: list[str]` parameter (matching the reference exactly) has no correct
   implementation without it — building a client-side fan-out-and-merge instead would be throwaway
   work the moment `Store` gets real multi-collection support, and the CLI's `-c` was already
   documented as repeatable and simply never implemented that way.

2. **WAL mode + `busy_timeout`.** Ported from the Node reference's `src/db.ts` (`openDatabase`),
   adapted to Python's `sqlite3`:
   - `Store.__init__` sets `PRAGMA busy_timeout = <ms>` (default 120000, overridable via a
     `QMD_SQLITE_BUSY_TIMEOUT` env var — same name and semantics as the Node version, `0` restores
     fail-fast behavior) before attempting `PRAGMA journal_mode = WAL`.
   - The WAL pragma itself needs a short bounded retry-on-`SQLITE_BUSY` loop for a _cold_ database
     (the journal migration takes a brief exclusive lock that does not honor `busy_timeout` — same
     caveat the Node comment documents). Once a database is already in WAL mode, the pragma is a
     cheap no-op.
   - Motivation: `Store.__init__` currently calls `sqlite3.connect(db_path)` with no `timeout=`
     argument (Python's stdlib default: 5 seconds, no WAL) — this has been invisible through
     sub-projects #2/#3 because every test uses `:memory:` or a single short-lived connection, and
     the CLI's per-invocation `Store` rarely overlaps with another process. The MCP server is the
     first thing in this rewrite that holds one connection open for a long-running process's
     lifetime, which is exactly the scenario ("an `update`/`query` racing a long `embed`") the Node
     version's comments say this fix was built for.

## Tool/resource behavior

| Tool/Resource                                                                                                            | Behavior                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query(query: str, limit=10, min_score=0.0, collections: list[str] \| None=None, intent: str \| None=None, rerank=True)` | `Store.query(query, limit=limit, min_score=min_score, collection=collections, intent=intent, skip_rerank=not rerank)`. Each result formatted via `_formatting.py`: `#docid`, rounded score, `extract_snippet`-derived line + snippet (line-numbered), `context`. Returns `{content: [text summary], structuredContent: {results: [...]}}`.                    |
| `get(file: str, from_line: int \| None=None, max_lines: int \| None=None, line_numbers=True)`                            | Same `:line`/`:line:count` suffix parsing already built for the CLI's `get` (Task 9 of sub-project #3's plan) — explicit `from_line`/`max_lines` win over a parsed suffix. `Store.find_document_by_identifier`. Returns an MCP `resource`-typed content block (`qmd://...` URI) on success; `isError` text block on not-found (no "did you mean" — deferred). |
| `multi_get(pattern: str, max_lines: int \| None=None, max_bytes=DEFAULT, line_numbers=True)`                             | Comma-separated docids/paths only (no globs). Calls the shared resolution function extracted from the CLI's `multi-get`. Returns one `resource`-typed content block per resolved document, or a text block noting skip/error per entry.                                                                                                                       |
| `status()`                                                                                                               | `Store.get_status_counts()` + `list_collections()`, reformatted as `structuredContent` (not the CLI's plain-text table).                                                                                                                                                                                                                                      |
| Resource `qmd://{path}`                                                                                                  | `resources/read` only, no `list`. Same `find_document_by_identifier` lookup as `get`, always full body, `add_line_numbers`-formatted.                                                                                                                                                                                                                         |

## Concurrency

Two independent concerns, two independent fixes:

1. **Cross-process** (CLI write while the MCP server's connection sits open): WAL mode +
   `busy_timeout`, described above. Lets a `pyqmd embed`/`collection add` invocation and the MCP
   server's long-lived connection coexist without "database is locked" failures.
2. **Cross-thread, within the MCP server process**: the official `mcp` SDK dispatches sync tool
   functions through `anyio.to_thread.run_sync(...)` — a real thread pool. Two concurrent MCP tool
   calls can run their handler bodies on two different worker threads at the same time. Python's
   `sqlite3.Connection` is not safe for concurrent multi-threaded use of one connection object
   (unlike the Node reference, whose single-threaded JS event loop already serializes all SQL
   execution for free, no matter how many requests appear concurrent). Fix: one shared `Store` for
   the server process's lifetime, with a single lock (`threading.Lock`) wrapped around every Store
   call the MCP tool handlers make, serializing access. This is the closer behavioral match to how
   the Node version already works today, not a new restriction relative to parity — Node never had
   true concurrent DB access either.

   The lock lives in `qmd/mcp/server.py` (module-level, or as a small wrapper object the tool
   handlers hold a reference to), not inside `Store` itself — `Store`'s own callers (the CLI,
   tests) never share one instance across threads, so this is purely an MCP-layer concern and
   `Store`'s public API stays unchanged by it.

## Error handling

MCP tools signal failure via an `isError: true` content block, never by raising an unhandled
exception up through the SDK (which would surface as a generic protocol-level error, losing the
clean message). `qmd/mcp/_errors.py`'s helper catches the same exception set the CLI's
`run_or_exit` catches (`sqlite3.IntegrityError`/`ValueError`/`FileNotFoundError`/
`NotADirectoryError`) and converts each into a clean, one-line `isError` text block. Anything else
propagates as a genuine crash (visible in server logs), matching the CLI's "anything unexpected
gets a real traceback" convention.

## Testing strategy

- Unit tests per tool against a real in-memory `Store` (seeded, no mocks for `Store` itself),
  with a fake `embed`/`rerank` function for anything touching `qmd.llm` — same pattern as
  sub-projects #2/#3.
- One `@pytest.mark.slow` end-to-end test exercising the real stdio transport: spawn the server,
  send real MCP protocol messages, assert on real results against real MLX models — mirrors
  sub-project #3's Task 16.
- A dedicated test for the WAL/busy_timeout fix: two connections to the same on-disk database
  (not `:memory:`), one holding a write transaction open, confirming the second doesn't immediately
  fail with `SQLITE_BUSY` (waits and succeeds within the timeout instead).
- A dedicated test for the cross-thread lock: two overlapping tool calls (real threads, e.g. via
  `concurrent.futures.ThreadPoolExecutor`) against one shared `Store`, confirming no
  `sqlite3.ProgrammingError`/corruption and that both complete correctly.
- HTTP phase: a smoke test starting `run_streamable_http_async`, hitting `/mcp` with a real HTTP
  client, and confirming the origin-guard genuinely rejects a disallowed `Origin`/`Host` header
  (not just that it accepts a valid one).
