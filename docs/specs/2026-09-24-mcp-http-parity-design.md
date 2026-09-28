# `mcp --http` parity — Design

Date: 2026-09-24.
**Status:** Implemented (2026-09-24). Tier 1 and Tier 2 landed; Tier 3 daemon
lifecycle and REST stay deferred per §5/§4.5.
Outcome: gap matrix + fix designs. `COMMAND_STATUS.md`'s `mcp --http` row is
✅ verified for matching parity (output fidelity stays 🟡 — no byte-level
output-text audit vs Node HTTP was scoped).

## 1. Purpose

Only the stdio transport is parity-tested today (`COMMAND_STATUS.md` `mcp --http` row;
roadmap "Next step" pickup list). Both transports exist (`pyqmd mcp` vs
`pyqmd mcp --http`), but HTTP has no parity coverage and several known or
suspected divergences from Node's HTTP mode have never been audited in one
place.

This spec audits pyqmd's HTTP transport against the Node reference
(the Node `qmd` repo: `src/mcp/server.ts`, `src/mcp/origin-guard.ts`,
`src/cli/qmd.ts` `mcp` case, `src/cli/mcp-pid.ts`) and the existing pyqmd
implementation (`src/pyqmd_mlx/mcp/server.py`, `src/pyqmd_mlx/cli/commands/mcp.py`,
`src/pyqmd_mlx/mcp/_formatting.py`, `src/pyqmd_mlx/mcp/_instructions.py`).
It designs fixes precisely where they are required and records explicit
diverge/defer verdicts everywhere else, so a follow-up plan can ship
incrementally without re-auditing.

Prior art: sub-project #4 design (`2026-09-11-python-mcp-server-design.md`)
built both transports; this spec does not relitigate it — it measures the
HTTP half against Node and tiers the gaps.

## 2. Tier definitions

The gap matrix (§6) gives every row a tier:

- **Tier 1 (must-match):** security-relevant or transport-correctness. Ships in
  the first follow-up plan and flips the `COMMAND_STATUS.md` row on landing.
- **Tier 2 (should-match + REST decision):** user-visible defaults and tool
  niceties, plus the `POST /query` / `POST /search` build-vs-defer decision
  (§4.5). Verdicts land in-spec; implementation may split across follow-ups.
- **Tier 3 (explicitly not-ported):** daemon lifecycle only (§5). Deferred with
  re-entry conditions, not designed.

Columns per row: Node behavior (source ref) | pyqmd behavior (source ref) |
Tier | Verdict (`match` / `fix` / `diverge-by-design` / `deferred`) | Follow-up
(parity test or plan task, or "none").

## 3. Tier 1 — must-match

### 3.1 Origin-guard semantics

Node (`origin-guard.ts`): `resolveOriginGuard` + `checkRequestOrigin`.

- Missing `Origin` is allowed (non-browser clients omit it; browsers cannot).
- Loopback `Origin` always allowed: `localhost`, `*.localhost`, `::1`,
  `0:0:0:0:0:0:0:1`, `::ffff:127.x`, full `127.0.0.0/8`. `http://0.0.0.0:*`
  is explicitly untrusted.
- `QMD_ALLOWED_ORIGINS=*` disables all checks (`disabled: true`).
- Host check is conditional: concrete bind (e.g. `localhost`, `127.0.0.1`) →
  enforce; wildcard bind (`""`, `0.0.0.0`, `::`, `[::]`, `*`) → enforce only
  when an explicit allowlist exists. Non-loopback bind address self-allows.
- Failure shape: `403` with JSON-RPC `{"jsonrpc": "2.0", "error": {"code":
-32003, "message": "Forbidden: <reason>"}, "id": null}`, applied ahead of
  routing (covers `/mcp` and REST alike). Request log line on reject.

pyqmd at audit time (`server.py::_build_transport_security`, since removed)
delegated to the SDK's `TransportSecuritySettings` with always-enforced
allowlists. As shipped (corrected by external review): the SDK layer is off
entirely (`enable_dns_rebinding_protection=False`) and the wrapper in
`build_http_app` is the only guard — a second exact-match layer underneath
can only refuse what the wrapper allows, so keeping it made the allow-rows
above unreachable (live 421s on `[::1]`, `127/8`, wildcard-bind hosts). The
`_build_transport_security` builder and its unit tests were deleted with it.
A second review round then made the wrapper fail closed everywhere: a
malformed Origin port (`:abc`, `:99999`) or a non-ASCII-digit octet is a
403, not a 500, and websocket (or any non-http, non-lifespan) scopes are
closed at the guard — pyqmd, like Node, serves no websocket routes.

Design (follow-up plan implements; this spec pins the verdicts):

| Sub-row                                        | Verdict                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| Missing `Origin` allowed                       | `fix` if SDK rejects it; else `match` with a regression test proving it             |
| `*.localhost`, full `127/8`, `::ffff:` origins | `fix` (extend defaults or pre-check ahead of middleware)                            |
| `http://0.0.0.0:*` untrusted                   | `fix` (ensure never allowlisted by default)                                         |
| `QMD_ALLOWED_ORIGINS=*` disables               | `fix` (explicit bypass, with Node's warning log)                                    |
| Wildcard-bind Host rule                        | `fix` (conditional enforcement, not always-on)                                      |
| Failure body shape (`-32003` JSON-RPC)         | `fix` on the `/mcp` path at minimum; middleware-native codes elsewhere only if free |
| Guard applied ahead of all routes              | `fix` (health + REST + `/mcp` uniformly)                                            |

YAGNI guard: no custom crypto, no per-route allowlists beyond Node's. If the
SDK cannot express a sub-row, a thin pre-check in `build_http_app` (same
process as the existing `/health` custom route) is preferred over forking
middleware.

### 3.2 stdio↔HTTP tool equivalence

Same 4 tools (`query` / `get` / `multi_get` / `status`) plus the `qmd://{path}`
resource return identical `structuredContent` / resource shapes over both
transports. The shared `build_server(store)` factory already guarantees this
structurally (both `run_stdio` and `build_http_app` consume it); the gap is
proof, not plumbing.

- `query.file` stays the bare display-path on both transports (already matched;
  see `_formatting.py`'s comment — Node's own surface is inconsistent here by
  design, CLI uses `qmd://` while MCP `query` does not).
- `status` nesting (`{"counts": ..., "collections": [...]}` vs Node's flat
  `totalDocuments`/`needsEmbedding`/...) is recorded as `diverge-by-design`:
  the parity suite already normalizes it (`_status_shape`), and renaming
  pyqmd's keys to Node's camelCase buys nothing for MCP clients that read
  either shape through the same extract function.
- Follow-up: an HTTP-transport leg in the parity suite running the same 6
  `McpScenario`s through `build_http_app` (via `starlette.testclient` or a
  real `streamablehttp_client`), reusing `parity/_mcp_client.py`'s
  `normalize_call_tool_result` so stdio/HTTP can never drift apart.

### 3.3 `/health`

Reachable without origin failure; body `{status: ok}` exists
(`test_build_http_app_serves_health_endpoint`). Node adds `uptime` seconds
(`server.ts:1010`). Verdict: `fix` — add `uptime` (time since process start).
Cheap, aids observability of the long-lived warm processes §4.5 relies on.
No auth, no allowlist bypass beyond the uniform guard (§3.1).

### 3.4 Bind + error behavior

- `--port` / `--host` / `QMD_HOST` resolution: Node defaults `port 8181`,
  `host "localhost"` with `QMD_HOST` fallback (`server.ts:975`, `qmd.ts:4854`);
  pyqmd defaults `8000` / `127.0.0.1` with no env fallback (`cli/commands/mcp.py`).
  Default-value parity itself is Tier 2 (§4.1); Tier 1 covers only that
  resolution never crashes and precedence is documented: flag > env > default.
- `EADDRINUSE` → clean `Port N already in use. Try a different port with --port.`
  (Node `qmd.ts:4917`). pyqmd today propagates the traceback. Verdict: `fix`.
- SIGTERM/SIGINT: log `Shutting down (SIGTERM)...` style breadcrumb and close
  store/transport (Node `server.ts:1164`). pyqmd relies on SDK teardown.
  Verdict: `fix` to the extent of a log line + ordered close; no custom signal
  machinery beyond what the SDK already provides.

## 4. Tier 2 — should-match + REST decision

### 4.1 Defaults: port and host

Node: `8181` / `localhost`. pyqmd: `8000` / `127.0.0.1`. Verdict: `fix` —
adopt Node's `8181` / `localhost` plus the `QMD_HOST` env fallback, with a
breaking-change note for existing `8000`-bound scripts. Rationale: this is a
migration project and every gratuitous default difference is a papercut; the
flag shapes (`--host` / `--port`) already match, only values and the env
fallback change. The follow-up plan also aligns the `Started on http://...`
log line.

As shipped, a hostname bind listens on every address it resolves to, so
`localhost` binds both `127.0.0.1` and `::1`. That is a deliberate superset
of Node, whose `listen()` takes only the first resolved address (`::1` on
macOS): every client that reaches Node's server reaches pyqmd's, and
`127.0.0.1` clients (the old pyqmd default) keep working. An address
literal binds exactly that address, as in Node.

### 4.2 `query`: `searches` / `candidateLimit`

Node `query` takes `query` xor `searches` (typed `lex`/`vec`/`hyde`, first gets
2× weight) plus `candidateLimit`, with exact error strings for missing/both
(`server.ts:365`). pyqmd takes `query` only; `Store.query` has no typed
dispatch (explicitly deferred since #4). As shipped: `searches` is rejected
with Node's xor strings (a real `searches` implementation needs a
`Store.query` dispatch mode — its own sub-project, not a transport fix), but
`candidateLimit` is passed straight through to `Store.query`'s existing
`candidate_limit` parameter (an external review caught that the original
"reject with Node's error text" design was wrong here — Node supports the
parameter, so it has no rejection text to match; only the neither/both
`query`/`searches` strings are Node's). `candidateLimit` must be a JSON
integer ≥ 1: booleans and strings are refused (as Node's `z.number()`
refuses them, and per the project's bool-first rule), and values below 1
error rather than reaching SQLite, where a negative `LIMIT` means no limit.
Full `searches` implementation is out of this spec.

### 4.3 `get` niceties

Node `get`: `similarFiles` "Did you mean" suggestions on not-found
(`server.ts:479`), `<!-- Context: ... -->` header prefix when the document has
context (`server.ts:494`), `excluded_by_ignore` distinct error, URI-segment
encoding via `encodeQmdPath`. pyqmd `_get_impl`: none of the four. Verdicts:
`similarFiles` → `fix` if `Store` can supply candidates cheaply (it already
does for the CLI path — check before inventing), else documented `deferred`;
context header + ignore-path + URI-encoding → `fix` (small, mechanical).

### 4.4 `multi_get` defaults + shape

Default `maxBytes`: Node `65536` (`DEFAULT_MULTI_GET_MAX_BYTES`) vs pyqmd
`10 * 1024`. Skipped-file wording: Node appends `Use 'qmd_get' with
file="..." to retrieve.`; pyqmd's `[SKIPPED: ...]` lacks the pointer.
Empty-match error text parity (`No files matched pattern: ...` — already
matched). Verdict: `fix` the default and the skip-pointer wording; both are
one-line changes with test coverage in the follow-up.

### 4.5 REST `POST /query` + `POST /search`: build-vs-defer

Node serves structured search without the MCP envelope (`server.ts:1019`):
body requires `searches` array, optional `collections`/`filter`/`limit`/
`minScore`/`candidateLimit`/`intent`/`rerank`, returns `{results: [...]}` with
`qmd://`-prefixed `file` fields (note: REST prefixes, MCP `query` does not —
§3.2). pyqmd has no REST surface.

- For building: running in MCP/HTTP mode is the preferred run method because
  MLX model-load (~1.47s CPU-bound per process; no mmap-shaped fix — see
  `parity/README.md` "Performance benchmark" and `COMMAND_STATUS.md`
  "Performance") does not amortize across one-shot CLI invocations. REST would
  extend that warm-model benefit to non-MCP clients (curl, scripts, probes)
  reusing the already-loaded embedding/rerank models in the HTTP process.
- Against: REST's required `searches` array drags in the §4.2 typed-subquery
  gap (faithful REST needs the dispatch pyqmd lacks); it is a new
  unauthenticated surface needing §3.1 guard coverage and its own parity tests;
  nothing consumes it today.

Recommendation: **defer REST to a follow-up decision, do not design it here
beyond this section.** The spec records the perf argument above verbatim so a
future session need not re-derive it, and names the re-entry trigger: a real
non-MCP consumer (script, probe, second tool) that would otherwise pay cold
model-load per call. If built later, it ports Node's request validation
verbatim (`searches` required-array, `filter` object + AST validation, typed
`limit`/`minScore`/`candidateLimit` coercions) and the `qmd://`-prefixed file
shape — not a redesign.

### 4.6 `status` + instructions notes

`status` nesting: `diverge-by-design` (see §3.2). Instructions: pyqmd's
`_instructions.py` lacks Node's global-context paragraph (no
`get_global_context` exists — deferred since #2) and documents plain-`query`
instead of typed sub-queries. Verdict: `diverge-by-design` with a pointer to
the `Store` capability gap; revisit only when that gap closes. Additionally,
`_instructions.py` documents its own staleness characteristic (snapshot at
startup; long-lived HTTP servers show stale counts until restart) — the spec
keeps that note and adds it to the matrix so it is a recorded decision, not
lore.

## 5. Tier 3 — deferred: daemon lifecycle

`--daemon`, `mcp stop`, PID-file identity helpers (`mcp-pid.ts`, per-`--index`
scoping so named daemons coexist), daemon status line in `status`, log-file
trampoline (`openSync(logPath, "w")`, `detached: true`, `child.unref()`),
`Already running (PID ...)` guard, stale-PID reaping (`isQmdMcpPid`), and the
`Started on http://... (PID ...)` / `Logs: ...` output (`qmd.ts:4823-4929`).

Rationale: foreground `mcp --http` already delivers the warm-model win (§4.5);
daemon is process supervision with real new failure modes (PID recycling,
stale pidfiles, log trampoline ownership, per-index scoping) and zero
retrieval benefit. Revisit trigger: a need for survive-terminal-exit,
launchd, or concurrent multi-index daemons. On revisit, port from
`mcp-pid.ts` + the `qmd.ts` `mcp` case directly (paths, guards, messages
listed above) — no redesign needed.

## 6. Gap matrix (appendix)

| #   | Dimension                                                      | Node                            | pyqmd                                                | Tier | Verdict                                                                                                                          | Follow-up                                                |
| --- | -------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1   | Missing `Origin` allowed                                       | `origin-guard.ts:140` allowed   | SDK middleware (verify)                              | 1    | `fix` if rejected; else `match` + test                                                                                           | HTTP parity leg                                          |
| 2   | Loopback origin set (`*.localhost`, `127/8`, `::ffff:`, `::1`) | `isLoopbackHostname`            | `127.0.0.1`/`localhost` only                         | 1    | `fix`                                                                                                                            | unit + HTTP tests                                        |
| 3   | `0.0.0.0` origin untrusted                                     | explicit exclusion              | verify                                               | 1    | `fix`                                                                                                                            | unit test                                                |
| 4   | `QMD_ALLOWED_ORIGINS=*` disables                               | `disabled: true` + warning log  | no equivalent                                        | 1    | `fix`                                                                                                                            | unit + log test                                          |
| 5   | Wildcard-bind Host rule                                        | conditional enforce             | always enforce                                       | 1    | `fix`                                                                                                                            | unit tests                                               |
| 6   | Failure body `-32003` JSON-RPC                                 | `server.ts:999`                 | `421`/`403` middleware                               | 1    | `fix` on `/mcp`                                                                                                                  | HTTP test                                                |
| 7   | Guard ahead of all routes                                      | ahead of routing                | verify per-route                                     | 1    | `fix`                                                                                                                            | HTTP tests                                               |
| 8   | Tool equivalence stdio↔HTTP                                    | shared `createMcpServer`        | shared `build_server`                                | 1    | `match` (add proof)                                                                                                              | 6-scenario HTTP leg reusing `normalize_call_tool_result` |
| 9   | `status` shape                                                 | flat `totalDocuments`...        | nested `counts`                                      | 1    | `diverge-by-design`                                                                                                              | none (suite normalizes)                                  |
| 10  | `/health` shape                                                | `{status, uptime}`              | `{status: ok}`                                       | 1    | `fix` (add `uptime`)                                                                                                             | HTTP test                                                |
| 11  | `EADDRINUSE` message                                           | `Port N already in use...`      | traceback                                            | 1    | `fix`                                                                                                                            | CLI test                                                 |
| 12  | SIGTERM/SIGINT breadcrumb                                      | `Shutting down (SIG...)`        | SDK default                                          | 1    | `fix` (log + ordered close)                                                                                                      | manual/CLI test                                          |
| 13  | Default port/host                                              | `8181`/`localhost` + `QMD_HOST` | `8000`/`127.0.0.1`, no env                           | 2    | `fix` (adopt `8181`/`localhost` + `QMD_HOST`; breaking-change note)                                                              | CLI tests                                                |
| 14  | `searches`/`candidateLimit`                                    | xor + exact errors              | `searches` rejected; `candidateLimit` passed through | 2    | `fix` (loud `searches` rejection w/ Node xor strings; `candidateLimit` passthrough — Node supports it, no rejection text exists) | MCP tool tests                                           |
| 15  | `get` similar/context/ignore/URI-encoding                      | all present                     | absent                                               | 2    | `fix` (similar conditional on `Store`)                                                                                           | MCP tool tests                                           |
| 16  | `multi_get` default + skip wording                             | 64KB + `qmd_get` pointer        | 10KB, no pointer                                     | 2    | `fix`                                                                                                                            | MCP tool tests                                           |
| 17  | REST `/query`+`/search`                                        | present (`searches`-required)   | absent                                               | 2    | `deferred` w/ trigger (§4.5)                                                                                                     | none now                                                 |
| 18  | Instructions global-context                                    | present                         | absent (no `Store` cap.)                             | 2    | `diverge-by-design`                                                                                                              | none (revisit w/ `Store`)                                |
| 19  | Instructions staleness note                                    | n/a (rebuilt per request)       | snapshot-at-startup                                  | 2    | `diverge-by-design` (documented)                                                                                                 | none                                                     |
| 20  | Daemon (`--daemon`/`stop`/pidfiles)                            | `qmd.ts:4860` + `mcp-pid.ts`    | absent (foreground only)                             | 3    | `deferred` w/ trigger (§5)                                                                                                       | none now                                                 |

## 7. Out of scope (this round)

No code, no parity-suite changes, no `COMMAND_STATUS.md` flip. The flip
(`mcp --http` → ✅ verified) happens when Tier 1's follow-up plan lands, not
on spec approval. No `skills` (plural), no qrels-margin calibration, no PyPI
work — all tracked separately on the roadmap.
