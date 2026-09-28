# Multi-agent pyqmd workflow and skills — design

**Date:** 2026-09-22
**Status:** Implemented (skills); the shared-server setup is documented in
the bundled `references/mcp-setup.md`.
**Related:** `2026-09-18-skill-commands-design.md` (`pyqmd skill
show`/`install`, the one bundled skill this design extends);
`2026-09-24-mcp-http-parity-design.md` (the HTTP server's origin/host
guard); `2026-09-18-bench-command-design.md` (what `pyqmd-bench` drives).

## Goal

Several agents (Claude Code, Claude Desktop, OpenCode, scripts, other MCP
clients) share the same pyqmd collections with:

- one connection story;
- one retrieval discipline;
- a skill split that keeps index mutation safe while making research and
  search-quality work repeatable.

## Transport: one shared HTTP server, stdio for local one-offs

pyqmd is MLX-only, so it runs on an Apple Silicon Mac; there is no Linux
host option. An always-on Mac runs `pyqmd mcp --http` as a long-lived
server. The MLX models load once and every HTTP client shares them.
Local, occasional clients keep per-client stdio.

| Client                   | Connection                                     |
| ------------------------ | ---------------------------------------------- |
| Claude Code (heavy use)  | HTTP to the shared server                      |
| OpenCode CLI (heavy use) | HTTP to the shared server                      |
| Scripts / other clients  | HTTP `POST /mcp`, `POST /query`, `GET /health` |
| Claude Desktop (local)   | stdio `pyqmd mcp`                              |
| OpenCode Desktop (local) | stdio `pyqmd mcp`                              |

Why not the alternatives:

- **stdio everywhere** pays the full MLX model load per agent, per query.
- **Two indexes** (Node `qmd` somewhere else plus pyqmd on the Mac)
  doubles the index and needs `embed --force` on each side, because the
  two tools' vectors aren't compatible.

The shared server is the only setup where repeated embedding-backed
queries stay cheap.

### Server operation

- Default bind is loopback. Bind off-loopback (`--host`) only when other
  machines need it.
- Liveness: `GET /health`; index health: `pyqmd status`.
- The endpoints are unauthenticated. Put authentication (a VPN, or a
  reverse proxy with auth) in front before exposing the server off-host,
  and set `QMD_ALLOWED_ORIGINS` / `QMD_ALLOWED_HOSTS` for the origin/host
  guard when binding off-loopback.

## Skills

Before this design there was one bundled skill (`pyqmd`, plus
`references/mcp-setup.md`), installed with `pyqmd skill install`. The
plural `skills` discovery command was deferred until a second bundled
skill existed; this design adds three.

### 1. Fix the bundled `pyqmd` skill (read path, no index writes)

- **Collection routing:** discover the user's collection names with
  `collection list` and `status` rather than assuming them; when to pass
  `-c/--collection`; how the default scope interacts with `collection
include`/`exclude` (an explicit `-c` always searches, even an excluded
  collection).
- **MCP first for repeated queries:** a one-shot CLI call pays seconds of
  MLX load. Use `--format json` / `--format files` for agents, `--intent`
  and `--no-rerank` where they fit, and `--chunk-strategy auto` for code
  collections only.
- **Retrieval discipline:** snippets are not documents. `get`/`multi-get`
  before asserting something is absent; cite `qmd://collection/path` plus
  the `#docid`.
- **No speculative indexing:** never run `collection add`, `update`,
  `embed` or `cleanup` unless the user explicitly asked.

`references/mcp-setup.md` gains the shared-server pattern, the per-client
table above, and the origin/host allowlist notes.

### 2. New `pyqmd-librarian` (the only skill that mutates the index)

- **Confirm before write:** `collection list`/`show` → `cleanup
--dry-run` preview → explicit user confirmation → `update [-c]` /
  `embed [-c]` → verify with `status`.
- Review a collection's `update-cmd` hook before its first run.
- **Intake checklist:** convert PDFs/docs to one `.md` per document,
  keep dated filenames, exclude noise at add time (`Archive/**`,
  `drafts/**`, `node_modules/**`, `.git/**`, `dist/**`, `build/**`,
  `.venv/**`), add a context for each new collection.

### 3. New `pyqmd-researcher` (multi-hop reads, read-only)

- Plan → `query` with an intent → full `get`/`multi-get` retrieval →
  synthesize with citations → state what wasn't found.
- Backend selection: `search` (exact terms, BM25, no MLX load),
  `vsearch` (semantic), `query` (default: hybrid plus rerank); `--filter`
  metadata expressions; `--min-score` and `-n` tuning.

### 4. New `pyqmd-bench` (evaluation loop, read-only)

- The fixture format (queries plus expected files), `pyqmd bench
<fixture> [--json] [-c]`, reading recall/MRR/nDCG per backend
  (bm25/vector/hybrid/full-reranked), tuning one thing at a time
  (chunking, rerank, `min-score`), then re-benching.
- No bundled default fixture: the user writes it, or an agent does when
  asked.

## Packaging and docs

- Each skill lives at `src/pyqmd_mlx/skills/<name>/SKILL.md` (plus
  `references/` as needed) and is wired into `skill list`/`show`/`install`.
- With four bundled skills, the deferred plural `pyqmd skills
list|get|path [--json]` discovery command is unblocked.
- Keep the repo's command reference and
  `src/pyqmd_mlx/skills/pyqmd/SKILL.md` in sync on any CLI surface change.
- Update `COMMAND_STATUS.md`'s skills rows.
