---
name: pyqmd
description: Search and retrieve documents from a pyqmd-indexed markdown knowledge base. Use when the user's question could be answered by content already indexed in a pyqmd collection, or when asked to search/look up/find notes or docs. Prefer `pyqmd query` for most searches; use `pyqmd get`/`multi-get` to retrieve full documents by path or docid.
---

# pyqmd

Hybrid search (BM25 + vector + rerank) over your indexed markdown collections.

## Workflow: search, retrieve, cite

1. **Search** with `pyqmd query "<question>"` to find relevant documents.
2. **Retrieve** the full content of the most relevant hit(s) with `pyqmd get`
   or `pyqmd multi-get` — search results are snippets, not full documents.
3. **Cite** what you found using its `qmd://collection/path` or `#docid`, so
   the user can verify it themselves.

## Search commands

- `pyqmd query "<question>"` — **recommended default.** Hybrid keyword +
  semantic search with reranking. Pass `--intent "<what you're actually
looking for>"` to sharpen ranking when the literal query text doesn't
  capture your real goal.
- `pyqmd search "<keywords>"` — plain BM25 keyword search, no LLM involved.
  Faster, but literal — use when you know the exact terms.
- `pyqmd vsearch "<question>"` — vector similarity only, no reranking.

Common options (all three): `-c/--collection <name>` (repeatable, restricts
to specific collections — omit to search every collection not explicitly
excluded via `collection exclude`), `-n <num>` (result limit), `--min-score
<num>`, `--format json` (for structured output), `--full` (full document
body instead of a snippet), `--full-path` (show the on-disk filesystem path
instead of `qmd://collection/path` + docid).

`query`-only: `--no-rerank` (skip reranking, faster/cheaper), `--intent
<text>` (see above), `--chunk-strategy <regex|auto>` (chunk code files at
AST boundaries for best-chunk selection).

## Collection routing

Collection names are the user's own — discover them with
`pyqmd collection list` (and the contexts shown in `pyqmd status`)
rather than assuming any. A common layout splits by content kind, e.g.
`notes` (journals, meetings, decisions), `library` (converted
PDFs/docs), `code` (repositories).

- Default (no `-c`): searches every collection not excluded via
  `collection exclude`. Prefer this for "where did I see X?" questions.
- `-c <name>`: restrict when you know the domain. Repeatable
  (`-c notes -c library`).
- `qmd://<collection>/<path>` URIs identify hits; cite them with the
  `#docid` shown in results.
- Code collections only: pass `--chunk-strategy auto` on `query`/`embed`
  for AST boundaries; prose collections use regex chunking.

MCP-first: for more than one embedding-backed query, prefer the shared
`pyqmd mcp --http` daemon (models stay loaded) over one-shot CLI calls,
which pay seconds of MLX model-load per invocation. See
`references/mcp-setup.md` for the per-client connection table.

## Retrieving documents

- `pyqmd get <path-or-docid>` — full content of one document. Accepts a
  `qmd://collection/path` URI, a bare path, or a `#docid` (the short hash
  shown next to search results). Supports a line-range suffix:
  `notes/a.md:100` starts at line 100; `notes/a.md:100:40` reads 40 lines
  from line 100. Output is line-numbered by default (`--no-line-numbers` to
  disable).
- `pyqmd multi-get <pattern>` — multiple documents at once, either a
  comma-separated list of paths/docids or a glob pattern (`17*.md`,
  `journals/*.md`, `{readme,changelog}.md`). `--format json/csv/md/xml/files`
  for structured output; plain text otherwise.

Both respect `--full-path`.

## Metadata filtering

`--filter '<json>'` on `search`/`vsearch`/`query` narrows results by
extracted document metadata, using a recursive JSON AST:

```json
{
  "and": [
    { "key": "status", "operator": "eq", "value": "published" },
    { "key": "tags", "operator": "in", "value": ["guide", "reference"] }
  ]
}
```

Operators: `eq`/`ne`/`gt`/`lt`/`gte`/`lte`, `in`/`nin`/`all`, `exists`,
composed with `and`/`or`/`not`.

## Discovery

- `pyqmd status` — index health, per-collection file/embedding counts.
- `pyqmd collection list` / `pyqmd collection show <name>` — what's indexed.
- `pyqmd ls [collection]` — collections, or files within one.
- `pyqmd skills list|get|path [--json]` — bundled agent skill docs (`get` takes `<name>` or `--all [--full]`; `path` takes `[name]`).
- `pyqmd <command> -h` (or `--help`) — flags for any command or subcommand.

## Benchmarking search quality

- `pyqmd bench <fixture.json> [--json] [-c <name>] [--samples N]` — runs a
  user-supplied fixture of queries + expected files against all 4
  retrieval backends (bm25/vector/hybrid/full-reranked) and reports
  recall/MRR/nDCG per backend. Needs a fixture file the user (or you,
  if asked to help build one) writes by hand — there's no bundled
  default fixture. Useful when the user wants to know whether a config
  change (chunking, embeddings, reranking) helped or hurt their search
  quality. `--samples N` re-runs hybrid/full with N different
  query-expansion seeds and reports each metric's mean and min–max — use
  it when comparing two configurations, so a difference smaller than the
  spread isn't mistaken for a real one.

## Setup and maintenance — only when the user asked

These commands write to the index. Don't run them speculatively or to "help" —
only when the user explicitly asks you to index something, refresh it, or
clean it up:

- `pyqmd collection add <path> --name <n>` — index a new directory.
- `pyqmd update [-c <name>]` — re-scan for new/changed files.
- `pyqmd embed [-c <name>] [--force] [--chunk-strategy <regex|auto>]` — generate/refresh vector embeddings (`auto` chunks code files at AST boundaries).
- `pyqmd cleanup [--dry-run]` — clear caches, purge inactive docs, vacuum.
- `pyqmd collection include/exclude <name>` — toggle whether a collection
  participates in default (no `-c`) search results.

## MCP tools

If pyqmd is configured as an MCP server (`pyqmd mcp`), the same
capabilities are available as tools: `query`, `get`, `multi_get`, `status`.
See `references/mcp-setup.md` for client configuration and exact tool
parameter shapes.

## Pitfalls

- Don't `cat`/`sed`/`head`/`tail` a document's on-disk path even when
  `--full-path` reveals it — use `pyqmd get`'s own `:from:count` slicing,
  which is index-aware and stays consistent with search results.
- A search result's snippet is not the whole document — retrieve with `get`
  before asserting a document doesn't contain something.
- `-c`/`--collection` always searches the named collection(s), even one
  excluded via `collection exclude` — exclusion only changes the _default_
  (no `-c`) scope.
- Never run indexing/maintenance commands (`collection add`, `update`,
  `embed`, `cleanup`) unless the user explicitly asked for it.
- The first `pyqmd query` downloads the query-expansion model (~934 MB)
  from Hugging Face. If it fails with
  `Could not load query-expansion model`, report that to the user rather
  than retrying in a loop — they can point `PYQMD_EXPAND_MODEL` at another
  Hub repo id or a local MLX model directory. `pyqmd search` (BM25) needs
  no model and still works.
