# pyqmd — Python/MLX rewrite of qmd

A from-scratch Python port of `qmd` (the Node/TypeScript tool in the sibling
`qmd` repo), targeting Apple Silicon + MLX exclusively. No GGUF, no
cross-platform GPU backend selection. See the private roadmap,
`docs/superpowers/specs/2026-09-10-python-mlx-rewrite-roadmap.md` in the main
checkout, for the full rewrite decision, rationale, and current work order
(a public history snapshot is in `docs/specs/`).

## Workflow

Non-trivial work goes through: a design/spec doc → an implementation plan
→ implementation. Specs and plans are **private**: they are always written
to the main checkout's gitignored `docs/superpowers/specs/` and
`docs/superpowers/plans/`, by absolute path, even from a worktree (find the
main checkout with `git rev-parse --git-common-dir`). Check both for the
most recent files before starting new work. Published, cleaned copies of
specs live in `docs/specs/` under the same file names. A roadmap item that
bundles multiple independent pieces gets decomposed into separate
spec/plan/implementation cycles rather than done as one giant change (see
the private roadmap's own sub-project decomposition and "Next step" pickup
order for the pattern).

When finishing a branch, publish its spec before opening the PR: copy it to
`docs/specs/`, clean it (no index file/collection names or queries, no
host names or paths with a username, no plan links, `docs/superpowers/…`
links rewritten to `docs/specs/…`; keep spike results and rejected
alternatives; set an accurate status line; superseded specs get a banner),
run the private-info check on it, and commit it on the branch. The private
spec stays the source of truth: amend it, then re-publish by cleaning it
again, never by editing the public copy.

Commits, commit messages and PR descriptions go through a private-info
check whose tooling lives in the main checkout's
`docs/superpowers/private/` (chained as legacy git hooks, so it runs in
every worktree). Run it on a PR description before `gh pr create`:
`<main>/.venv/bin/python <main>/docs/superpowers/private/check_private.py --text <body-file>`.

Prefer an isolated workspace (a git worktree) for implementation work over
committing directly to `main`, so a broken or half-finished change never
blocks whatever else might be happening in the repo concurrently.

## Command parity status — read before and after touching CLI/MCP surface

[COMMAND_STATUS.md](COMMAND_STATUS.md) tracks, command by command, whether
each Node `qmd` command exists in pyqmd, whether its results match Node's,
and whether its output text/formatting matches Node's. Before starting any
work that could touch a command's existence, behavior, or output (CLI
commands, MCP tools, `--format` output, colors/sections in `status`/`ls`/
`get`/etc.), read that file to see whether the work affects one of its
rows. After finishing, update the affected row(s) — and the "Roadmap
tie-back" section if a numbered sub-project's status changed — so the file
stays accurate as of the most recent commit. Don't leave it stale for the
next session to rediscover by re-diffing Node and pyqmd from scratch.

## Commands

```sh
pyqmd collection add <path> --name <n> [--mask <glob>] [--exclude <glob>]...
pyqmd collection list                 # List all collections
pyqmd collection show <name>          # Show one collection's details
pyqmd collection remove <name>        # Remove a collection (destructive: cascades embeddings/metadata)
pyqmd collection rename <old> <new>
pyqmd collection update-cmd <name> [command...]  # Set/clear a collection's pre-update hook
pyqmd collection include <name>       # Include a collection in default (no -c) search results
pyqmd collection exclude <name>       # Exclude a collection from default (no -c) search results
pyqmd embed [-c/--collection <name>] [--force] [--chunk-strategy <regex|auto>]  # Generate vector embeddings
pyqmd status                          # Index/collection health summary
pyqmd update [-c/--collection <name>] [--pull]  # Re-scan collections; re-index new/changed files
pyqmd cleanup [--dry-run]             # Clear LLM cache, purge inactive docs, vacuum
pyqmd pull                            # No-op: MLX models download automatically on first use
pyqmd query <query>                   # Hybrid search: FTS + vector + RRF fusion + rerank (recommended)
pyqmd search <query>                  # Full-text keyword search (BM25, no LLM)
pyqmd vsearch <query>                 # Vector similarity search only
pyqmd get <file>[:from[:count]]       # Get a document by path, qmd:// URI, or docid (#abc123)
pyqmd multi-get <pattern>             # Get multiple docs, comma-separated or glob (17*.md, {a,b}.md)
pyqmd ls [collection]                 # List collections or files in a collection
pyqmd mcp                             # Start MCP server (stdio transport)
pyqmd mcp --http [--port N]           # Start MCP server (HTTP transport)
pyqmd skill list                      # List bundled skill names (pyqmd, pyqmd-librarian, pyqmd-researcher, pyqmd-bench)
pyqmd skill show [name]               # Print a bundled skill doc (default: pyqmd)
pyqmd skill install [name] [--global] [--force] [--yes]  # Install a skill into .agents/skills/<name>
pyqmd skills list|get|path [--json]   # Discover bundled runtime skills (get takes <name>|--all [--full]; path takes [name])
pyqmd bench <fixture.json> [--json] [-c/--collection <name>] [--samples N]  # Run search-quality benchmarks against your own corpus
pyqmd --version                       # Print the pyqmd release version and exit
pyqmd [command...] -h, --help         # Help for pyqmd or any (sub)command
```

### Search/query options

```sh
-c, --collection <name>   # Restrict to collection(s), repeatable
-n <num>                  # Result limit (default 20 for search/vsearch, 10 for query)
--min-score <num>         # Minimum score threshold
--filter <json>           # Metadata filter (recursive and/or/not, eq/ne/gt/lt, in/nin/all, exists)
--full                    # Show full document content, not just a snippet
--no-rerank               # (query only) skip LLM reranking
--intent <text>           # (query only) sharpen ranking with a stated intent
--format <kind>           # cli (default) | json | csv | md | xml | files
--line-numbers            # Add line numbers to output
--full-path               # Show on-disk paths instead of qmd:// + docid
--chunk-strategy <regex|auto>  # AST-aware chunking for code files (embed/query; default regex)
```

**Keep this in sync**: whenever a CLI command or flag is added,
changed, or removed, update this section _and_
`src/pyqmd_mlx/skills/pyqmd/SKILL.md` — both describe pyqmd's real command
surface for different audiences (this repo's own agent sessions vs. a
stranger's installed pyqmd) and must not drift apart.

## Architecture

- Names: distribution `pyqmd-mlx` (what `uv tool`/pip see), import package
  `pyqmd_mlx` (`src/pyqmd_mlx/`), command `pyqmd`. Not `qmd`/`pyqmd` as
  import names — both collide with unrelated PyPI packages (see
  `docs/specs/2026-09-24-import-package-rename-design.md`).
- SQLite FTS5 for full-text search (BM25), `sqlite-vec` for vector similarity
- MLX in-process for embeddings, reranking, and query expansion
  (`mlx_embeddings` + `mlx-lm`) — no subprocess, no HTTP bridge, no GGUF
- Model defaults live in `src/pyqmd_mlx/llm/_constants.py` as HF Hub repo
  ids, downloaded into the HF cache on first use. The query-expansion
  model is a mixed 4/6-bit MLX conversion (llama.cpp `Q4_K_M` layout) of Node's
  `tobil/qmd-query-expansion-1.7B-gguf` (rebuild with
  `scripts/convert_expand_gguf.py`), re-hosted as
  `after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6`; override it with the
  `PYQMD_EXPAND_MODEL` env var (a Hub repo id or a local MLX model dir,
  e.g. a local output dir of that script). A load failure is an `ExpansionModelError` → one-line
  `Error:` + exit 1 (CLI, including `bench`) / `is_error` (MCP).
  An older gitignored `models/qmd-query-expansion-1.7b-mlx` dir in a dev
  checkout holds the broken pre-fix weights; delete it.
- Reciprocal Rank Fusion (RRF) combines FTS + vector result lists before rerank
- Metadata extraction/storage/filtering: `src/pyqmd_mlx/store/_metadata.py` +
  `_metadata_filter.py` (extraction happens at `collection add` time only)
- Index stored at `~/.cache/pyqmd/index.sqlite` by default, overridable via
  the `PYQMD_DB` env var (`pyqmd_mlx/cli/_db.py`) — a separate cache directory from
  the live Node `qmd`'s `~/.cache/qmd/`, so both can coexist during migration

## Development

```sh
just develop        # uv sync --dev + install pre-commit hooks
just test-fast       # uv run pytest -m "not slow" -- fast unit suite, run this by default
just test-slow       # real MLX model loads / HF downloads -- slow, run before a release
just test-parity     # pyqmd vs. frozen Node qmd golden-snapshot suite (see parity/README.md)
just lint            # ruff check + format --check
just install         # uv tool install --editable . -- installs the real `pyqmd` command
```

Test markers (`pyproject.toml`): `slow` (real MLX models), `parity` (cross-checks
against captured Node reference — see `parity/README.md`).

## Commit conventions

Markdown/YAML/JSON files are auto-formatted by `prettier` via a `pre-commit`
hook (`.pre-commit-config.yaml`) — it rewrites files in place on commit if
they don't already match its formatting. If you'd rather not commit twice
(once for your change, once for prettier's own reformatting getting
re-added), run `prettier --write <file>` yourself before committing.

Commits are validated by `commitlint` (`.commitlintrc.mjs`) via a `commit-msg`
pre-commit hook. **Both the type and the scope are from fixed, enforced
lists — check `.commitlintrc.mjs` before choosing either; don't invent a new
scope even if it seems descriptive.** As of this writing the valid scopes are
`store`, `cli`, `mcp`, `llm`, `bench`, `skills`, `parity`, `scripts`,
`config`, `docs`, `specs`, `tests` — but treat `.commitlintrc.mjs` itself as the source of
truth, not this list, since it can change. Types are conventional-commit
standard (`feat`/`fix`/`docs`/`test`/etc.) plus a `doh` escape hatch that
never bumps version or appears in a changelog.

**Every commit an agent makes must end with a `Co-Authored-By: <Model Name>
<email>` trailer identifying the acting model** — e.g. `Co-Authored-By:
Claude Sonnet 5 <noreply@anthropic.com>`. This is the only reliable way to
tell, from git history alone, which model produced a given change when
something needs debugging or re-reviewing later. A commit with no such
trailer predates this convention and can't be attributed to a specific
model from git metadata — don't assume a name for it.

## Important: do NOT run automatically

- Never run `pyqmd collection add`, `pyqmd embed`, or any command that
  writes to the index without being explicitly asked — matches the
  standing rule in the parent Node `qmd` repo's own `CLAUDE.md`.
- Never modify the SQLite database directly; only `pyqmd_mlx.store.Store` methods
  touch `conn`.
- `PRAGMA foreign_keys` is only ever enabled at first-time schema creation,
  never re-enabled on reopening an existing on-disk DB — cascade deletes
  cannot be relied on for tables added after the initial schema; check
  `remove_collection`'s explicit cleanup as the pattern to follow.
- Python's `bool` is an `int` subclass — every scalar type-check on
  user-supplied values (metadata, filter literals, YAML-parsed config) must
  check `isinstance(x, bool)` before any numeric check, or a bare `true`/
  `false` silently gets treated as `1`/`0`.
- `parity/capture_node_snapshots.py` is manual and on-demand only — never
  invoke it automatically. It drives a real, frozen Node `qmd` checkout
  (`bun`) and writes golden snapshots other tests compare against; re-run
  it only when deliberately updating the pinned reference.

## Status

Sub-projects #1 (MLX LLM layer), #2 (storage layer), #3 (CLI), #4 (MCP+HTTP
server), #6 (local install, no PyPI yet), #7 (metadata filtering), #8 (the
`update` command), #9 (the `cleanup` command), #10 (CLI output
formatting/parity polish), #11 (`skill` agent-discoverability commands),
and #12 (the `bench` user-facing search-quality command) are complete.
#5 (AST-aware chunking) is complete too — every numbered sub-project is
done. Released as 0.x Beta on GitHub (install from a tag); PyPI
publishing is the next milestone.
Full status, known limitations, and the
roadmap live in the main checkout's private `docs/superpowers/` tree and
this project's memory — see the roadmap doc linked above for the
authoritative list.
