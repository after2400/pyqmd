# qmd: Python/MLX rewrite — roadmap

**Status:** History snapshot as of the 0.x public launch. Work tracking continues privately.

All numbered sub-projects are complete (#1-4 and #6-#12, plus #5), as is
the unnumbered, cross-cutting parity suite.

Date: 2026-09-10

## Decision

Rewrite this fork of qmd from Node/TypeScript to Python, targeting
**Apple Silicon + MLX exclusively**. No GGUF, no cross-platform support
(no Metal/Vulkan/CUDA/CPU backend selection, no Windows/Linux).

## Why

- The user develops almost exclusively in Python; Node/TypeScript is
  not a natural fit for ongoing maintenance.
- This fork has already diverged from upstream `tobi/qmd` in practice
  and is not expected to keep merging upstream changes, so there is no
  `git merge` compatibility cost to a language rewrite. (This was the
  first question raised and the blocking concern — confirmed moot.)
- The fork's hardware target is a single machine (Apple M4 Max /
  128GB), so cross-platform GPU support (the current
  `QMD_LLAMA_GPU`/CPU-fallback matrix in `src/llm.ts`) is unneeded
  complexity being carried for no benefit.

## Key research findings from this session (avoid re-deriving these)

- **`llama-cpp-python` has no reranking support** — open, unresolved
  upstream issue
  ([abetlen/llama-cpp-python#1794](https://github.com/abetlen/llama-cpp-python/issues/1794)).
  Python is actually _behind_ `node-llama-cpp` here. This is one reason
  "just port GGUF handling to Python" would have been a regression —
  it's moot now since the plan drops GGUF entirely.
- **No viable in-process MLX binding exists for Node/TypeScript**:
  - [`mlx-node/mlx-node`](https://github.com/mlx-node/mlx-node) (157★,
    explicitly supports the Qwen3 architecture family) exposes only a
    chat/generation API (`ChatSession.send()`/`sendStream()`) — no
    access to raw next-token logits. Disqualifying: Qwen3-Reranker's
    actual scoring method _is_ reading the logit on a "yes" vs "no"
    token, not sampling text. Also requires macOS 26+ for the
    published binary.
  - [`frost-beta/node-mlx`](https://github.com/frost-beta/node-mlx)
    (283★) is raw tensor/array ops only (numpy-equivalent) — no
    model-loading or transformer-forward-pass layer. Using it would
    mean reimplementing a chunk of `mlx-lm` ourselves in TypeScript.
  - This absence is the direct reason a Node-based qmd would have
    needed an HTTP bridge to a separate (Python) MLX server for
    reranking. The Python rewrite makes that bridge unnecessary.
- **Python's MLX ecosystem is mature and sufficient**:
  - [`mlx_embeddings`](https://github.com/Blaizzy/mlx-embeddings)
    (Blaizzy, 440★, actively maintained) supports embeddings _and_
    reranking, including the Qwen3-Reranker family (confirmed via its
    Qwen3-VL-Reranker-2B example), loads directly from `mlx-community/*`
    MLX-format repos, and has a simple `load()` + `model.process()`
    API. This is the library the new LLM layer should build on for
    embedding and reranking.
  - `mlx-lm` is the equivalent for text generation (query expansion
    role).
  - Both run in-process in the same Python process as the rest of the
    application — no subprocess, no HTTP server, no port management.
- **Candidate MLX models already identified**:
  - `mlx-community/Qwen3-Reranker-0.6B-mxfp8` and
    `mlx-community/Qwen3-Reranker-8B-mxfp8` (converted via
    `mlx-embeddings` itself).
  - `vserifsaglam/Qwen3-Reranker-4B-4bit-MLX` (converted via `mlx-lm`
    0.26.3, traced to the official `Qwen/Qwen3-Reranker-4B` checkpoint).
  - An embedding-model equivalent will need the same kind of check
    (does `ggml-org/embeddinggemma-300M` or an equivalent have a
    legitimate MLX conversion, or does the new embed model choice need
    to change too).
- **Ready-made HTTP reranker servers were investigated as a
  Node-compatible bridge** (`embed-rerank`, `mlx-rerank`) and found
  `embed-rerank` to be a credible, Cohere-API-compatible option with
  built-in support for the `vserifsaglam` model above. **This whole
  line of investigation is now moot** given the Python rewrite decision
  — recorded here only so it isn't accidentally re-investigated. The
  Python rewrite talks to `mlx_embeddings` directly, in-process.
- Separately, and now superseded: this session ran a GGUF reranker
  evaluation (BEIR SciFact benchmark) trying to replace qmd's current
  0.6B default reranker with a larger third-party GGUF quantization.
  Two independent conversions (`mradermacher`, `dengcao`) both failed
  identically at runtime (`node-llama-cpp`'s `createRankingContext()`:
  "Computing rankings is not supported for this model") — recorded in
  `src/llm.ts` near `DEFAULT_RERANK_MODEL` and commit `8262698`. This
  finding is about the _current Node/GGUF_ codebase and is now
  historical context, not directly relevant to the Python/MLX rewrite,
  except as the reason MLX was considered at all.

## Follow-up research findings (post-decision, avoid re-deriving these)

- **4B reranker showed no quality gain over the 0.6B default — reverted,
  not adopted** (2026-09-16). Motivated by this doc's own rationale above
  (MLX unlocks larger rerankers GGUF/`llama-cpp-python` couldn't do at
  all) — tested whether cashing that in actually helps. Swapped
  `DEFAULT_RERANK_MODEL` to `vserifsaglam/Qwen3-Reranker-4B-4bit-MLX` and
  compared against the `mlx-community/Qwen3-Reranker-0.6B-mxfp8` default
  on the scifact profile's 30-query qrels-mode set (same embedded index
  for both, only the rerank model varied):

  | metric    | Node baseline | pyqmd 0.6B | pyqmd 4B |
  | --------- | ------------- | ---------- | -------- |
  | mrr       | 0.7715        | 0.7785     | 0.7733   |
  | ndcg@10   | 0.7838        | 0.8105     | 0.8063   |
  | recall@10 | 0.8667        | 0.9333     | 0.9333   |

  4B is flat-to-marginally-worse, and the gap is smaller than the ~0.04
  run-to-run noise band `parity/README.md`'s own qrels-mode margin
  calibration already found for this suite — not a real regression, just
  no detectable lift. (That noise band was re-measured over 30 Node runs on
  2026-09-24 — σ 0.021 MRR, 0.017 nDCG@10, 0.015 Recall@10 — and traced to
  Node's sampled query expansion, not reranking; see `parity/README.md`.)
  Cost side is unambiguous, though: `pyqmd query`
  (CLI, cold, same index/query, 5 runs each) went from ~6.6s/~2.68GB
  peak RSS at 0.6B to ~17.3s/~4.34GB at 4B (~2.6x slower, ~1.6x more
  memory). Reverted `DEFAULT_RERANK_MODEL` back to 0.6B (confirmed via
  `git diff` after the experiment — no net change landed). Verdict: not
  worth it _for this model_; doesn't invalidate the "MLX unlocks bigger
  rerankers" rationale itself, since Node/GGUF still categorically can't
  do this at all. If picked up again, try the already-identified
  `mlx-community/Qwen3-Reranker-8B-mxfp8` instead, and/or a larger query
  set than scifact's 30 — n=30 is thin enough that this result is
  suggestive, not conclusive.

## Sub-project decomposition and build order

Rewrite is too large for one spec. Build and validate in this order —
each gets its own brainstorming session (questions → approaches →
design → spec) before implementation. Numbers are stable identifiers,
not a live to-do order — #7 and #8 were added after #1-6 and
deliberately numbered last rather than inserted mid-sequence, to avoid
renumbering churn against existing memory/spec references by number.
That same reasoning applies to this Completed/Remaining split below: it
reorganizes the _presentation_, not the numbers.

### Completed

1. ✅ **LLM layer via MLX** (embed / rerank / query-expansion using
   `mlx_embeddings` + `mlx-lm`). Built and validated first, standalone,
   before anything else — it was the highest-uncertainty, highest-value
   piece.
2. ✅ **Storage layer** (SQLite FTS5 + `sqlite-vec` — both have official
   Python bindings; this is largely mechanical porting of the existing
   schema/queries in `src/store.ts`, not a redesign).
3. ✅ **CLI command surface** (collection/context management,
   search/query/bench — port the command _semantics_ from
   `src/cli/qmd.ts`, not the Bun-specific plumbing).
4. ✅ **MCP + HTTP servers** (Python has an official `mcp` SDK — portable;
   `src/mcp/server.ts` is the reference for tool surface).
5. ✅ **Packaging/distribution** — local install only so far (`just
install`, `uv tool install --editable .`); no PyPI publish yet. (PyPI
   package + console-script entry point, replacing the npm package +
   `bin/qmd` shell-wrapper story — note the current wrapper's Darwin
   Metal-residency env var handling, `GGML_METAL_NO_RESIDENCY`, becomes
   irrelevant once GGUF is dropped.)
6. ✅ **Metadata filtering** (added 2026-09-11, during sub-project #4's
   brainstorming — this was never actually part of the original 6-item
   decomposition above; it kept surfacing as an explicitly deferred gap
   in sub-projects #2, #3, and #4 without ever being tracked anywhere,
   so it gets its own numbered slot now instead of being silently
   deferred a fourth time). New `Store`-level metadata storage/indexing
   plus a filter-AST parser/evaluator (recursive `and`/`or`/`not`,
   `eq`/`ne`/`gt`/`lt`, `in`/`nin`/`all`, `exists` — reference:
   `parseMetadataFilter`/`MetadataFilter` in `src/store.ts` and its
   usage in `src/mcp/server.ts`'s `query` tool), exposed through the
   CLI's `search`/`query` commands and the MCP `query` tool's `filter`
   parameter. Independent of the transport/protocol work in #4 — pure
   Store-schema work, deserves its own brainstorm rather than being
   bundled into another sub-project's scope. Numbered last (not
   inserted mid-sequence) to avoid renumbering churn against existing
   memory/spec references to sub-projects #4–#6 by number; build order
   is a judgment call at brainstorming time for whichever sub-project
   is picked up after #6, not fixed by this list position.
7. ✅ **`update` command** (added 2026-09-12 during sub-project #7's
   brainstorming, completed 2026-09-14; spec
   `2026-09-13-python-update-command-design.md`, in this directory). Re-scans a collection's files in place, reusing
   `scan_and_register_collection`'s existing indexed/updated/
   unchanged/removed logic (the path this entry described as already
   supported when it was proposed — exactly what #8 built): changed/new
   documents get re-indexed (re-embedding surfaced via the pending-embed
   hint), removed files deactivated, and orphaned content hashes cleaned
   up each run. Adds an optional per-collection `update_command` hook
   (set/cleared via `pyqmd collection update-cmd` or `add --update-cmd`),
   run before each collection's re-scan, and a `--pull` flag (git pull
   --ff-only, skipped per-collection on failure). Resolves the parked
   idempotent-`collection add`-vs-promoting-`update` question this entry
   flagged for re-check: `add` on an existing name or (path, pattern)
   now fails with a clean error pointing at `update`/removal, and
   `update` is the re-index path. Verified: 587 passed / 6 deselected on
   the fast suite, ruff clean, 18 update-CLI tests incl. real git pulls.
8. ✅ **`cleanup` command** (spec
   `2026-09-16-python-cleanup-command-design.md`, in this directory). `pyqmd cleanup [--dry-run]` clears the (currently unused)
   `llm_cache` table, hard-purges `documents` rows soft-deleted by
   `update`/`collection remove` (with explicit `documents_fts`/
   `document_metadata` cleanup, matching `remove_collection`'s
   no-FK-cascade pattern), reuses the existing
   `cleanup_orphaned_content()`, then FTS-optimizes and `VACUUM`s.
   Reports one orphaned-content number rather than Node's two (orphaned
   vectors/orphaned content move together at this project's scale), and
   deliberately has no `-c`/`--collection` scoping — orphan-content
   cleanup must stay index-wide for correctness since content hashes can
   be shared across collections. Surfaced during #8's brainstorm
   (2026-09-13) as a gap explicitly deferred out of that sub-project's
   scope; closes it out. Verified: 646 passed / 6 deselected on the fast
   suite, ruff clean.
9. ✅ **CLI output formatting/parity polish** (added 2026-09-16, surfaced by
   a manual side-by-side pass of real `qmd`/`pyqmd status`,
   `ls`, and `get` output side by side; bundled the follow-up the
   folder-context spec's "Out of scope" section already named but left
   untracked, `2026-09-15-python-folder-context-design.md`). `status`
   gained per-collection `Pattern:`/`Files: N (updated X ago)` lines (via
   a new batched `Store.get_collection_document_stats`), an `Orphaned:`
   content-hash hint (the bit #9's design deferred here), and static
   `Examples`/dynamic `Models`/dynamic `Tips` sections (Tips actually
   inspects which collections lack context or an `update_command`,
   ported from Node's real logic rather than just its text). `ls` now
   prints `size  date  qmd://collection/path` (right-aligned byte size
   via a new `Store.get_active_documents_with_size`, classic `ls -l`
   date formatting) instead of bare relative paths — a default-format
   replacement, not an opt-in flag, since `get`/`multi-get` already cover
   machine consumption via `--format json`. A new `_theme.py` adds a
   small bold/dim/cyan/green/yellow palette (auto-stripped by Click when
   not a TTY, verified empirically before rollout) applied to
   `status`/`get`/`context`/`collection`/`ls`, deliberately excluding
   `update`/`embed`/`cleanup` per this entry's own scope. `_format_bytes`
   gained the space Node's own formatter has (`29.2 MB`, not `29.2MB`),
   and `status`'s `Updated:` line is now a humanized relative time
   instead of a raw ISO timestamp with microseconds. Two real-usage bugs
   surfaced and fixed post-review, both caught via an actual qmd vs
   pyqmd `ls` diff rather than the tests written alongside the port:
   `format_bytes` applied `.toFixed(1)` uniformly, printing `191.0 B`
   instead of Node's bare-int `191 B` for sub-1024 sizes; and
   `format_ls_time` read hour/day/month straight off the stored UTC
   timestamp instead of converting to local time first (Node's
   `Date.getHours()` etc. are local-time), so `ls`'s date column was off
   by the viewer's UTC offset. Verified: 680 passed / 6 deselected on
   the fast suite, ruff clean.

- ✅ **#11 `skill`/`skills` commands** (added 2026-09-17, completed
  2026-09-18) — see `2026-09-18-skill-commands-design.md`. Shipped `pyqmd skill show`/`skill install`
  (singular only — the plural `skills list/get/path` generic
  multi-skill discovery is deferred until pyqmd has 2+ bundled skills,
  e.g. a future `release` skill). Bundled `src/qmd/skills/qmd/SKILL.md`
  - `references/mcp-setup.md` are fresh content written against
    pyqmd's real, verified command and MCP tool surface, not ported from
    Node's (which describes capabilities pyqmd doesn't have — a
    different MCP `query` shape, `doctor`/`trust`/`init`/`bench`, HTTP
    daemon mode). `install` matches Node's anti-staleness design
    (overwrites the copied `SKILL.md` with a stub pointing back at
    `skill show`) and its optional `.claude/skills/qmd` symlink
    (TTY-aware prompt, `--yes`/`--force`, self-loop detection). This work
    also surfaced and fixed a real gap: `CLAUDE.md`'s own "Commands"
    section had already gone stale (missing `collection
include`/`exclude`, `pull`, `--full-path`) — fixed first, then a
    "keep both in sync" instruction added to `CLAUDE.md` itself. Note:
    this list's own position numbers (1-9 above) drift from the stable
    `#N` sub-project identifiers used everywhere else in this repo once
    metadata filtering's #6 slot is accounted for (a pre-existing
    inconsistency, not introduced here) — this entry uses its real `#11`
    identifier rather than a falsely-sequential "10." to avoid adding to
    that drift.

- ✅ **#12 `bench` command** (added 2026-09-17, completed
  2026-09-19) — see `2026-09-18-bench-command-design.md`. Shipped `pyqmd bench <fixture.json> [--json]
[-c <name>]`, running all 4 retrieval backends (bm25/vector/hybrid/
  full-reranked) per query and scoring with pyqmd's own already-tested
  IR metrics (`recall_at_k`/`reciprocal_rank`/`ndcg_at_k`), relocated
  from dev-only `parity/_ir_metrics.py` into the shipped
  `src/qmd/bench/_metrics.py` package so the command actually ships in
  the wheel. Deliberately does not match Node's own bench metric set
  (precision@k/recall@1,3,5/F1) — a real design tradeoff, not an
  oversight, since `bench`'s audience is an installed user's own
  corpus, not a Node-vs-pyqmd comparison (see the design spec's "What
  bench is actually for" section). Fixture JSON keeps Node's field
  names for portability, honors a fixture-level `collection` field
  with the same `-c` over `fixture.collection` over all-collections
  precedence as Node, and rejects (rather than silently mis-scoring)
  a migrated fixture using Node's structured `lex:`/`vec:`/`hyde:`/
  `intent:` query syntax, which pyqmd doesn't support. This entry uses
  its real `#12` identifier rather than a falsely-sequential list
  position, matching the `#11` entry immediately above.

- ✅ **#5 AST-aware chunking** (completed 2026-09-20) — see
  `2026-09-19-ast-aware-chunking-design.md`.
  Ported Node's tree-sitter chunking (`src/ast.ts`) to
  `src/qmd/store/_ast.py`: per-language S-expression queries and scores
  verbatim, merged with the existing regex break points behind opt-in
  `--chunk-strategy auto` on `embed`/`query` (default stays `regex`),
  plus an `AST Chunking` section in `status`. One deliberate difference:
  JavaScript uses the dedicated `tree-sitter-javascript` grammar where
  Node reuses the TypeScript WASM for `.js` — equivalent captures for
  the ported (TS-pattern-free) JS query. This entry uses its real `#5`
  identifier, matching the `#11`/`#12` entries above.

### Cross-cutting (unnumbered)

- ✅ **pyqmd ↔ Node qmd parity test suite** (design spec
  `2026-09-12-python-node-parity-suite-design.md`, in this directory). Not one of the numbered sub-projects above — a validation
  tool spanning both the Python and Node repos, built to give #1-4/#6/#7
  a real migration-safety check rather than trusting each sub-project's
  own tests in isolation. The build plus a full final-review fix
  cycle (10 items) were complete as of 2026-09-13: real golden-snapshot
  captures against the pinned Node commit, structural + MCP + quality
  parity, and a from-scratch MCP-capture mechanism (`parity/_mcp_client.py`)
  that caught several real Node/pyqmd MCP-surface bugs no prior capture
  could see. Real suite result: 22 passed, 1 skipped, 1 xfailed (the
  genuine, documented multi-get glob-matching gap).

### Remaining

No numbered sub-projects remain open — all of #1-#12 are complete (see
entries above), except #6's PyPI-publish half, the next milestone after the
0.x beta.
