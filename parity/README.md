# pyqmd ↔ Node qmd parity suite

Validates that `pyqmd` behaves equivalently to a frozen, pinned commit of
the reference Node `qmd` — structurally (CLI/MCP output shape) and by
search quality (IR metrics on a real corpus). Full design rationale:
`../docs/specs/2026-09-12-python-node-parity-suite-design.md`.

## Running the suite

```sh
uv run pytest parity/                                    # against the built-in scifact profile
uv run pytest parity/ --dataset-config path/to/mine.yaml  # against your own profile
PARITY_DATASET_CONFIG=path/to/mine.yaml uv run pytest parity/  # equivalent, via env var
```

No `bun`/Node dependency is needed to run the suite day-to-day — it
compares live `pyqmd` against already-captured JSON snapshots.

## Building your own dataset profile

Create a YAML file (anywhere on disk — it doesn't need to live in this
repo) with:

```yaml
name: mine
corpus_dir: /path/to/your/markdown/collection
queries_file: /path/to/your/queries.yaml # a flat YAML list of query strings
# qrels_file: omit this if you don't have curated relevance judgments --
# the suite automatically runs in agreement-mode (comparing pyqmd's
# results against Node's own captured results for the same queries,
# rather than against a known-correct answer) when it's absent.
```

Then capture a Node baseline from your own corpus once (requires `bun`
and a working Node `qmd` checkout — do this _before_ you fully switch
away from Node):

```sh
uv run python -m parity.capture_node_snapshots --qmd-repo-root /path/to/qmd --dataset-config your-profile.yaml
```

This writes into `parity/node_ref/<your-profile-name>/`, which is
gitignored for every profile except the built-in `scifact` one — your
captured baseline never gets committed, regardless of where your profile
YAML itself lives.

## Re-capturing the built-in scifact profile

Only do this deliberately, when intentionally updating the pinned Node
reference (check `parity/node_ref/scifact/COMMIT.txt` for what's
currently pinned):

```sh
uv run python -m parity.capture_node_snapshots --qmd-repo-root ../qmd
```

To re-capture only one phase — e.g. just the CLI flows, which takes a
few minutes instead of the quality baseline's hours — pass `--phase`
(`structural`, `cli-flow`, `mcp`, or `quality`; default `all`):

```sh
uv run python -m parity.capture_node_snapshots --qmd-repo-root <path-to-qmd-checkout> --phase cli-flow
```

A single-phase capture refuses to run unless the Node checkout's `HEAD`
matches `COMMIT.txt`, and leaves `COMMIT.txt` untouched, so one profile
directory never mixes snapshots from two Node commits.

The quality phase runs the full query set `--quality-runs` times
(default 30) against a single indexed corpus, clearing Node's `llm_cache` with
`qmd cleanup` between passes, and records the per-metric mean plus the
run-to-run standard deviation. See the qrels-mode margin note below for why.

```sh
uv run python -m parity.capture_node_snapshots --qmd-repo-root <path-to-qmd-checkout> --phase quality --quality-runs 30
```

### Output-text parity for CLI flows

Besides the extracted shapes in `cli_flow/`, the flow capture writes
`cli_flow_raw/<flow>.json`: each step's Node args/stdout/stderr/exit
code, with run-specific paths already replaced by `<CORPUS_n>`/
`<INDEX_DIR>`/`<CWD>` at capture time, so no real (personal) path is
ever committed. `test_cli_flow_step_text_matches_node` normalizes both
sides with `parity/_text_normalize.py` (strips color, maps pyqmd's own
temp paths to the same placeholders, and replaces durations, `… ago`
times, docids, and `qmd`/`pyqmd` in hint text) and
then requires an exact match of stdout and stderr separately. Until a
`cli_flow_raw/` capture exists, every case skips with a message saying
how to create one.

A deliberate difference is declared on its step in
`parity/scenarios/cli_flow_scenarios.py`, never silently:
`text_subs=[TextSub(pattern, replacement, reason)]` for a narrow
substitution applied to both sides, or `text_skip_reason="..."` to skip
the step's text check entirely. Both require a reason. Allowed
categories: Node-only machinery, deliberate pyqmd supersets, and
unavoidable output differences (see the 2026-09-24 output-text parity
design spec).

## Known gaps

`just test-parity` should show a clean, all-green suite -- any scenario
marked `xfail(strict=True)` would be listed here, with why, so a real
regression is never lost in a wall of expected red. If any of these ever
starts unexpectedly passing, the suite fails loudly (XPASS) until the
marker is removed. None are currently marked this way (the previous
entry, `multi_get_glob_pattern_not_supported`, was resolved once
multi-get gained real glob-pattern support).

MCP scenarios are captured too, via `parity/_mcp_client.py`: the capture
script spawns a real Node `qmd mcp` server over stdio and speaks the MCP
protocol to it directly (the `mcp` package's client, the same package
pyqmd's own server is built on -- no separate script or client/server pair
needed). Building this out surfaced real, previously-unverified
differences in Node's own MCP surface (the `query` tool's `file` field is
deliberately bare, unlike `get`/`multi_get`'s `uri`; `status`'s document
counts are flat top-level fields, not nested under a `counts` key like
pyqmd's) -- all fixed and verified against a live Node MCP server, not
just assumption.

## Performance benchmark (`parity/benchmark.py`)

A separate, manual script -- `just bench-scifact` -- times pyqmd against Node qmd on the
SciFact profile (indexing throughput, cold-start latency, per-query CLI latency, peak RSS).
It asserts nothing; see `benchmark.py`'s own docstring for methodology.

Two findings from running it are worth keeping in mind when interpreting future runs:

- **pyqmd's per-process MLX model-load cost doesn't shrink on repeat launches the way
  Node's does.** Timing `mlx_embeddings.load()` across three fresh, back-to-back processes
  for the same model showed flat `user` (CPU) time each run (~1.47s) -- the cost is CPU-bound
  (safetensors deserialization, MLX array construction, `mx.eval()` materializing weights
  into Metal-backed unified memory), not disk-I/O-bound. Node's GGUF loading via llama.cpp is
  mmap-backed, so a second cold process just maps already-page-cached bytes -- cheap. There is
  no mmap-shaped fix for MLX's side of this: the bottleneck isn't reading bytes off disk, it's
  constructing the model each time. The practical equivalent of Node's cheap repeated
  invocations is **not reloading at all** -- run `pyqmd mcp` (stdio or `--http`) for repeated
  querying instead of one CLI process per query; see the README's "Performance note".
- **A real, now-fixed bug inflated every pyqmd CLI command's startup, including pure-BM25
  `search`.** `pyqmd_mlx/store/store.py` imported `pyqmd_mlx.llm` (embed/rerank/expand_query) at module
  level, and since every command opens a `Store`, every command -- `search`, `status`, `ls`,
  `get`, none of which touch embeddings -- paid the full `mlx_embeddings` -> `mlx_vlm` ->
  `transformers` import chain (~700ms) regardless. Fixed by deferring those imports into the
  function bodies that actually call the models (`pyqmd_mlx/llm/embed.py`, `rerank.py`, `expand.py`).
  `search` dropped from ~1.14s mean to ~0.45s in local testing; the remaining gap against
  Node's ~64ms is process/interpreter startup (`uv`/Python/Typer vs. bun), not MLX. Re-run
  `just bench-scifact` after any change near `pyqmd_mlx.store.Store`'s imports to catch a
  regression here -- the benchmark's `search` row is the canary.

**How the qrels-mode quality test's margin works.** Node's own metrics
aren't reproducible run to run: its query expansion samples at temperature
0.7 (`qmd/src/llm.ts`, `expandQuery`), so each run searches with slightly
different query variants. (An earlier note blamed the rerank step; the
2026-09-24 calibration traced it to expansion instead.) pyqmd samples
its expansions with Node's settings and system prompt but seeds each query
from a hash of the query and model name, so it stays deterministic: two full
passes in separate processes gave identical rankings (2026-09-25). So the
baseline in `node_query_baseline.json` is the
mean of 30 Node query passes over one indexed corpus, captured 2026-09-24
against the pinned commit, with `qmd cleanup` clearing Node's expansion
cache between passes. Each metric's allowed shortfall is
`max(3·σ, 1/num_queries)`, where σ is Node's run-to-run standard deviation
and the floor is one query's worth of change (`parity/_quality_baseline.py`).
A baseline without a `calibration` block (a single-run capture, e.g. an
older personal profile) falls back to a fixed 0.05.

| metric       | Node mean (30 runs) | σ      | margin | pyqmd (2026-09-25) |
| ------------ | ------------------- | ------ | ------ | ------------------ |
| mrr          | 0.7579              | 0.0208 | 0.0623 | 0.7880             |
| ndcg_at_10   | 0.7781              | 0.0174 | 0.0523 | 0.7995             |
| recall_at_10 | 0.8722              | 0.0154 | 0.0461 | 0.8667             |

Recall@10 passes with 0.041 to spare (0.8667 against a 0.8261 threshold).
It is one seed's draw; the multi-seed spread below is the better guide. Before
2026-09-25 pyqmd decoded expansions greedily, which put it at 0.8333 on every
run, below Node's 0.872 mean. The Recall@10 spikes traced that gap to greedy
decoding (its `vec:` lines drift away from the query) and to a system prompt
Node sends implicitly. Seeded sampling with that system prompt
(`docs/specs/2026-09-25-seeded-expansion-sampling-design.md`)
closed it: ten alternate seeds averaged 0.884 (range 0.860–0.900), in line
with Node. Since then expansion also decodes under Node's output grammar
(`docs/specs/2026-09-25-expansion-output-grammar-design.md`),
which moved the ten-seed mean to 0.870 (range 0.833–0.900).
Re-quantizing the expansion model to a mixed 4/6-bit recipe
(`docs/specs/2026-09-25-expand-model-requantize-design.md`) gave
a thirty-seed mean of 0.860 (range 0.827–0.900), against 0.871 for the old
4-bit model on the same seeds. The shipped seed lands just above that mean.
It is one draw and wasn't tuned.
See the calibration spec
(`docs/specs/2026-09-24-qrels-margin-calibration-design.md`) for
the margin method and the superseded 2026-09-13 n=5 finding.
