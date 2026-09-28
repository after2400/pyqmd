# Query-expansion model: HF-hosted default + `PYQMD_EXPAND_MODEL` override

**Date:** 2026-09-24
**Status:** Implemented (2026-09-24), amended by docs/specs/2026-09-24-expand-model-gguf-weights-design.md
(the Hub-repo-id default, `PYQMD_EXPAND_MODEL` override and clean load errors
stand; the hosted weights were replaced with a conversion of Node's GGUF
checkpoint, see "Post-implementation finding" at the end)
**Related:** roadmap Backlog entry "PyPI publish prerequisites (other half
of #6)", item 2 (`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`);
the LLM-layer spec's deferred finding about `DEFAULT_EXPAND_MODEL`'s
`__file__`-anchored path (`2026-09-10-mlx-llm-layer-design.md`).

## Problem

`src/pyqmd_mlx/llm/_constants.py` builds `DEFAULT_EXPAND_MODEL` as
`Path(__file__).resolve().parent.parent.parent.parent / "models" /
"qmd-query-expansion-1.7b-mlx"`. That only resolves in an editable install
from the one checkout that holds the gitignored `models/` directory — git
worktrees lack it, and a built wheel resolves it to
`<venv>/lib/python3.12/models/...`, which doesn't exist (the wheel ships no
model files). `mlx_lm.load` then treats the path as a Hub repo id and raises
`HFValidationError`, and nothing catches it at the expansion call in
`Store` (`expanded_lines = self._expand_fn(query, self._expand_model)`), so
every `query` that misses the strong-signal BM25 shortcut crashes with a
traceback. This affects `uv tool install .` and `uv tool install git+<repo>`
today, not only a future PyPI release. There is no way to override the
model: `status` only displays it.

## Where the weights come from

The local weights were self-converted in sub-project #1 (LLM-layer plan,
Task 7):

```sh
uv run mlx_lm.convert --hf-path tobil/qmd-query-expansion-1.7B --mlx-path models/qmd-query-expansion-1.7b-mlx -q
```

i.e. tobil's merged fine-tuned checkpoint (4.06 GB safetensors), quantized to
4-bit affine, group size 64 (~934 MB). As of 2026-09-24 no MLX conversion of
`qmd-query-expansion-1.7B` exists on the Hub (third-party MLX builds exist
only for the newer `qwen3.5-2B` variant).

**License.** `tobil/qmd-query-expansion-1.7B` has no model card/license tag;
its sibling `tobil/qmd-query-expansion-1.7B-gguf` is declared MIT (**wrong
assumption, corrected below:** that repo is _not_ the same weights — see
"Post-implementation finding"), and the base model `Qwen/Qwen3-1.7B` is Apache-2.0.
Re-hosting the conversion under MIT with attribution to both is consistent
with that.

## Decision

1. **Re-host** the existing 4-bit conversion on the Hugging Face Hub as
   `after2400/qmd-query-expansion-1.7B-mlx-4bit` and make that repo id the
   default — the same shape as the embed/rerank defaults (`mlx-community/...`).
2. **Override** via a `PYQMD_EXPAND_MODEL` environment variable, accepting
   either a Hub repo id or a local MLX model directory (`mlx_lm.load`
   accepts both).
3. **Fail clearly**: a load failure becomes a one-line `Error: ...` and exit
   code 1 (CLI), or an `is_error` tool result (MCP) — never a raw traceback,
   and never a silent fallback to un-expanded search.

### Rejected alternatives

- **Load `tobil/qmd-query-expansion-1.7B` directly** (no conversion): nothing
  to host, but a 4 GB download, more RAM, and an unquantized variant pyqmd
  has never been tested with.
- **Convert on first use** into `~/.cache/pyqmd/models/`: nothing to host,
  but the same 4 GB download plus a conversion wait, and more code with more
  failure modes.
- **Warn and degrade** to un-expanded hybrid search on load failure: always
  returns results, but quietly worse ones.

## Design

### Hosting (outward-facing; separate explicit go-ahead)

Upload the 8 files of the local conversion to
`after2400/qmd-query-expansion-1.7B-mlx-4bit`, replacing the stub
`README.md` with a model card: `license: mit`, `library_name: mlx`,
`base_model: tobil/qmd-query-expansion-1.7B`, attribution to tobil's qmd
model and Qwen3-1.7B (Apache-2.0), and the exact convert command and
`mlx-lm` version above. Upload happens via `hf upload` after the user logs
in with `hf auth login`, only on their explicit go-ahead. Code and unit
tests don't depend on the upload; only end-to-end verification of the
default does.

### Resolution (`pyqmd_mlx.llm`)

- `_constants.py`: `DEFAULT_EXPAND_MODEL =
"after2400/qmd-query-expansion-1.7B-mlx-4bit"`, plus
  `EXPAND_MODEL_ENV_VAR = "PYQMD_EXPAND_MODEL"`. The `pathlib` logic is
  removed.
- New `resolve_expand_model() -> str`: returns `$PYQMD_EXPAND_MODEL` when
  set and non-empty, else `DEFAULT_EXPAND_MODEL`. Exported from
  `pyqmd_mlx.llm`.
- `Store.__init__`: `self._expand_model = expand_model or
resolve_expand_model()`. One choke point covers the CLI, the MCP server,
  and `bench` (all construct `Store` via `get_store()`); `Store` already
  reads `QMD_SQLITE_BUSY_TIMEOUT` itself, so this follows existing
  precedent. `status` then displays the effective model; `_hf_link` already
  links a repo id and leaves an absolute path unchanged.
- `expand_query(query, model=DEFAULT_EXPAND_MODEL)` keeps its signature;
  `Store` always passes the resolved model explicitly.

A developer checkout that already has `models/qmd-query-expansion-1.7b-mlx`
can set `PYQMD_EXPAND_MODEL` to that directory to skip the download, or let
the default download once into the HF cache.

### Error handling

- New `ExpansionModelError(RuntimeError)` in `pyqmd_mlx/llm/expand.py`,
  exported from `pyqmd_mlx.llm`.
- `expand.py`'s `_load` wraps any exception from `mlx_lm.load` as
  `ExpansionModelError`, chained with `from exc`:
  `Could not load query-expansion model '<id>': <original error>. Set
PYQMD_EXPAND_MODEL to a Hugging Face repo id or a local MLX model
directory to override.`
- `get_or_load` only caches successful loads, so fixing the env var and
  retrying works without restarting a long-running MCP server.
- `ExpansionModelError` joins `EXPECTED_EXCEPTIONS` in both
  `pyqmd_mlx/cli/_errors.py` and `pyqmd_mlx/mcp/_errors.py`.
- CLI `query` and `bench` wrap their `store.query(...)` calls in
  `run_or_exit`, printing `Error: <message>` to stderr and exiting 1. The
  MCP `query` tool already runs through `tool_result_or_error`, so it
  returns an `is_error` result without further change.

### Tests

Fast unit tests, no MLX loads (TDD):

- `resolve_expand_model()`: env unset → default; empty → default; set →
  the env value.
- `Store` picks up `PYQMD_EXPAND_MODEL` when no explicit `expand_model` is
  passed, and an explicit argument still wins.
- `_load` with a monkeypatched `mlx_lm.load` that raises → an
  `ExpansionModelError` whose message names the model id and
  `PYQMD_EXPAND_MODEL`, with the original exception as `__cause__`.
- CLI `query` (and `bench`) with an `expand_fn` that raises
  `ExpansionModelError` → exit code 1, clean `Error:` line on stderr, no
  traceback.
- MCP `query` with the same failing `expand_fn` → `is_error` result.

`requires_expansion_weights` marker: redefined as "the resolved expansion
model is available **without a network download**" — the resolved value is
an existing local directory, or
`huggingface_hub.snapshot_download(repo, local_files_only=True)` succeeds.
CI and fresh worktrees still skip instead of pulling ~934 MB; machines with
the HF cache populated or `PYQMD_EXPAND_MODEL` set run the marked tests. The
skip message and the `pyproject.toml` marker description are updated to
mention `PYQMD_EXPAND_MODEL`.

### Verification

Reproduce the original bug report's setup: `uv build --wheel`, install into
a scratch venv, build a throwaway index under a scratch `PYQMD_DB` (never the
user's real index), then run `pyqmd query` with a query that misses the
strong-signal shortcut:

- before the upload: the clean `Error:` message and exit 1, no
  `HFValidationError` traceback;
- with `PYQMD_EXPAND_MODEL` pointing at the local conversion: results;
- after the upload: results with the default repo id.

### Docs

- `CLAUDE.md` (Architecture) and `src/pyqmd_mlx/skills/pyqmd/SKILL.md`:
  document `PYQMD_EXPAND_MODEL` and that the expansion model downloads from
  the Hub on first `query`.
- `COMMAND_STATUS.md`: note on the `query` row.
- `2026-09-10-mlx-llm-layer-design.md`: mark the `DEFAULT_EXPAND_MODEL`
  deferred finding resolved, pointing here.
- Roadmap Backlog "PyPI publish prerequisites", item 2: mark done.

## Out of scope

- A `--expand-model` CLI flag (the env var covers CLI, MCP, and bench).
- Pinning the Hub revision (the embed/rerank defaults aren't pinned either;
  the repo is owned by the project).
- Error wrapping for embed/rerank model loads (their defaults are real Hub
  repo ids; not the bug being fixed).
- Revisiting the checkpoint's output quality (the LLM-layer spec's Task 7
  format finding).

## Post-implementation finding (2026-09-24): the source checkpoint is broken

A separate spike found that `tobil/qmd-query-expansion-1.7B`'s
`model.safetensors` — the source of the conversion hosted at
`after2400/qmd-query-expansion-1.7B-mlx-4bit` — is plain Qwen3-1.7B outside
layers 0–1 and the embeddings (layers 2–27 bit-identical to the base), and
matches neither the base nor base + that repo's shipped adapter: a partial or
failed merge. Quantization is not the cause (the unquantized safetensors also
gets 0/18 queries into `hyde:`/`lex:`/`vec:` format). The "same weights"
assumption above is wrong: Node's default, `tobil/qmd-query-expansion-1.7B-gguf`
(MIT), is a separate, fully fine-tuned checkpoint.

Working candidates the spike found: the GGUF weights converted to MLX (18/18
in format with greedy decoding, also at 4-bit, ~3× faster than the current
default), and `tobil/qmd-query-expansion-1.7B-v2` (Apache-2.0, merged from its
adapter; 18/18).

What stands and what's paused:

- **Stands:** the Hub-repo-id default mechanism, `PYQMD_EXPAND_MODEL`, and
  `ExpansionModelError` reporting — none depend on which weights are hosted.
- **Paused:** treating the weights at
  `after2400/qmd-query-expansion-1.7B-mlx-4bit` as correct. That public repo
  still serves the broken conversion; replacing its contents (or choosing a
  new repo id) is a user decision once the spike concludes, and needs its own
  explicit go-ahead since it's outward-facing.
- **Interim:** on the user's go-ahead, the repo's model card got a
  "known-bad weights, replacement pending" warning (Hub commit `02a04a0`);
  the weight files are unchanged. The PR for this branch waits for the
  spike, so the replacement weights land in the same change.

**Resolved:** the weight choice went to Node's own
`tobil/qmd-query-expansion-1.7B-gguf`, converted by
`scripts/convert_expand_gguf.py` and uploaded to the same repo id (Hub
commit `cb00e6e`) — see `2026-09-24-expand-model-gguf-weights-design.md`.
