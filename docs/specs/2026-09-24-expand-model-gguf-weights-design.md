# Query-expansion model: replace the broken weights with Node's GGUF checkpoint

**Date:** 2026-09-24
**Status:** Implemented (2026-09-24), amended by docs/specs/2026-09-25-expand-model-requantize-design.md
(weight source chosen by the user: Node's `tobil/qmd-query-expansion-1.7B-gguf`)
**Related:** `2026-09-24-expand-model-hf-default-design.md` (the Hub-repo-id
default and `PYQMD_EXPAND_MODEL` override this builds on; its
"Post-implementation finding" section paused the weight choice pending this
work); `2026-09-10-mlx-llm-layer-design.md` (Task 7's checkpoint-quality
finding, whose root cause this identifies).

## Problem

pyqmd's query expansion has never worked. The default model,
`after2400/qmd-query-expansion-1.7B-mlx-4bit`, is a 4-bit `mlx_lm.convert`
of `tobil/qmd-query-expansion-1.7B`'s `model.safetensors`. It answers every
prompt with free-form prose, so `Store._parse_expanded_lines` never finds a
`hyde:`/`lex:`/`vec:` line. In effect, `query` has always been
original-query FTS + vector + rerank. Nothing catches the empty result,
because pyqmd has no fallback when expansion produces nothing (Node does).

## Root cause (spike, 2026-09-24)

The source checkpoint is broken; conversion and quantization are not the
cause.

- `tobil/qmd-query-expansion-1.7B`'s `model.safetensors` (repo sha
  `c8ff036`) differs from base `Qwen/Qwen3-1.7B` in only 16 of 311 tensors:
  layers 0–1, `embed_tokens`, and `lm_head`. Layers 2–27 are bit-identical
  to the base. The weights also don't equal base + the LoRA adapter shipped
  in the same repo. It's a partial or failed merge.
- Loaded unquantized (bf16), it still gets 0 of 18 queries into the
  `hyde:`/`lex:`/`vec:` format.
- The prompt isn't the cause. pyqmd, Node (`src/llm.ts`), and the training
  pipeline (`finetune/dataset/prepare_data.py`, `finetune/eval.py`) all use
  `/no_think Expand this search query: <q>` through the Qwen3 chat template
  with `add_generation_prompt=True`.
- There are three distinct "1.7B" checkpoints. Besides the broken
  safetensors, the F16 GGUF inside the same repo is a fully fine-tuned model
  (all 28 layers differ from base). Node's actual default,
  `tobil/qmd-query-expansion-1.7B-gguf` (sha `7816de0`, MIT), is a third:
  all 196 LoRA-targeted matrices differ from that in-repo GGUF, so it comes
  from a separate training run. The earlier HF-default spec's claim that
  the `-gguf` repo holds "the same weights" was wrong.

### Spike measurements

Format check: 18 varied queries (15 from `finetune/evals/queries.txt` plus
3 extra), greedy decoding (pyqmd's current decoding), scored with
`finetune/reward.py:score_expansion`, the finetune repo's own 0–1 rubric.

| Checkpoint (MLX)                         | License     | Queries with `lex:`+`vec:` | Rubric   | s/query |
| ---------------------------------------- | ----------- | -------------------------- | -------- | ------- |
| Current default (after2400, 4-bit)       | –           | 0/18                       | 0.00     | 2.5     |
| `tobil/...-1.7B` safetensors, bf16       | none stated | 0/18                       | 0.00     | 6.1     |
| **Node's `-gguf` f16 → MLX 4-bit**       | **MIT**     | **18/18**                  | **0.72** | **0.3** |
| Node's `-gguf` f16 → MLX bf16            | MIT         | 18/18                      | 0.70     | 0.7     |
| In-repo GGUF → MLX 4-bit                 | none stated | 18/18                      | 0.73     | 0.7     |
| `tobil/...-1.7B-v2` (LoRA merged), 4-bit | Apache-2.0  | 18/18                      | 0.73     | 1.1     |
| `tobil/...-qwen3.5-2B`, bf16             | Apache-2.0  | 16/18, prose mixed in      | 0.00     | 2.1     |
| `bmeyer2025/...-qwen3.5-2B-mlx-4bit`     | Apache-2.0  | 18/18, prose mixed in      | 0.18     | 1.6     |

Node's sampling settings (temp 0.7, top-p 0.8, top-k 20, presence penalty
0.5) gave the same format results as greedy for every 1.7B checkpoint.

Retrieval impact: SciFact parity profile (30 queries with qrels), scratch
index, `Store.query(limit=20)`:

| Expansion                      | MRR     | nDCG@10 | Recall@10 | No rerank: nDCG / Recall |
| ------------------------------ | ------- | ------- | --------- | ------------------------ |
| none (= current default)       | 0.779   | 0.811   | 0.933     | 0.760 / 0.887            |
| Node's `-gguf` weights, 4-bit  | 0.780   | 0.785   | 0.833     | 0.725 / 0.787            |
| _Node qmd's recorded baseline_ | _0.771_ | _0.784_ | _0.867_   | –                        |

With Node's weights, pyqmd lands close to Node's own recorded baseline,
which confirms the diagnosis. Expansion doesn't help SciFact: its queries
are long scientific claims that already work well as search strings, and
with 30 queries one query moves a metric by ~0.03. The goal here is parity
with Node and a working feature, not a SciFact gain.

## Decision

1. **Weights:** convert `tobil/qmd-query-expansion-1.7B-gguf`'s
   `qmd-query-expansion-1.7B-f16.gguf` (pinned to sha `7816de0`) to MLX,
   then quantize to 4-bit affine, group size 64. These are the same weights
   Node ships (Node pulls the `q4_k_m` file from the same repo).
2. **Hosting (outward-facing; needs a separate explicit go-ahead):** replace
   the contents of the existing `after2400/qmd-query-expansion-1.7B-mlx-4bit`
   repo instead of creating a new one. `DEFAULT_EXPAND_MODEL` stays
   unchanged. `mlx_lm.load` → `snapshot_download` checks the Hub for a newer
   revision, so existing installs pick up the fix without a pyqmd upgrade
   (except under `HF_HUB_OFFLINE`).
3. **Safety net:** port Node's post-processing, so a model that emits
   nothing usable degrades the way Node does instead of silently disabling
   expansion.
4. **Decoding unchanged:** greedy, `max_tokens=400`, no grammar. It's
   deterministic, it matches `finetune/eval.py`, and it measured 18/18 on
   the new weights.

   _Superseded 2026-09-25:_ greedy decoding turned out to cost Recall@10 on
   SciFact. Expansion now samples with Node's settings, Node's default system
   prompt, and a per-query seed; see
   `2026-09-25-seeded-expansion-sampling-design.md`.

### Rejected alternatives

- **`tobil/qmd-query-expansion-1.7B-v2`:** no measurable win (same rubric
  score, and slightly higher MRR on SciFact, within the noise of 30
  queries). It would make pyqmd's expansion diverge from Node's.
- **The in-repo GGUF:** no license is stated, and it isn't what Node ships.
- **qwen3.5-2B variants:** they write reasoning prose before the expansion
  lines even with an empty `<think>` block (tobil's full-precision model
  does it too, so it isn't the MLX conversion). The rubric scores them near
  zero, and they're slower.
- **Keep the broken weights and add only the fallback:** expansion would
  still contribute nothing.
- **A new repo id:** it strands every existing install on the broken
  weights until they upgrade pyqmd.

## Design

### Conversion script — `scripts/convert_expand_gguf.py`

`mlx_lm.convert` can't read GGUF, so the conversion needs a committed,
reproducible script (the spike's version lives in a session scratchpad):

1. `hf_hub_download` the F16 GGUF at the pinned revision, and
   `Qwen/Qwen3-1.7B`'s `config.json`, tokenizer files, and chat template at
   a pinned revision (Apache-2.0; the fine-tune reuses the Qwen3
   tokenizer).
2. `mx.load(gguf)` and rename tensors: `token_embd` → `model.embed_tokens`,
   `output_norm` → `model.norm`, `output` → `lm_head`, and
   `blk.N.{attn_norm, ffn_norm, attn_q_norm, attn_k_norm, attn_q, attn_k,
attn_v, attn_output, ffn_gate, ffn_up, ffn_down}` → the matching
   `model.layers.N.*` names. No q/k un-permutation: llama.cpp's Qwen2/Qwen3
   converter doesn't permute.
3. Sanity checks that fail loudly: every base tensor name is present and
   shapes match; `output.weight == token_embd.weight`, so
   `tie_word_embeddings: true` holds and `lm_head` is dropped (if they
   differ, keep `lm_head` and untie).
4. Cast to bf16, write an MLX model dir, then run `mlx_lm.convert -q
--q-bits 4 --q-group-size 64` on it.
5. A smoke test that generates for 3 fixed queries and asserts each output
   parses into at least one `lex:` and one `vec:` line.

The script prints the source revisions and the `mlx-lm` version for the
model card.

### Safety net — expansion post-processing (Store)

In `Store._retrieve_and_fuse`, replace the bare `_parse_expanded_lines`
result with a pure helper (unit-testable, no MLX) that mirrors Node's
`LlamaCpp.expandQuery` + `store.ts:expandQuery`:

1. Parse `hyde:`/`lex:`/`vec:` lines (existing logic).
2. **Query-term filter:** lowercase the query, replace `[^a-z0-9\s]` with
   spaces, and split into terms. Drop any part whose text contains none of
   the terms as a substring. If the query has no terms, keep everything.
3. **Fallback:** if nothing survives, use `hyde: Information about <q>`,
   `lex: <q>`, `vec: <q>`.
4. **Original-query dedup:** drop parts whose text equals the original
   query exactly. pyqmd already searches the original query in both FTS and
   vector, so after a fallback only the `hyde:` line adds a list, as in
   Node.

The strong-signal BM25 shortcut and everything downstream of the parsed
parts are unchanged.

### Code/config changes

- `_constants.py`: the comment on `DEFAULT_EXPAND_MODEL` names the real
  source (`tobil/qmd-query-expansion-1.7B-gguf` via
  `scripts/convert_expand_gguf.py`). The id itself doesn't change.
- A developer checkout's gitignored `models/qmd-query-expansion-1.7b-mlx`
  now holds known-bad weights. Docs say to delete it or leave
  `PYQMD_EXPAND_MODEL` unset.

### Hosting (only after the user's go-ahead)

- `hf upload after2400/qmd-query-expansion-1.7B-mlx-4bit <dir> . --delete
"*"`, so the Hub file set matches the new conversion exactly (no stale
  shards or index).
- Model card: `license: mit`, `library_name: mlx`,
  `base_model: tobil/qmd-query-expansion-1.7B-gguf`; attribution to tobil's
  qmd model and Qwen3-1.7B (Apache-2.0); the source revision, the script,
  and the `mlx-lm` version; and a note that revisions before this commit
  held a broken conversion. Remove the interim known-bad warning (Hub
  commit `02a04a0`).

### Tests

Fast (TDD, no MLX):

- Post-processing helper: parses mixed valid and invalid lines; drops parts
  with no query term; keeps everything when the query has no terms (e.g.
  punctuation only); falls back to the three-line set when nothing parses
  or nothing survives the filter; drops parts equal to the original query
  (so an all-invalid expansion yields exactly one `hyde` part).
- `Store.query` with a stub `expand_fn` returning prose → the fused lists
  include the fallback `hyde` list (checked via `RankedListMeta`).

Slow:

- `tests/test_integration.py::test_expand_query_returns_nonempty_lines`,
  the strengthened format assertion that fails today, should pass. Extend
  it to assert at least one `lex:` and one `vec:` line, and add the
  `requires_expansion_weights` marker (it's currently only `slow`, so it
  would trigger a ~934 MB download on a cold cache).

Parity: `parity/test_quality.py`'s SciFact quality bar keeps passing (spike
numbers are within its 0.05 margin of Node's baseline on every metric).

### Verification

1. Run the conversion script; the smoke check passes.
2. Re-run the spike's 18-query format check against the local output
   directory via `PYQMD_EXPAND_MODEL`: at least 17/18 with `lex:`+`vec:`.
3. `just test-fast`, `just test-slow` (with `PYQMD_EXPAND_MODEL` set to the
   local output directory), `just test-parity`.
4. After the upload: clear the repo from the HF cache, then run
   `pyqmd query` against a scratch `PYQMD_DB` with no override. The model
   downloads from the default id and `status` shows it.

### Docs

- `COMMAND_STATUS.md`: the expansion-model note currently sits on the
  `vsearch` row. Move it to the `query` row, and replace "Known-bad
  weights" with the new source and the fallback behavior.
- `2026-09-24-expand-model-hf-default-design.md`: the "Post-implementation
  finding" section gets a pointer here.
- `2026-09-10-mlx-llm-layer-design.md`: mark the Task 7 checkpoint-quality
  finding resolved, pointing here.
- `CLAUDE.md` / `SKILL.md`: no command-surface change. Only the developer
  note about the stale local `models/` directory, if one exists there.

## Out of scope

- GBNF-constrained decoding and Node's sampling settings (Node needs the
  grammar for its GGUF runtime; the new weights don't need it under greedy
  decoding, and the fallback covers malformed output).
- Caching expansions in `llm_cache` (Node caches them; pyqmd writes nothing
  there today).
- Retrieval tuning (expansion's effect on SciFact is neutral; judging it on
  short keyword queries needs a user corpus via `pyqmd bench`).
- Pinning the Hub revision in code (unchanged from the HF-default spec).
- Switching to v2 or a qwen3.5 model; revisit if tobil publishes a better
  checkpoint.
