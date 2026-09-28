# Query expansion: re-quantize the hosted model to a mixed 4/6-bit recipe

**Date:** 2026-09-25
**Status:** Implemented (2026-09-25)
**Related:** `2026-09-10-python-mlx-rewrite-roadmap.md` (backlog entry
"Re-quantize the query-expansion model");
`2026-09-25-expansion-output-grammar-design.md` (its run-on spike traced
the gap to quantization);
`2026-09-24-expand-model-gguf-weights-design.md` (the current conversion
script); `scripts/replay_query.py` (the replay harness).

## Problem

Under Node's output grammar, EOS is allowed only after a newline, so the last
`hyde:` line sometimes runs on to the 600-token cap. pyqmd's hosted model,
`after2400/qmd-query-expansion-1.7B-mlx-4bit`, does this about twice as
often as Node: 73 of 280 expansions in the 10-salt replay against Node's
~13%, and a mean expansion time of 1.37s against Node's 0.48s. The grammar
work's spike ruled out the prompt and the sampler and found that unquantized
weights come close to Node's rate, so the cause is quantization.

Node's `q4_k_m` file averages 5.03 bits per weight. It keeps `attn_v`,
`ffn_down` (on selected layers), and the output head at Q6_K. pyqmd's
conversion is a uniform 4-bit affine quantization (group size 64) with the
output head tied to the 4-bit embedding.

## Evidence (spike, 2026-09-25)

Each candidate was converted from the same f16 GGUF and measured with the
10-salt replay (`--expand seeded:1` … `seeded:10`, 28 queries each) using
the shipped sampler (Node's settings plus presence penalty 0.5).

| weights                                         | size   | run-ons      | mean time | Recall@10 | MRR   | nDCG@10 |
| ----------------------------------------------- | ------ | ------------ | --------- | --------- | ----- | ------- |
| current: 4-bit, tied                            | 0.95GB | 73/280 (26%) | 1.37s     | 0.870     | 0.784 | 0.798   |
| A: `mixed_4_6`, untied 6-bit `lm_head`          | 1.2GB  | 19/280 (7%)  | 0.91s     | 0.896     | 0.778 | 0.799   |
| **B: `mixed_4_6` layers, tied 6-bit embedding** | 1.0GB  | 23/280 (8%)  | 0.90s     | 0.874     | 0.782 | 0.797   |
| C: plain 8-bit, tied                            | 1.7GB  | 30/280 (11%) | 1.28s     | 0.881     | 0.791 | 0.805   |
| B without presence penalty                      | 1.0GB  | 30/280 (11%) | 0.94s     | 0.857     | 0.773 | 0.785   |
| Node `q4_k_m` (grammar spike)                   | —      | 18/140 (13%) | 0.48s     |           |       |         |

- The 6-bit output head and the 6-bit `v_proj`/`down_proj` layers are what
  matter. Uniform 8-bit does worse on run-ons than either mixed recipe and
  is slow, because each token reads twice the bytes.
- A and B are within noise of each other; B is 200MB smaller. The f16 GGUF's
  output head equals its embedding, so B stores that matrix once at 6 bits
  where Node stores it twice (Q4_K embedding, Q6_K output).
- Retrieval quality is within salt-to-salt noise across all recipes.
- The remaining ~0.4s gap to Node is per-token overhead in Python/MLX, not
  run-ons.

### Decision: keep the presence penalty

Without it B scores slightly lower on all three metrics (each difference is
about 1–1.5× the salt noise) and runs on more (30 vs 23). qmd passes the
penalty; node-llama-cpp 3.20 only skips it because no repeat penalty is set.
pyqmd keeps following qmd's stated settings, as `COMMAND_STATUS.md` already
records.

## Design

### 1. `scripts/convert_expand_gguf.py` builds recipe B

The script keeps its GGUF → bf16 MLX steps and changes only the quantization
call. A new pure function decides each module's bits:

```python
def quant_bits(path: str, num_layers: int) -> int:
    """6 for the tied embedding/output head, and for v_proj/down_proj on the
    layers llama.cpp's Q4_K_M keeps at Q6_K; 4 for everything else."""
```

The layer rule is mlx-lm's `mixed_4_6` rule (itself a copy of llama.cpp's
`use_more_bits`): layer `i` gets more bits when
`i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2`. The embedding
path (`model.embed_tokens`) always gets 6. mlx-lm's built-in recipe can't be
used directly because it raises only an `lm_head` module, which a tied model
doesn't have.

`num_layers` comes from the base `config.json` (`num_hidden_layers`), not a
constant. The quantization call becomes
`convert(..., quantize=True, q_group_size=64, q_bits=4,
quant_predicate=<closure returning {"group_size": 64, "bits":
quant_bits(path, n), "mode": "affine"}>)`. Tying is unchanged: the script
still drops `lm_head.weight` when it equals the embedding and sets
`tie_word_embeddings` from that check.

The module docstring, the model card title and body, and the card's usage
snippet change from "4-bit" to "mixed 4/6-bit (llama.cpp `Q4_K_M` layout)"
and to the new repo id. The card's Provenance section describes the recipe.
Its paragraph about the broken safetensors merge goes, since that history
belongs to the old repo. The smoke check is unchanged.

No recipe flag: the script exists to build the hosted model.

### 2. Hosting and default model

- New Hub repo: `after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6`.
- `DEFAULT_EXPAND_MODEL` in `src/pyqmd_mlx/llm/_constants.py` points at it,
  with its comment updated; `tests/test_expand_model.py` follows.
- The expansion seed hashes the model id, so every query's draw changes.
  That's expected; nothing pins a specific expansion's text.
- The old repo `after2400/qmd-query-expansion-1.7B-mlx-4bit` is deleted
  after the merge. pyqmd has never been published to PyPI, so no install
  depends on it.

Order of operations:

1. Build the model locally with the script.
2. Run the acceptance replay against the local dir (`PYQMD_EXPAND_MODEL`).
3. Upload to the new repo (**needs the owner's go-ahead**).
4. With the default switched to the new id, run the slow suite and
   `just test-parity` against the Hub copy. The shipped seed hashes the Hub
   id, so these numbers are only final after the upload.
5. Merge.
6. Delete the old repo (**needs the owner's go-ahead**).

### 3. Acceptance (manual, once)

10-salt replay on the built model, shipped sampler:

- run-ons ≤ 37 of 280 (~13%, Node's rate);
- mean expansion time ≤ 1.0s;
- 0 malformed;
- salt means above the grammar acceptance gates: Recall@10 0.851, MRR
  0.760, nDCG@10 0.774.

Then `just test-slow` and `just test-parity` pass against the Hub id. The
parity Recall@10 threshold is 0.8261 and B's lowest salt was 0.827, so the
shipped draw could land close to it. If it fails, stop and report; don't
tune the seed or the recipe to pass.

**Outcome (2026-09-25):** the built model missed the 10-salt Recall@10 gate
(0.845), although its weights are byte-identical to spike B (0.874). The
replay seed hashes the model id, and a local `PYQMD_EXPAND_MODEL` path is the
id, so the same weights at another path draw other expansions. A 10-salt mean
is only good to about ±0.008 Recall@10, less than the gaps these gates try to
resolve. A 30-salt comparison against the old model on the same salts
put the new model 0.011 Recall@10
behind (about 2× the standard error), with run-ons down from 25% to 7%. The
owner accepted that trade-off. Future expansion-model gates should use 30
salts, not 10.

### 4. Docs

- `CLAUDE.md` Architecture: repo id and "4-bit MLX conversion" wording.
- `COMMAND_STATUS.md` `query` row: repo id and wording; the run-on sentence
  gets the new count and loses the "higher rate comes from the 4-bit
  weights" clause. The presence-penalty note stays.
- `parity/README.md`: the pyqmd column and the ten-seed sentence use the
  new model's numbers.
- Roadmap: the backlog entry is marked done with the acceptance numbers.
  The 2026-09-24 "Next step" entry that names the old repo is history and
  stays as written.
- The Hub model card is rendered by the script, so it stays in step.

## Testing

- `tests/test_convert_expand_gguf.py`: `quant_bits` returns 6 for
  `model.embed_tokens`; for `v_proj`/`down_proj` on layers 0, 5, and 27
  with `n = 28` (first eighth, `(i - n//8) % 3 == 2`, last eighth); 4 for
  those on layers 3 and 4; and 4 for `q_proj`, `gate_proj`, and the like.
  `test_model_card_states_license_source_and_provenance` checks the new repo
  id and recipe wording.
- `tests/test_expand_model.py`: the default id.
- Slow and parity suites as in the acceptance section.

## Out of scope

- A recipe flag on the conversion script.
- Closing the remaining latency gap to Node (Python/MLX per-token overhead).
- Changing the presence penalty.
- Re-quantizing the embedding or rerank models.
