# Opt-in cross-encoder reranker backend (IBM Granite) — design

**Status:** Implemented (2026-11-04), written against v0.9.1. The default stays Qwen3-Reranker;
Granite is opt-in. Calibration is fitted by median matching (see §3 and "Evaluation results");
the SciFact intent loss did not repeat on ConditionalQA (§5). Open: a `pyqmd bench` run on a real
notes corpus before any default flip.
**Related:** `docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`
(model-defaults backlog); `docs/specs/2026-09-24-expand-model-hf-default-design.md`
(the `PYQMD_EXPAND_MODEL` precedent); `docs/specs/2026-09-24-qrels-margin-calibration-design.md`
(calibration tooling to reuse).

## Why

Motivation: a model set whose weights come from US organisations. Today:

| Role            | Default                     | Maker        | US-made? |
| --------------- | --------------------------- | ------------ | -------- |
| Embedding       | `embeddinggemma-300m-8bit`  | Google       | yes      |
| Rerank          | `Qwen3-Reranker-0.6B-mxfp8` | Alibaba      | **no**   |
| Query expansion | fine-tune of Qwen3-1.7B     | Alibaba base | **no**   |

This spec covers the reranker only. Expansion needs its own spec (re-run the
SFT in the sibling Node repo's `finetune/` on a US base; Gemma 4 E2B/E4B,
now Apache 2.0, is the leading candidate). The embedding model needs no change,
and keeping it avoids a re-index.

## Scope

In: a second rerank backend (BERT-style cross-encoder), config-driven backend
choice, a `PYQMD_RERANK_MODEL` override, score calibration, tests, docs.

Out: changing the default rerank model (decided after the evaluation below),
changing the expansion model, changing the embedding model, an upstream
`mlx-embeddings` release.

## Candidates considered (state as of 2026-11-04)

- **`ibm-granite/granite-embedding-reranker-english-r2` — chosen.** 149M-parameter
  ModernBERT cross-encoder, Apache 2.0, 8192-token context, English only.
  Published BEIR avg 55.0–55.8 (comparable to `gte-reranker-modernbert-base`
  at 54.8–56.1, above `bge-reranker-large` at 53.0–54.3). Loads in the
  `mlx-embeddings` version pyqmd already pins.
- **NVIDIA `Nemotron-3.5-Rerank-8B`** (Oct 2026). Rejected: built on Mistral's
  Ministral-3-8B, so the base model is not US-made; 8B parameters (~5 GB at 4-bit);
  the only MLX conversion is third-party and needs a custom loader; throughput
  falls roughly tenfold on 4K-token inputs; OpenMDW-1.1 license.
- **NVIDIA `llama-nemotron-rerank-1b-v2`.** Llama 3.2 1B base, so US lineage.
  Fallback if Granite disappoints. `mlx-embeddings` ships a `llama_bidirec`
  model, so a port may be cheap. Not evaluated.
- **Zero-shot yes/no scoring with an instruct LLM (Gemma 4, Granite 4).**
  Reuses the existing code path, but small non-reranker models usually score
  well below a trained reranker. Not evaluated; only worth it if both
  cross-encoders fail.
- Nothing new from Snowflake, Nomic or Google for reranking.

## Spike results

Done 2026-11-04 with throwaway scripts (recreate from this description).

**Port fidelity.** `mlx_embeddings.utils.load` loads the checkpoint as
`ModernBertForSequenceClassification` with no conversion. Against a PyTorch
(`transformers` 5.17) reference on four query/passage pairs:

| Pair                 | PyTorch logit | MLX, SiLU head | MLX, library GELU head |
| -------------------- | ------------- | -------------- | ---------------------- |
| password, relevant   | 2.411         | 2.411          | 2.326                  |
| password, irrelevant | 0.754         | 0.754          | 0.830                  |
| tides, relevant      | 2.793         | 2.793          | 2.650                  |
| tides, irrelevant    | 0.688         | 0.688          | 0.772                  |

The backbone is right. The one defect is in `mlx-embeddings` 0.1.0:
`ModernBertPredictionHead` hardcodes GELU, while the checkpoint's
`classifier_activation` is `silu`. Patching the head activation to SiLU
reproduces PyTorch to 3–4 decimals. (The classifier is also built with
`bias=True` regardless of `classifier_bias`; harmless here because the
checkpoint carries `classifier.bias`.)

**Calibration.** Irrelevant pairs score logit ≈ 0.7 (sigmoid ≈ 0.67–0.70), not
≈ 0. This is the model's real calibration, not a port bug (PyTorch gives the
same). It matters for blending; see Design.

**Ranking smoke test.** 22 natural-language queries, each with one correct
document among 32 documents (this repo's published specs, title plus first
1,800 characters). Queries were written by hand, so treat this as a smoke test.

|                                 | top-1 | top-3 | MRR   | speed      |
| ------------------------------- | ----- | ----- | ----- | ---------- |
| Granite, GELU (library default) | 0.86  | 1.00  | 0.917 | 20 ms/pair |
| Granite, SiLU (fixed)           | 0.86  | 1.00  | 0.917 | 19 ms/pair |
| Qwen3-Reranker-0.6B (current)   | 0.91  | 0.95  | 0.939 | 54 ms/pair |

Roughly tied; the gap is one query, and Qwen ranked one target 6th where
Granite ranked it 3rd. GELU vs SiLU made no ranking difference here, but we fix
it anyway: the logits are wrong, and calibration constants should be fitted to
correct ones.

## Design

### 1. Backend selection by checkpoint config, not model name

`llm/rerank.py` reads the model's `config.json` (`architectures`) once at load:

- `*ForSequenceClassification` → **cross-encoder backend** (new).
- otherwise (causal LM) → existing **yes/no-logit backend**, unchanged.

Name regexes (as `is_qwen3_embedding_model` uses for embeddings) are avoided:
config is authoritative and works for any future checkpoint. Public signature
`rerank(query, documents, model) -> list[float]` is unchanged, so `Store`,
`bench` and the MCP server need no changes. The per-model cache holds a small
backend object.

### 2. Cross-encoder backend

- Load with `mlx_embeddings.utils.load` via `load_quietly_if_cached`; imports
  stay deferred, like the other backends, so non-rerank commands don't pay for
  them.
- If `config.classifier_activation == "silu"` and the head's activation is not
  already SiLU, replace it. Make this a no-op once upstream is fixed. Isolate it
  in one small function with a comment naming the upstream bug.
- Encode `(query, document)` as a pair, truncating the document only
  (`only_second`) at a cap of 2048 tokens (checkpoint supports 8192, chunks are
  far shorter). The Qwen backend's note that batching with padding reordered
  results applies to a different architecture; start unbatched (19 ms/pair), then
  try batching as a measured optimisation, accepting it only if scores match
  unbatched within 1e-3.
- Output is the raw logit. Convert to the score scale in step 3.
- Check whether casting F32 weights to fp16/bf16 (or quantizing to 8-bit)
  preserves scores before considering it. The F32 checkpoint is 598 MB.

### 3. Score calibration (the real design question)

`Store.hybrid_query` blends `(rrf_position_weight + rerank_score) / 2` and
applies `--min-score` to that blend. Both assume a rerank score near 0 for
irrelevant and near 1 for relevant (Qwen3-Reranker behaves that way).
Granite's raw sigmoid sits at ≈ 0.67 for irrelevant and ≈ 0.92–0.94 for
relevant, which compresses the rerank signal to a quarter of its range inside the
blend and shifts every `--min-score` threshold.

Decided: the backend returns `sigmoid((logit − b) / t)` with per-model
constants `b` (bias) and `t` (temperature) stored with the model (a small table
keyed by repo id, defaulting to `b=0, t=1` for unknown checkpoints). `b, t` are
fitted by **median matching** on a labelled set: the median relevant and median
other logit are mapped to Qwen3-Reranker's median relevant and median other
score. The qrels-margin-calibration tooling compares pyqmd with Node end to end
and can't score a reranker alone, so `scripts/eval_rerankers.py` does the fit on
SciFact. Rejected: per-query min-max normalisation, which makes `--min-score`
meaningless across queries; and **Platt scaling** (tried first), which
calibrates to the labelled set's ~3% relevant prevalence and gave relevant
documents a median score of 0.295 against Qwen's 0.995, flattening the rerank
half of the blend.

### 4. Configuration

Add `PYQMD_RERANK_MODEL` (Hub repo id or local MLX model dir) and
`resolve_rerank_model()`, mirroring `PYQMD_EXPAND_MODEL` / `resolve_expand_model`;
`Store` uses it when no `rerank_model` argument is given. `pyqmd query` and
`pyqmd bench` also take `--rerank-model <model>` (passed to the `Store` through
`get_store(rerank_model=...)`), which beats the env var; the flag makes a
per-run A/B of two rerankers possible without touching the environment. A load
failure is a one-line `Error:` + exit 1 (CLI/`bench`) / `is_error` (MCP), as for
the expansion model; add a `RerankModelError` for that.

### 5. Intent

The store currently prepends intent to the query (`f"{intent}\n\n{query}"`),
matching Node. A cross-encoder has no instruction channel, so the intent text
becomes part of the pair's first segment. Check whether this helps or hurts on
the evaluation set; if it hurts, the cross-encoder backend ignores intent for
scoring (intent still affects expansion and chunk selection). Decide with data.

**Outcome (2026-11-04): left as is; the loss did not repeat on real intents.**
On SciFact, with the synthetic prefix "biomedical research abstracts", Granite's
nDCG@10 fell 0.022 (0.7988 → 0.7764). On ConditionalQA, where every question
comes with its own scenario as the intent, adding the scenario _raised_ nDCG@10
for both rerankers: Qwen +0.080 [+0.054, +0.107] (rerank-only) and +0.046
[+0.022, +0.069] (blended with BM25 position); Granite +0.061 [+0.033, +0.089]
and +0.044 [+0.020, +0.069]. So a cross-encoder using prefixed intent is fine in
practice, and the SciFact drop reflects a generic, uninformative prefix on a
corpus that did not need one. Nothing to change in `Store`.

## Evaluation (gate for flipping the default)

1. SciFact via `scripts/prepare_scifact_corpus.py` and
   `scripts/compare_rerank_fixtures.py` / `score_rerank_fixture.py` (needs a
   dataset download; do it on request).
2. `pyqmd bench` on a real personal notes corpus with a hand-written fixture,
   `--samples 5`, comparing full-reranked nDCG@10 / MRR / recall@k for both
   rerankers.
3. Latency per query (candidate count × per-pair time) on a warm model.
4. Pass criterion, to confirm before running: Granite is no
   worse than Qwen3 by more than a margin set from the sampling spread in `bench`.

## Evaluation results (2026-11-04)

Item 1 only (SciFact, reranker alone); items 2–4 are the maintainer's.
`scripts/eval_rerankers.py`: 300 SciFact test queries, per query the 30 best BM25
documents (pure-Python BM25, OR semantics, because pyqmd's FTS ANDs every query
term and a long claim matches nothing) plus the judged-relevant ones, so both
rerankers see the same 9,051 pairs. The old `rerank-fixture.json` has no generator
in the repo, hence the new script.

|                     | MRR    | nDCG@10 | recall@10 | s per 30 docs |
| ------------------- | ------ | ------- | --------- | ------------- |
| Granite (F32)       | 0.7650 | 0.7988  | 0.9287    | 0.7–0.9       |
| Qwen3-Reranker-0.6B | 0.7620 | 0.7996  | 0.9493    | 1.2–1.5       |

A tie on ranking quality (Granite loses 0.02 recall@10), 1.4–2× faster. Fitted
calibration `(b, t) = (1.237, 0.0709)`: calibrated medians relevant 0.9954 / other
0.0025 (Qwen 0.9954 / 0.0025); share above 0.5 relevant 0.752 / other 0.073 (Qwen
0.788 / 0.104). Dtype: casting to bf16 or quantizing to 8 bits moved logits by
≤ 0.017 but saved ≤ 6% time and reordered 25/30 and 16/30 query rankings (top-1
unchanged), so Granite stays F32 and no converted copy is hosted. Batching was not
attempted. Not measured: `pyqmd bench` on a real notes corpus (left to the maintainer),
and non-English text (Granite is English-only).

### ConditionalQA (gov.uk guides), added 2026-11-04

An exploratory script (not committed) on the 285 dev questions: 652 gov.uk pages
chunked as `pyqmd embed` chunks them (2,566 chunks); per question, the BM25 top-30
chunks as the shared pool (the gold page is in the pool for 239/285, 83.9%); each
reranker scores the chunk text through the real `rerank()`; a page ranks by its best
chunk. "Blended" averages a 1-list RRF-style position weight with the reranker score,
as `Store.query` does (a proxy: no vector search, no expansion). nDCG@10, with and
without the question's scenario as the intent:

| ranker              | no intent, rerank-only | no intent, blended | scenario, rerank-only | scenario, blended |
| ------------------- | ---------------------- | ------------------ | --------------------- | ----------------- |
| BM25 position only  | 0.5984                 |                    |                       |                   |
| Qwen3-Reranker-0.6B | 0.6949                 | 0.6920             | 0.7752                | 0.7374            |
| Granite             | 0.7051                 | 0.6808             | 0.7660                | 0.7246            |

Granite minus Qwen, paired bootstrap 95% CI: every difference is within ±0.016
and every interval includes zero (rerank-only +0.010 [−0.010, +0.029] without
intent, −0.009 [−0.027, +0.008] with; blended −0.011 [−0.027, +0.005] and −0.013
[−0.032, +0.006]). Per question: Qwen 1.6–1.8 s, Granite 0.7–1.0 s. With 285
questions an interval is about ±0.02, so a true loss of up to ~0.03 nDCG@10 in
the blended setting is not excluded. The blended runs are slightly below Granite's
rerank-only ones for both models, so the calibration (fitted on SciFact) transfers
without an obvious scale problem.

## Testing

- Unit (fast): backend dispatch from a fake config; score mapping with `b, t`;
  SiLU patch applied once and a no-op when already SiLU; env-var resolution and
  error path; truncation of long documents. No model downloads.
- Fidelity test: store the four reference logits from the table above as
  constants (no `torch` dependency) and assert the MLX backend matches within
  1e-3. Marked `slow` since it loads the real model.
- Existing rerank/store tests keep passing unchanged (Qwen backend untouched).

## Docs and bookkeeping

- `COMMAND_STATUS.md`: note on the `query` row (and `bench`) about the opt-in
  reranker and its calibration status.
- `CLAUDE.md` Architecture section: document `PYQMD_RERANK_MODEL` next to
  `PYQMD_EXPAND_MODEL`. Also `src/pyqmd_mlx/skills/pyqmd/SKILL.md` if the env var
  is user-facing (the CLAUDE.md "keep in sync" rule covers commands and flags; an
  env var is borderline, so check).
- Commit type `feat(llm)`. Source under `src/pyqmd_mlx/` releases.
- `README`: Apache 2.0 attribution note for IBM Granite if it's ever the default.
- Upstream: send the one-line `mlx-embeddings` fix (read
  `config.classifier_activation` in `ModernBertPredictionHead`). `pyproject.toml`
  pins `mlx-embeddings==0.1.0` exactly, so the local patch stays until that pin moves.

## Risks and open questions

- Calibration constants are fitted on one corpus and may not transfer. Mitigate by
  fitting on more than one set, as in the qrels-margin-calibration spec.
- English only. Non-English corpora should keep Qwen3-Reranker; say so in docs.
- No instruction-following (see Intent).
- The smoke test is small and self-written; the evaluation gate exists because of that.
- Whether to re-host a smaller (fp16/8-bit) MLX conversion on the Hub, as was done
  for the expansion model, depends on the dtype check in the cross-encoder section.
- The expansion model (the other non-US piece) is unaddressed here by design.
