# qmd: MLX LLM layer — design

**Status:** Implemented (2026-09-10)
Date: 2026-09-10
Parent: `docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md` (sub-project #1 of 6)

## Purpose

Prove out a Python/MLX replacement for qmd's current Node/`node-llama-cpp`/GGUF
LLM layer (`src/llm.ts`), standalone, before any other part of the rewrite
begins. This is the highest-uncertainty, highest-value piece of the whole
rewrite: until embedding, reranking, and query expansion are shown to work
correctly and at acceptable latency via MLX, nothing else in the roadmap is
worth building.

This sub-project produces a working Python module and a validation harness
that proves it — not a CLI, not a storage layer, not anything wired into the
existing `qmd` command. Those are later sub-projects.

## Research finding: `mlx_embeddings` cannot rerank the plain-text Qwen3-Reranker family

Recorded here so it isn't re-derived. The parent roadmap doc's claim that
`mlx_embeddings` "supports embeddings _and_ reranking, including the
Qwen3-Reranker family" is **wrong for plain-text (non-VL) Qwen3-Reranker
checkpoints**, and this design corrects for it:

- `mlx_embeddings`'s yes/no-logit reranking method (`get_binary_logits()`,
  confirmed present in the actual released `mlx_embeddings==0.1.0` wheel, not
  just an unreleased branch) is implemented only on its `qwen3_vl` model
  class, gated by `config.json`'s `"model_type": "qwen3_vl"`.
- `mlx-community/Qwen3-Reranker-0.6B-mxfp8` (the model this spec targets) has
  `"model_type": "qwen3"` (plain, non-VL) — confirmed by inspecting its
  `config.json`. That routes to `mlx_embeddings`'s separate, embedding-only
  `qwen3.py` model class, which **discards `lm_head.weight` entirely**
  (`sanitize()` explicitly drops it) and only produces last-token-pooled
  embeddings. It has no path to yes/no-logit scoring at all — calling
  `mlx_embeddings.load()` on this checkpoint silently gives you an embedding
  model, not a reranker.
- The parent roadmap's other supporting evidence — `embed-rerank`'s MLX
  backend, cited as a "credible" option supporting
  `vserifsaglam/Qwen3-Reranker-4B-4bit-MLX` — does not actually validate real
  reranking either: its own source (`app/backends/mlx_reranker_backend.py`)
  documents itself as "functional v1... not a full transformer cross-encoder
  (no attention layers yet)" and scores pairs via a mean-pooled embedding
  dotted with a linear head that falls back to a **random vector, seeded by
  a hash of the model name**, whenever no trained head file is present —
  which it isn't, for that model. It does not run the model's transformer at
  all.
- **Fix**: don't route reranking through `mlx_embeddings`. Use `mlx-lm`
  directly instead — `mlx_lm.load()` keeps the LM head intact (it's built
  for generation), so `qmd_llm.rerank()` implements the yes/no-logit scoring
  itself on top of a plain `mlx-lm`-loaded causal LM. This is the same
  technique `mlx_embeddings`'s own `qwen3_vl` code already proves works
  (`token_logits[:, yes_id] - token_logits[:, no_id]`), just applied outside
  that library. See the Python API section below for the exact
  implementation. Net effect: `mlx_embeddings` is used only for the **embed**
  role (where its plain `qwen3`/`gemma3_text` classes are genuinely correct
  embedding implementations); **rerank** and **query-expansion** both go
  through `mlx-lm` directly.

## Non-goals

- No config-resolution system (env vars, per-repo `index.yml` overrides) —
  that's sub-project #3 (CLI command surface), once this module has a real
  caller.
- No concurrent/pooled model contexts for throughput — today's `src/llm.ts`
  has an `LLMSessionManager`/`LLMSession` abstraction for idle-unload and
  concurrent access in a long-running server process; that's only relevant
  once this module is embedded in the MCP/HTTP server (sub-project #4).
- No _shared_ repo history with qmd's own commits. Revised during plan
  review: `python/` is `git init`'d as its own standalone repo, physically
  nested inside this fork's working tree but excluded wholesale from qmd's
  `.gitignore`, so qmd's git never tracks or commits any of it. This
  reverses this design's original "reuse this repo now, split later"
  call — the trigger was wanting qmd's own commit history to stay free of
  Python build-out churn, not a change in the file-location decision (the
  code still physically lives at `python/` inside this fork, per Location &
  tooling below; only its git history is separate, from the start rather
  than deferred to sub-project #6).
- No API-shape mirroring of `src/llm.ts`'s classes/session pattern. The
  _semantics_ (model choices, prefix formatting, scoring method) are ported
  faithfully; the Python surface is designed fresh and idiomatically, since
  shape-mirroring does not restore any actual `git merge` compatibility with
  upstream `tobi/qmd` (already conceded moot — see parent roadmap doc).

## Location & tooling

- New top-level directory: `python/`, git-init'd as its own standalone repo
  (see the Non-goals entry above) and excluded wholesale from qmd's own
  `.gitignore`. It is not a scratch/throwaway location — just a separately
  version-controlled one, physically nested inside this fork.
- Dependency management: `uv` + `pyproject.toml`. Python floor bumped to
  `>=3.12` (from the originally-considered `>=3.10`, which only matched
  `finetune/`'s convention) to match the pinned `.python-version` and dev
  tooling brought over from `<tooling-reference-repo>` (the
  user's reference Python repo for this tooling): `ruff` (lint + format),
  `pre-commit` (generic hygiene hooks + ruff + a local prettier hook for
  markdown/yaml/json), and a `.python-version` file pinning `3.12`.
  Commitlint and semantic-release from that reference repo are deliberately
  _not_ brought over — qmd already has its own top-level release process,
  and enforcing Conventional Commits wasn't asked for here.
- Task running: `python/.Justfile` (dot-prefixed, matching the user's
  personal convention from the reference repo), with `develop` (uv sync +
  install pre-commit hooks), `test`/`test-fast`/`test-slow`, and `lint`
  recipes.
- The LLM layer itself is a subpackage: `python/qmd_llm/`.
- Validation/comparison scripts live in `python/scripts/`.
- Tests live in `python/tests/` (pytest).

## Model choices

| Role            | Model                                                        | Mechanism                                          | Notes                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------- | ------------------------------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rerank          | `mlx-community/Qwen3-Reranker-0.6B-mxfp8`                    | `mlx-lm` (custom yes/no-logit scoring — see below) | Parity with current default (`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`). `vserifsaglam/Qwen3-Reranker-4B-4bit-MLX` stays available as an optional second comparison point (not the default), matching the earlier GGUF 0.6B-vs-4B evaluation.                                                                                                                                              |
| Embed           | `mlx-community/embeddinggemma-300m-*`                        | `mlx_embeddings` (`load()` + forward pass)         | Confirmed to exist and be `mlx_embeddings`-compatible (converted from `google/embeddinggemma-300m`), same family as current default (`ggml-org/embeddinggemma-300M-GGUF`). Exact quantization (likely 8-bit, for parity with current Q8_0) to be confirmed during implementation.                                                                                                         |
| Query expansion | Self-converted MLX build of `tobil/qmd-query-expansion-1.7B` | `mlx-lm` (`load()` + `generate()`)                 | This is qmd's own custom fine-tuned model (see `finetune/CLAUDE.md`), currently deployed as `tobil/qmd-query-expansion-1.7B-gguf`. No third-party MLX conversion exists; convert the merged HF checkpoint (`tobil/qmd-query-expansion-1.7B`) directly via `mlx_lm.convert` (mlx-lm supports the Qwen3 architecture family). This is a mechanical setup task, not an open design question. |

### Rerank scoring implementation

Ported from the official `Qwen/Qwen3-Reranker-0.6B` reference implementation
(same technique `node-llama-cpp`'s `createRankingContext()` uses today, and
the same technique `mlx_embeddings`'s `qwen3_vl.get_binary_logits()` uses —
just run manually on top of `mlx-lm` instead of through that library):

- System prompt: `Judge whether the Document meets the requirements based on
the Query and the Instruct provided. Note that the answer can only be
"yes" or "no".`
- Pair format: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
  (default instruction: `Given a web search query, retrieve relevant
passages that answer the query`).
- Chat wrapping: `<|im_start|>system\n{system prompt}<|im_end|>\n<|im_start|>user\n{pair}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Score: run the wrapped prompt through the `mlx-lm`-loaded model, take
  `logits[:, -1, :]` (final-position logits over the vocabulary), look up
  `token_true_id = tokenizer.convert_tokens_to_ids("yes")` and
  `token_false_id = tokenizer.convert_tokens_to_ids("no")` (looked up
  dynamically per tokenizer, not hardcoded — the official reference does the
  same, and IDs aren't guaranteed stable across tokenizer versions), and
  compute `score = sigmoid(true_logit - false_logit)`.

Faithfully port from `src/llm.ts`, regardless of API shape:

- The nomic-style task-prefix formatting for embedding queries
  (`formatQueryForEmbedding`).
- The yes/no-logit reranking scoring method above (not text
  generation/sampling) — now implemented via `mlx-lm` rather than
  `mlx_embeddings` (see the research finding above).
- Default model choices above.

## Python API (`qmd_llm`)

Fresh, idiomatic design. No config-resolution layer, no session/pooling
abstraction — those are deferred to the sub-project that actually wires this
into a long-running process.

```python
def embed(texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]: ...
def rerank(query: str, documents: list[str], model: str = DEFAULT_RERANK_MODEL) -> list[float]: ...
def expand_query(query: str, model: str = DEFAULT_EXPAND_MODEL) -> list[str]: ...
```

- Each function takes an explicit `model` parameter rather than reading from
  a hidden global/config, so the validation harness can run multiple model
  sizes (e.g. 0.6B vs 4B reranker) side-by-side in the same process.
- Each function lazily loads and caches its model in a module-level dict
  keyed by model id, so repeated calls with the same model id don't reload
  weights. `embed`'s cache and `rerank`/`expand_query`'s cache are separate
  dicts, since they load through different libraries (`mlx_embeddings` vs
  `mlx-lm`) and cache different object types.
- `embed()` is implemented via `mlx_embeddings.load()` + a forward pass
  (this library's plain-text embedding path is correct and used as
  intended).
- `rerank()` and `expand_query()` are both implemented via `mlx_lm.load()` —
  `rerank()` runs the manual yes/no-logit scoring described above,
  `expand_query()` uses `mlx_lm.generate()`. Neither goes through
  `mlx_embeddings` (see the research finding above for why).
- `rerank`'s return shape (list of scores, one per input document, same
  order) matches the semantics of `node-llama-cpp`'s `rankAll` today, so
  score comparison in the validation harness is direct.

## Validation / comparison harness

Goal: prove score/ranking parity and acceptable latency against the current
Node/GGUF reranker, without wiring MLX into qmd's live TS pipeline (that's
sub-project #3's job) and without qmd's `bench` command needing to know
anything about Python or MLX.

Steps:

1. **Fixture export** (`scripts/export-rerank-fixture.ts`, Node,
   throwaway/one-time): runs hybrid retrieval (pre-rerank) against an
   already-indexed BEIR SciFact collection for a fixture of queries, and
   dumps `{query, candidate_texts[], qrels}` per query to a static JSON
   file. This captures real query/document pairs, decoupled from the live
   index, so downstream steps don't need a running qmd store at all.
2. **Baseline scores** (small script alongside the existing
   `src/bench-rerank.ts`, real query/doc pairs from the fixture instead of
   `bench-rerank.ts`'s synthetic timing docs): runs the _current_ Node/GGUF
   Qwen3-Reranker-0.6B over the static fixture, dumps per-query scores.
3. **Candidate scores** (`python/scripts/rerank_fixture.py`): runs
   `qmd_llm.rerank` over the _same_ static fixture, dumps per-query scores.
4. **Compare** (`python/scripts/compare.py`): loads both score sets plus the
   fixture's qrels, computes:
   - Rank correlation (Spearman and/or Kendall tau) between baseline and
     candidate rankings per query.
   - P@k, R@1/3/5, MRR, F1 for both baseline and candidate against qrels —
     the same metrics `qmd bench` already reports (see `src/bench/score.ts`),
     recomputed here so the two are directly comparable without touching
     qmd's TS scoring code.
   - Prints a summary table.

Only step 1 touches existing Node code, and it's a throwaway export script,
not production wiring — steps 2-4 need zero changes to `src/`.

## Success criteria

- **Correctness**: rank correlation and P@k/R@k/MRR/F1 parity (not
  necessarily bit-identical scores, but equivalent ranking quality) between
  the MLX candidate and current Node/GGUF baseline on the SciFact comparison
  fixture.
- **Latency**: median rerank latency for a given document count within
  roughly 2x of the current GGUF reranker's median latency for the same
  count, measured on the same machine (M4 Max / 128GB). Some slowdown is
  acceptable since MLX unblocks reranking quality/model choices GGUF
  couldn't support at all (per the parent roadmap's GGUF reranker findings);
  a large regression would not be.
- **Embed and query-expansion**: sanity-checked via the pytest integration
  test (well-formed output, no NaNs/wrong dimensions) — not put through the
  same rigorous comparison harness as reranking, since reranking is the
  piece with actual architectural risk (logit-based scoring). If embed or
  query-expansion output looks meaningfully wrong during implementation,
  that's a signal to add comparison rigor there too before calling this
  sub-project done.

## Testing

`python/tests/` (pytest):

- Fast unit tests with no model loading: prefix-formatting logic,
  model-id-keyed caching behavior (same id → no reload; different id →
  reload).
- One `@pytest.mark.slow` integration test that loads the real MLX models
  and checks: embed returns vectors of the expected dimension, rerank
  returns scores in a sane range with no NaNs (mirroring the "Verify scores
  are valid" check in `bench-rerank.ts`), and expand_query returns a
  non-empty list of strings.

## Validation results (2026-09-10, corrected after final review)

Ran the full validation harness against a
pre-existing, already-indexed `scifact-eval` collection (5183 BEIR SciFact
documents, 5352 embedded chunks — reused read-only from an earlier session
rather than re-indexing, since its file-naming convention matched exactly)
with 30 queries:

| Metric | Baseline (Node/GGUF) | Candidate (MLX) |
| ------ | -------------------- | --------------- |
| MRR    | 0.819                | 0.819           |
| F1     | 0.900                | 0.900           |
| Recall | 0.900                | 0.900           |

Mean Spearman rank correlation between the two rerankers' rankings (30
queries, 0 excluded as degenerate): **0.911**.

Median latency: baseline 2070.0ms, candidate 4332.5ms — **2.09x ratio**.

_(A final whole-plan review caught that `compare_rerank_fixtures.py`
originally scored the full, unsliced candidate list rather than the
top-k slice — since baseline and candidate rerank the exact same
retrieval-stage candidate set and only reorder it, that made recall/F1
structurally near-incapable of differing between the two systems. Fixed
to slice to `top_k` before scoring, matching what `qmd bench`'s real
backends measure; the table above reflects the corrected numbers. The
original, uncorrected run showed MRR 0.820/0.821 and Recall 0.933/0.933 —
the correction doesn't change the pass/fail call, but the evidence for it
now rests on metrics that could actually have shown a difference.)_

**Call against Success Criteria:**

- **Correctness: PASS.** MRR and F1 are now identical to 3 decimal places
  between the two rerankers, and Recall ties too — for this fixture, the
  relevant document resolves to the same top-k membership and rank
  position under both rerankers whenever it does so at all. The 0.911 mean
  Spearman correlation is the metric that actually demonstrates the two
  rerankers' full orderings are highly similar (not merely tied on a
  top-k-insensitive aggregate) — the MLX reranker is genuinely reproducing
  the same relevance judgments as the current Node/GGUF reranker, not just
  agreeing on which single document ends up first.
- **Latency: borderline PASS.** 2.09x sits just above the "roughly ≤2x"
  target, but is not the "large regression" the criterion was written to
  guard against — this is the expected cost of `rerank()`'s known,
  deliberate simplification (Task 6: one forward pass per document, no
  batching, shipped correctness-first per YAGNI). If lower latency is
  wanted, batching multiple documents into a single `mlx-lm` forward pass
  is the identified next step — not a redesign, an optimization on top of
  already-correct behavior.
- **Embed/query-expansion**: embed's slow integration test passed (Task 5).
  The final whole-plan review caught that `embed()` applied the query-side
  text prefix to every input, including documents — `src/llm.ts` has a
  separate document formatter (`formatDocForEmbedding`) that the original
  "Faithfully port" scope in this spec only named the query side of; fixed
  by porting `format_doc_for_embedding` and adding a `kind: "query" |
"document"` parameter to `embed()` (default `"query"`, preserving prior
  behavior). This was load-bearing to fix now, since sub-project #2
  (storage) is the exact consumer that would silently embed documents
  wrong otherwise. Query-expansion's integration test was strengthened to
  assert the trained `hyde:/lex:/vec:` line format (it previously only
  checked non-blank lines) and now correctly **fails** — the currently
  published `tobil/qmd-query-expansion-1.7B` checkpoint does not produce
  that format (Task 7's finding, unchanged; not a code defect in this
  sub-project). The failing test is the intended outcome: an honest signal
  beats a green test that hides a known-broken checkpoint.

**Resolved 2026-09-24:** root cause was the checkpoint itself —
`tobil/qmd-query-expansion-1.7B`'s `model.safetensors` is a broken merge
(layers 2–27 identical to base Qwen3-1.7B). pyqmd now uses a conversion of
Node's `tobil/qmd-query-expansion-1.7B-gguf`, and the strengthened test
passes; see `2026-09-24-expand-model-gguf-weights-design.md`.

**Overall: sub-project #1's core hypothesis is validated.** The MLX
reranker is a viable, near-drop-in replacement for the current Node/GGUF
reranker on quality; the latency gap is small, understood, and has a known
optimization path if it matters in practice. Sub-project #2 (storage layer)
is unblocked.

## Deferred findings from the final whole-plan review

Recorded so they aren't lost, not because they block sub-project #2:

- Neither reranker scorer (Node baseline or MLX candidate) implements
  production's context-size truncation budget (`src/llm.ts`'s
  `RERANK_CONTEXT_SIZE = 4096` minus template/query overhead) — the current
  SciFact fixture's documents happened to fit, so this hasn't caused a
  wrong result yet, but reusing this harness on a corpus with longer
  documents could silently truncate differently on each side, or throw.
- The rerank fixture uses each document's full body text, not its
  `bestChunk` — production reranks chunks, never full bodies. SciFact's
  documents are single-chunk abstracts so this hasn't mattered here, but
  it means the harness's latency numbers in particular won't transfer to a
  multi-chunk-document corpus without this fix.
- ✅ **Resolved 2026-09-24.** `DEFAULT_EXPAND_MODEL`'s absolute-filesystem-path
  construction (anchored to `__file__`) pointed into a nonexistent path in
  any wheel install. It is now the HF Hub repo id
  `after2400/qmd-query-expansion-1.7B-mlx-4bit`, overridable via
  `PYQMD_EXPAND_MODEL` — see `2026-09-24-expand-model-hf-default-design.md`.
- Several Minor findings (latency-per-document normalization, an
  embed-normalization test gap, three separate `mlx-lm` load caches where
  the spec describes two, non-atomic corpus-zip downloads, an unguarded
  `zipfile.extractall`, a few unhandled empty-input edge cases, and the
  fixture/score JSON files not being committed anywhere for reproducibility)
  — see the final review's full report for detail if picked up later.
- The harness measures agreement between two _prompt-template_
  implementations, not just two numerical backends — `node-llama-cpp`'s
  `rankAll` builds its own internal Qwen template, while `build_rerank_prompt`
  hand-builds one from the official reference. The 0.911 correlation is a
  floor on achievable agreement, not a pure measurement of numerical
  fidelity; nobody has yet confirmed the two rendered prompts are
  byte-identical.

## Open follow-ups at completion

Sub-project #1 was completed with two open, non-blocking follow-ups: (1) whether the `rerank()` latency gap (2.09x)
needs closing before moving on — an early-exit or smaller candidate set
was suggested as a better first lever than batching, since ~43% of scored
pairs sit near zero on both systems — and (2) investigating the
`tobil/qmd-query-expansion-1.7B` checkpoint-quality finding from Task 7
(resolved 2026-09-24, see above).
