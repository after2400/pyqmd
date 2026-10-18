# Contextual embedding: experiment design

Status: run (2026-10-18); result at the end: no feature. Roadmap item: "opt-in `embed
--contextual`" under "Feature ideas from the PyPI name/collision spike". This
is the experiment only; a feature spec follows only if the rule below says so.
Same method as `2026-10-15-heading-breadcrumbs-experiment-design.md`.

## Question

Does a short context line written per chunk by a local MLX model (a sentence
that situates the chunk in its page, as in Anthropic's "contextual retrieval"),
embedded with the chunk, improve retrieval **by more than heading breadcrumbs
already do**?

The bar is breadcrumbs, not plain embedding. Breadcrumbs cost about 2% extra
embedding tokens and no generation. Contextual embedding costs hours of
generation and a full re-embed, so it has to earn that over the free option.

## Arms

All four embed the same chunks (the product's 2,700-character embedding
chunks, `embedding_chunks`) of the 652-page ConditionalQA corpus:

| Arm         | Embedded as                                            |
| ----------- | ------------------------------------------------------ |
| `base`      | `title: <doc title> \| text: <chunk>`                  |
| `crumbs`    | `title: <doc title> > <section path> \| text: <chunk>` |
| `ctx-small` | `title: <doc title> \| text: <context>\n\n<chunk>`     |
| `ctx-mid`   | the same, with the context written by the larger model |

`base` and `crumbs` are exactly the experiment's `base` and `full` variants.
The ctx arms do not use breadcrumbs, so each comparison isolates one change.
Combining the two is a follow-up if contextual wins.

Placement of the context (prepended to the chunk text) is the published
method; putting it in the title field or in a separate vector were considered
and rejected (they mix two changes, or change the index shape).

## Generators

Both are 4-bit MLX models from the Hub (ids verified to exist 2026-10-16):

- **small:** `mlx-community/Qwen3-1.7B-4bit` (about 1.0 GB).
- **mid:** `mlx-community/Qwen3-4B-4bit` (about 2.3 GB).

Same release, same chat template, same 4-bit quantisation, so a difference
between the two arms is a size effect. Both are hybrid thinking models and run
with the chat template's thinking switched off (the output must be the context
line, not reasoning). `Qwen3-4B-Instruct-2507` is a stronger 4B but a different
release, and was rejected for that reason: it would confound size with
release. If contextual wins, trying better generators is a follow-up.

## Generation

- **Prompt**, fixed here, one chat turn: the page text, then the chunk, then
  "Please give a short succinct context to situate this chunk within the
  overall document for the purpose of improving search retrieval of the chunk.
  Answer only with the succinct context and nothing else." (Anthropic's
  published wording, lightly shortened.) The final prompt text is frozen in the
  script before any scored run.
- **Decoding:** greedy (temperature 0), at most 60 new tokens, so output is
  deterministic.
- **Long pages:** a page longer than 18,000 characters (about 6,000 tokens) is
  cut to a window of 18,000 characters centred on the chunk. Measured in
  characters so the cut needs no tokenizer. ConditionalQA pages are mostly
  shorter.
- **Empty output:** a context that is empty after cleaning (reasoning block
  removed, whitespace collapsed) is embedded as the plain chunk, so that
  chunk's ctx vector equals its `base` vector. The count of such chunks is
  reported per generator.
- **Cache:** every generated context is stored on disk, keyed by generator id,
  prompt text, page text and chunk text. The run is resumable (an interrupted
  run continues where it stopped, which is not a second run), logs progress to
  the terminal and a log file, and is run by the user by hand.
- **Timing first:** generation speed is not known. Before the full run, time 20
  chunks per generator; the measured rate goes into this spec as an
  amendment, before any scored result. The sample size is not changed by it.

## Measurement

- Questions, labelling, metrics and bootstrap are the breadcrumbs experiment's:
  the 2,217 train-split questions (strict labelling), chunk MRR, hit@k,
  in-page hit@1, page MRR / nDCG@10 / Recall@10, page-clustered bootstrap
  (10,000 resamples, seed 0). Vector-only first.
- Intervals are **97.5%** (two generator sizes are tested, so each is held to
  a slightly stricter bar).
- Primary metric: chunk-level MRR. Primary contrast: each ctx arm minus
  `crumbs`. Secondary contrast: each ctx arm minus `base`.

## Decision rule (fixed now, before any generation)

Per generator size:

1. **Worth a feature spec:** the lower bound of (ctx − crumbs) chunk MRR is
   above zero **and** page nDCG@10's point estimate is not lower than
   `crumbs`'. The user decides whether the gain justifies the generation
   cost; nothing is decided automatically.
2. **Not worth it:** ctx beats `base` (lower bound above zero) but not
   `crumbs`. Recorded; no feature, since breadcrumbs already give most of the
   gain for free.
3. **No:** anything else.

Regression flag: a point estimate below −0.01 on any metric against `base` is
reported as a regression whatever the rule says.

Only if rule 1 is met for some size are the hybrid (FTS + vector + RRF) and
reranker stages measured, cheapest first, each with its own preregistered
amendment (as breadcrumbs did).

**One scored run.** A defect in the script may be fixed and the run repeated
once, with the defect recorded here. A result that merely disappoints may not.

## Scope

In scope: `scripts/bench_contextual.py` and its tests (reusing the breadcrumbs
script's chunking, labelling, metrics and bootstrap), written test-first; a
10-page smoke run with real models before the full run.

Not in scope: any change to `Store`, the CLI or the embedding fingerprint;
putting the context into full-text search (the published method does that too;
test it only after a vector-stage win); non-markdown files; other corpora;
query expansion.

## Risks

- Generation time may be too long for the mid model. The timing run decides
  whether it is run at all; if it is dropped, that is recorded as an
  amendment, and the question is answered for the small model only.
- ConditionalQA pages are government guidance pages. A result here may not
  carry to other kinds of notes; the same caveat applied to breadcrumbs.
- A context line can state things the page does not say (the model may be
  wrong). The experiment measures retrieval, not the lines' accuracy; a feature
  spec would need to address that.

## Amendment 1: timing (2026-10-16, before any scored result)

Measured on the first 20 chunks of the full corpus (2,566 chunks, 652 pages),
greedy, ≤60 new tokens, thinking off: small (Qwen3-1.7B-4bit) 0.9 s/chunk, mid
(Qwen3-4B-4bit) 2.0 s/chunk. Projected full generation: small about 40 minutes,
mid about 85 minutes. Both are well under the 8-hour limit, so both sizes stay
in. A 10-page smoke run (40 chunks, 200 resamples) ran the whole
generate → embed → score path without error; its numbers are not a result and
are discarded. The 20 timing chunks stay in the context cache and are reused by
the full run.

## Result (2026-10-18, the one scored run)

Run once, as preregistered: 2,217 questions (29 ambiguous excluded), 2,566
chunks, 652 pages, 97.5% page-clustered intervals, 10,000 resamples, seed 0.
Generation finished cleanly for both sizes (small 1.0 s/chunk, about 41 minutes;
mid 2.1 s/chunk, about 90 minutes); no chunk had an empty context. No script
defect, so no rerun.

Vector stage, means (base / crumbs / ctx-small / ctx-mid):
chunk MRR 0.4974 / 0.5216 / 0.5161 / 0.5149; page nDCG@10 0.6243 / 0.6350 /
0.6321 / 0.6321.

| Size  | chunk MRR, ctx − crumbs (97.5% CI) | page nDCG@10, ctx − crumbs | chunk MRR, ctx − base      |
| ----- | ---------------------------------- | -------------------------- | -------------------------- |
| small | −0.0055 [−0.0196, +0.0088]         | −0.0029 [−0.0132, +0.0075] | +0.0186 [+0.0083, +0.0290] |
| mid   | −0.0067 [−0.0197, +0.0066]         | −0.0029 [−0.0122, +0.0064] | +0.0175 [+0.0066, +0.0287] |

**Verdicts: small rule 2, mid rule 2. No feature spec.** Rule 1 needed the lower
bound of ctx − crumbs chunk MRR above zero; both lower bounds are about −0.02,
and both point estimates are slightly below zero. No regression flag against
base (no point estimate below −0.01). Hybrid and reranker stages are not
measured, as preregistered.

What this shows: at the vector stage on this corpus, a model-written context
line improves chunk MRR over plain embedding (interval above zero for both
sizes), but does not beat the free heading-path breadcrumbs. What it does not
show: that contexts are useless. The comparison is one corpus of government
guidance pages, one prompt, greedy 60-token lines, two small 4-bit models, and
embedding only (the published method also puts the context into full-text
search, which was out of scope). An interval of roughly ±0.014 on chunk MRR
cannot exclude a small gain over breadcrumbs, only a large one. The 4B model did
not beat the 1.7B model.

Consequence: breadcrumbs stay the only opt-in title enrichment; the contextual
line is shelved. Deferred minors of the script are listed in the PR.
