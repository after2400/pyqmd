# Heading breadcrumbs — experiment design

**Status:** Run (2026-10-15). A preregistered retest on the train split (Amendment 1) says yes: breadcrumbs improve vector retrieval on ConditionalQA. The product change is a separate spec.

**Related:** the roadmap's "Feature ideas from the PyPI name/collision spike"
entry (per-chunk heading breadcrumbs); baseline is the format set by
`2026-09-29-embedding-input-parity-design.md` (merged as #16).

## Question

Does prepending a chunk's section-heading path to the text pyqmd embeds
make vector search find the right chunk more often?

Today each chunk is embedded as `title: <doc title> | text: <chunk>`
(Node's format). A chunk from deep inside a long page ("Set it to 30
seconds" under `Configuration > Timeouts > Retry`) says nothing about
where it sits. The candidate format is
`title: <doc title> > <h2> > <h3> | text: <chunk>`.

This is an **experiment**: its output is an answer and a number for the
roadmap, not a feature. **No product code changes.** Nothing under `src/`
is touched: no flag, no `Store` hook, no fingerprint change, no CLI or MCP
surface. If the answer is yes, building it is a separate spec.

## Why ConditionalQA, not SciFact

SciFact abstracts have no headings, so a breadcrumb would only repeat the
title. ConditionalQA (`data/conditionalqa/`, prepared by
`scripts/prepare_conditionalqa_corpus.py`) is 652 gov.uk pages converted to
markdown with a `#` page title and `##`/`###`/`####` sections, and its
dev questions come with the evidence sentence that answers them. That makes
two things measurable:

- which **page** a question belongs to (page-level qrels, as in the parity
  quality test);
- which **chunk** of that page holds the answer (new here), which is the
  thing breadcrumbs should change.

## Measurement problem this design is built around

The parity quality test scores pages from 59 queries, one per page. At
that size one query is worth 0.017 Recall@10, and the previous embedding
change moved MRR by 0.010 and nDCG@10 by 0.003 ("flat"). A page is a hit
if any of its chunks ranks, so a better _matching chunk_ may not show up
at page level at all. So:

- the primary metric is **chunk-level**, over **all answerable dev
  questions** (about 285 minus unanswerable and unmatched ones), not 59;
- the 59-query set is kept only as a page-level regression guard;
- the decision rule is fixed below, before any number is seen.

## Design

One standalone script, `scripts/bench_breadcrumbs.py`, in the style of the
other `scripts/*.py` files, plus fast unit tests for its pure helpers
(`tests/test_bench_breadcrumbs.py`, in the style of
`tests/test_prepare_conditionalqa_corpus.py`). It uses existing library
code only: `pyqmd_mlx.store._chunking.embedding_chunks`,
`pyqmd_mlx.store._title.extract_title` and `pyqmd_mlx.llm.embed`. It never
opens the user's index or any SQLite database; vectors live in memory (and
in a gitignored on-disk cache, below).

### 1. Chunks

For each page: `embedding_chunks(content, path)`, giving `(text, pos)`
pairs at the same Node embed-time size the real `embed` uses. Both
variants embed exactly the same chunks, so only the title string differs.

### 2. Variants

| Variant | `title` passed to `embed(kind="document")`                                                   |
| ------- | -------------------------------------------------------------------------------------------- |
| `base`  | `extract_title(content, path)` (what `embed` stores today)                                   |
| `full`  | `<doc title> > <h2> > <h3> …`: the doc title, then every heading active at the chunk's start |
| `last`  | `<doc title> > <innermost heading>` (exploratory)                                            |

`full` is the **only** variant the decision rule applies to. `last` is
reported for information and cannot ship on its own; it exists because a
long path could dilute short chunks, and one extra column is cheap.
Reporting a second variant without a rule for it would invite picking
whichever looks better, hence the restriction.

**Heading path of a chunk (amended 2026-10-15 after run 1):** ATX headings
(`^#{1,6}\s+…`), ignoring `#` lines inside code fences (`find_code_fences`),
kept as a stack: a heading pops every entry of the same or deeper level
first, and a heading at a position opens its own section. The page's `#`
line is the doc title, so it is not repeated. A chunk's path is the path of
the section that holds the most of its characters (a tie goes to the earlier
section). The original rule, the path at the chunk's start, was wrong:
the chunker starts a chunk 405 characters before the heading it cuts at, so
the start usually lies in the previous section (see Results, run 1).

`embed()` takes one `title` per call, so the script groups a page's chunks
by title string and calls `embed` once per distinct title. Padding is
masked (the embedding input parity work, #16), so batch composition does
not change vectors.

### 3. Queries and gold labels

Source: `data/conditionalqa/raw/dev.json` (285 questions) and the
prepared corpus.

- **Eligible:** `not_answerable` is false and `evidences` is non-empty.
- **Gold page:** the question's `url` slug.
- **Gold chunks:** each evidence item is normalised the way
  `prepare_conditionalqa_corpus` writes it (`html.unescape`, whitespace
  collapsed, `- ` prefix for `<li>`), located in the page's markdown, and
  every embed chunk whose `[pos, pos + len(text))` overlaps a located span
  is gold. An evidence item that can't be found is dropped; a question
  with none left is excluded. The script reports how many questions and
  evidence items were excluded and why, and fails if more than 10% of
  eligible questions are excluded (the matching is then suspect).
- **Query text:** the question only, embedded with `kind="query"`. The
  scenario/intent is not used: it feeds query expansion and reranking,
  which this experiment deliberately leaves out.
- **Search:** cosine similarity of the query against every chunk vector
  in the corpus, in memory. No FTS, no RRF, no expansion, no rerank.

### 4. Metrics

All computed per query, per variant:

- **Chunk-level (primary):** reciprocal rank of the first gold chunk among
  all chunks in the corpus; hit@1, @5, @10.
- **In-page hit@1:** among the gold page's own chunks only, is the top one
  gold? This isolates the "which chunk" effect that breadcrumbs target,
  free of cross-page ranking.
- **Page-level:** page score = its best chunk's similarity; MRR, nDCG@10
  and Recall@10 against the single gold page. Reported on all eligible
  questions and on the 59-query quality-test subset
  (`data/conditionalqa/queries.json`).
- **Cost:** wall time to embed the corpus per variant, and mean/max
  breadcrumb length in characters.

### 5. Statistics

Differences are paired (`full` − `base`, per query). The confidence
interval is a **cluster bootstrap by gold page** (10,000 resamples, fixed
seed), because several dev questions share a page and are not independent.
The script prints the point estimate and the 95% interval for every
metric.

### 6. Decision rule (fixed in advance)

Ship-worthy, meaning "write the feature spec", only if **both**:

1. the 95% interval of the `full` − `base` difference in **chunk-level
   MRR** excludes zero (above it); and
2. on the 59-query subset, no page-level metric (MRR, nDCG@10,
   Recall@10) is lower for `full` than `base` by more than one query's
   worth (1/59 ≈ 0.017).

A page-level gain beyond Node's margin on that subset (about +0.04 to
+0.05, from the quality test) is reported as an extra win but is not
required. Anything else means **no**, and the numbers go in the roadmap.
In-page hit@1, hit@k and the `last` variant are context, not criteria.

This is a vector-only result. A "yes" must still be re-checked end to end
(`query`, with expansion, RRF and rerank) before release, because later
stages can hide or exaggerate a vector-level gain; that check belongs to
the follow-up feature spec.

### 7. Outputs and cache

- Chunk vectors per variant are cached under
  `data/conditionalqa/breadcrumbs-cache/` (gitignored with the rest of
  `data/`), keyed by model id and a hash of the exact formatted inputs, so
  re-running metrics or the bootstrap does not re-embed.
- The script prints a markdown results table and writes the same to
  `--out` (default under `data/conditionalqa/`). Results are pasted into
  this spec's "Results" section and summarised in the roadmap.
- It is read-only against the repo and the user's index.

## Testing

Fast unit tests, no model: heading-path extraction (nesting, popping,
same-level siblings, headings in code fences, heading at `pos`, no
headings), evidence location and chunk overlap (list items, tables,
entities, not found), the metric functions on small hand-made rankings,
and the cluster bootstrap on a fixture with a known difference. The
embedding run itself is the experiment and is not a test.

## Not in scope

- Any change under `src/`, `COMMAND_STATUS.md` or the skills.
- Fingerprints, re-embedding behaviour, and what `index_content` would
  pass to `embed()` (the feature spec, if the answer is yes).
- `embed --contextual`, non-markdown heading rules, SciFact.
- The `query` pipeline, FTS and the user's own index.

## Commits

Nothing here releases: `test(scripts): …` for the script and its tests,
then `docs(specs): publish the heading breadcrumbs experiment spec` once
results are in.

## Amendment 1: confirmatory retest on the train split

Written 2026-10-15, **before any run on the train split**. Runs 1 and 2
(see Results) used the 271 answerable dev questions. Those sit on only 59
distinct pages, and the bootstrap resamples pages, so its 95% interval was
about ±0.034 wide; run 2's +0.027 gain in chunk-level MRR could not be
separated from zero. The path rule used in run 2 was also fixed after seeing
run 1. This amendment fixes everything for one confirmatory run, so that the
result cannot be argued either way afterwards.

### What changes

- **Question set: the train split only.** `train.json` from the same pinned
  ConditionalQA commit as the dev file: 2,338 questions, 2,246 answerable with
  evidence, on 436 distinct pages, none of them a dev page, all in the corpus
  already embedded. Dev has been seen twice and plays no part in the
  decision; it is rerun for continuity and reported separately.
- **Gold labelling (strict).** A located evidence item counts only if it
  occurs exactly once on its page, matched on word boundaries (`(?<!\w)` and
  `(?!\w)` around the escaped text). An item that occurs more than once is
  ambiguous and is dropped; an item not found is missing. A question with no
  usable item left is excluded and counted, as ambiguous if it had at least
  one ambiguous item and as unmatched otherwise. This replaces "every
  occurrence of every item" (runs 1 and 2), which marked every chunk of a page
  as gold for 39 of 271 dev questions. The old labelling is rerun on the same
  train questions as a **sensitivity check only**, never for the decision.
- **Exclusion cap: 15%** of the answerable-with-evidence questions (page
  missing, unmatched or ambiguous). Strict labelling excludes more than the
  old rule; if the cap is exceeded the run fails and the labelling, not the
  threshold, is investigated.
- **Heading path: unchanged from run 2** (the path of the section holding
  most of the chunk, ties to the earlier section). Chunks, titles and the
  embedding model are unchanged, so the cached chunk vectors for `base` and
  `full` apply; only the train queries are embedded.

### Decision rule (same form as section 6, one run)

`full` versus `base` on the scored train questions:

1. the 95% interval of the difference in chunk-level MRR (cluster bootstrap
   by gold page, 10,000 resamples, seed 0) has a lower bound above zero; and
2. none of page MRR, page nDCG@10, page Recall@10 has a mean difference below
   −0.01 over the same questions. (The dev guard was one query's worth,
   1/n; with about 2,000 questions that would be 0.0005, which is inside
   noise, so a fixed 0.01 takes its place.)

Both → write the feature spec (re-embedding via a fingerprint, `.md` files
only, and a check of `query` end to end before release; the result here is
vector-only). Either fails → breadcrumbs are dropped for good and not
retested on this corpus.

### Reported, not decided

`last`; chunk hit@1, @5, @10; in-page hit@1; the sensitivity run with the old
labelling; the dev rerun; embed times. The point estimate and interval of the
primary difference are reported whatever the outcome. If the lower bound is
above zero but the gain is small (below +0.01) the report says so, and the
feature spec has to weigh that against the cost of re-embedding.

### What this amendment does not allow

- No new heading-path rules, depth caps, formats or variants chosen after
  seeing results.
- No second look at the train split. A bug in the script found afterwards may
  be fixed and the run repeated once, with the defect recorded here; a result
  that merely disappoints may not.
- No use of the dev questions to tune anything.

### Corrections to earlier text

- The claim in run 2's option list that more questions on this corpus would
  add little was wrong: the dev questions use only 59 of the 652 pages, and
  the train questions cover 436 others.
- Run 2's "about ±0.034 wide" interval comes from those 59 clusters.

## Results

## Results of Amendment 1 (train split, one preregistered run)

**Run:** 2026-10-15, script after the strict-labelling and split changes (Amendment 1's tooling), `--split train`, strict
labelling, 10,000 bootstrap resamples, seed 0, same chunks, titles and
embedding model as runs 1 and 2 (cached chunk vectors; only the train queries
were embedded). Nothing was changed after seeing results. One disclosure: a
smoke run on 40 pages (127 train questions) printed its numbers before the
decision run; no choice was made from it.

**Decision: YES.** Both halves of the rule hold.

1. Chunk-level MRR, `full` minus `base`: **+0.0241**, 95% interval
   **[+0.0135, +0.0346]**, lower bound above zero.
2. Page-level guard (limit −0.01): page MRR +0.0087, page nDCG@10 +0.0107,
   page Recall@10 +0.0144, all above the limit and all positive.

Coverage: 2,217 of 2,246 eligible train questions scored (1.3% excluded, all
as ambiguous; 0 page missing, 0 evidence not located; 539 evidence items were
ambiguous and dropped; 91 unanswerable and 1 without evidence were not
eligible). Under the 15% cap by a wide margin.

### Train, strict labelling (decision run): means and differences with 95% page-clustered CI

| Metric        | base   | full   | last   | full − base (95% CI)       | last − base (95% CI)       |
| ------------- | ------ | ------ | ------ | -------------------------- | -------------------------- |
| chunk_mrr     | 0.4974 | 0.5216 | 0.5166 | +0.0241 [+0.0135, +0.0346] | +0.0192 [+0.0085, +0.0294] |
| chunk_hit1    | 0.3956 | 0.4186 | 0.4118 | +0.0230 [+0.0063, +0.0393] | +0.0162 [−0.0005, +0.0324] |
| chunk_hit5    | 0.6130 | 0.6401 | 0.6355 | +0.0271 [+0.0146, +0.0397] | +0.0226 [+0.0099, +0.0351] |
| chunk_hit10   | 0.6901 | 0.7046 | 0.7059 | +0.0144 [+0.0028, +0.0258] | +0.0158 [+0.0045, +0.0271] |
| in_page_hit1  | 0.7718 | 0.8029 | 0.7957 | +0.0311 [+0.0155, +0.0466] | +0.0239 [+0.0083, +0.0394] |
| page_mrr      | 0.5891 | 0.5979 | 0.5932 | +0.0087 [−0.0004, +0.0179] | +0.0041 [−0.0050, +0.0130] |
| page_ndcg10   | 0.6243 | 0.6350 | 0.6318 | +0.0107 [+0.0028, +0.0187] | +0.0075 [−0.0005, +0.0156] |
| page_recall10 | 0.7600 | 0.7745 | 0.7754 | +0.0144 [+0.0045, +0.0248] | +0.0153 [+0.0054, +0.0256] |

### Reported, not decided

- **Sensitivity, old labelling (every occurrence), same train questions** (2,246 scored): chunk MRR
  +0.0230 [+0.0127, +0.0335]; page MRR +0.0090, nDCG@10 +0.0109, Recall@10
  +0.0147. The rule would also say YES. The labelling change did not drive
  the result.
- **Dev, strict labelling, for continuity** (269 questions on 59 pages): chunk
  MRR +0.0267 [−0.0075, +0.0606], chunk hit@5 +0.0818 [+0.0354, +0.1286],
  page nDCG@10 +0.0304. Same effect size as on train, but only 59 clusters,
  so the interval still spans zero. Dev is consistent with train, not
  independent evidence.
- **`last`** (innermost heading only) also helps, a little less: chunk MRR
  +0.0192 [+0.0085, +0.0294], page MRR and nDCG@10 not clearly above zero.
  Under the rule `last` cannot ship on its own.
- **Embed cost** was not re-measured (the cached vectors were reused). The
  path adds 30 to 52 characters on average (at most 219) to a chunk of up to
  2,700, roughly 8 to 15 tokens.

### Reading

- The effect is real but modest: about +0.024 chunk-level MRR (+5% of the
  baseline 0.497) and +0.03 on in-page hit@1, i.e. the right chunk of the
  right page is found a few points more often. Page-level gains are about
  +0.01, with page MRR's interval just touching zero. It is above the +0.01
  floor the amendment set for calling a gain small, so no "small gain"
  caveat is owed, but it is not large either.
- Run 1's apparent loss and run 2's narrow miss are explained: run 1
  measured mislabelled breadcrumbs, and run 2 had the right effect size with
  too few pages (59) to resolve it.
- Limits: vector-only (the product searches with FTS, RRF and a reranker, so
  the end-to-end gain may be smaller or larger); one corpus of well-structured
  gov.uk pages; the heading path rule was fixed after run 1 (disclosed above).
  A YES here means "write the feature spec", not "ship".

### What the feature spec has to settle

1. Where the path goes: in the title string for `.md` files only (that is
   what was tested), `Store.index_content` computing the dominant-section
   path per chunk.
2. The cost: stored vectors become stale, so the embedding fingerprint
   changes and every user re-embeds once; pyqmd's embedding input would
   no longer match Node's, which the parity work cared about. That trade is
   the project owner's call.
3. Verification before release: the parity quality test and an end-to-end
   `query` check on ConditionalQA, since this result stops at vector search.
4. Whether `last` (cheaper, about 80% of the gain) is the better form.

Two runs. Run 2 supersedes run 1, which had a defect found by the final
review; both are kept because run 2's fix was made after seeing run 1's
result.

## Run 2: path of the section holding most of the chunk

**Run:** 2026-10-15, script after the dominant-section path fix, same corpus, model,
resamples (10,000) and seed (0) as run 1. Only `full` and `last` were
re-embedded; `base` and the query vectors came from the cache and are
identical to run 1.

**Decision under the pre-registered rule: NO, by a narrow margin.** The 95%
interval for the chunk-level MRR difference is [−0.0072, +0.0609]: the
point estimate is +0.027 but the lower bound is just below zero. The
page-level guard half passed with room to spare (every page-level delta on
the 52 quality-test queries is ≥ 0). The rule needs both halves, so the
answer is no.

Every chunk now has a non-empty path (100%; run 1 had 75%, since a chunk
that started in the intro got none).

### All 271 scored questions (means; differences with 95% page-clustered CI)

| Metric        | base   | full   | last   | full − base (95% CI)       | last − base (95% CI)       |
| ------------- | ------ | ------ | ------ | -------------------------- | -------------------------- |
| chunk_mrr     | 0.5205 | 0.5475 | 0.5478 | +0.0270 [−0.0072, +0.0609] | +0.0273 [−0.0057, +0.0596] |
| chunk_hit1    | 0.4207 | 0.4317 | 0.4391 | +0.0111 [−0.0394, +0.0597] | +0.0185 [−0.0304, +0.0667] |
| chunk_hit5    | 0.6236 | 0.7011 | 0.6827 | +0.0775 [+0.0314, +0.1250] | +0.0590 [+0.0207, +0.0979] |
| chunk_hit10   | 0.7232 | 0.7601 | 0.7528 | +0.0369 [−0.0077, +0.0813] | +0.0295 [−0.0039, +0.0630] |
| in_page_hit1  | 0.8229 | 0.8118 | 0.8155 | −0.0111 [−0.0596, +0.0345] | −0.0074 [−0.0532, +0.0376] |
| page_mrr      | 0.5965 | 0.6272 | 0.6206 | +0.0307 [−0.0017, +0.0625] | +0.0241 [−0.0053, +0.0544] |
| page_ndcg10   | 0.6423 | 0.6725 | 0.6661 | +0.0302 [+0.0039, +0.0567] | +0.0239 [−0.0001, +0.0487] |
| page_recall10 | 0.8007 | 0.8266 | 0.8229 | +0.0258 [+0.0000, +0.0599] | +0.0221 [−0.0038, +0.0557] |

### Quality-test subset (52 queries), page level

| Metric        | base   | full   | last   | full − base | last − base |
| ------------- | ------ | ------ | ------ | ----------- | ----------- |
| page_mrr      | 0.7006 | 0.7161 | 0.7053 | +0.0156     | +0.0048     |
| page_ndcg10   | 0.7382 | 0.7540 | 0.7409 | +0.0158     | +0.0027     |
| page_recall10 | 0.8654 | 0.8846 | 0.8654 | +0.0192     | +0.0000     |

### Cost

Breadcrumbs add 30 to 52 characters on average (at most 219), roughly 8 to
15 tokens, to a chunk of up to 2,700 characters. Embed seconds are not
comparable across variants (see the note in run 1).

### Reading

- Direction changed from run 1. With the right section's path, every
  chunk-level and page-level metric moves up or stays flat, and the
  significant in-page hit@1 drop of run 1 is gone (−0.011, interval spans
  zero). That drop was an artifact of mislabelled breadcrumbs.
- It is still a "no" by the rule, and the margin is narrow, not decisive: the
  interval is about ±0.034 wide, so the data cannot separate "no effect"
  from "a gain of a few points". Run 2 is better described as inconclusive,
  leaning positive, than as evidence against.
- Secondary metrics that are individually clear of zero (not
  multiplicity-adjusted, so context, not criteria): chunk hit@5 +0.078,
  page nDCG@10 +0.030 (`full`); chunk hit@5 +0.059 (`last`). `last`, the
  innermost heading only, performs about as well as `full` here, so most of
  any gain does not need the whole path.
- Harness limits that probably reduce sensitivity: evidence matching is plain
  substring matching, so 39 of 271 questions have every chunk of their page
  marked gold (mean gold share is 48% of a page's chunks); short generic
  evidence such as "You must:" marks several chunks. This blunts chunk-level
  metrics for every variant alike, which is a reason a real gain could be
  hidden, not manufactured.
- The page-level guard used the 52 answerable quality-test queries, not 59:
  the other 7 are unanswerable and have no evidence to score chunks against.
  The limit was therefore 1/52 = 0.0192, not 1/59; the outcome does not
  change (all deltas ≥ 0).
- This is a vector-only result on one well-structured corpus, and run 2's
  path choice was fixed after seeing run 1. The pre-registered rule is the
  decision; no other variant or path rule was tried.

### What came next

See Amendment 1: the dev data covered only 59 pages, so a preregistered retest on
the train split (436 other pages) settled the question. The options listed here
when run 2 finished are superseded.

## Run 1: path at the chunk's start (superseded)

Defect: a chunk starts `EMBED_CHUNK_OVERLAP_CHARS` (405) characters before
the cut the chunker makes at a heading, so its start usually lies in the
previous section. The final review measured 1,758 of 1,914 non-first chunks
with a start path different from the path of the section holding most of
their text, and 74.6% of chunks having any path at all. The numbers below are
what the script computed before that fix; they measure mislabelled
breadcrumbs, not the feature. The "plausible reading" about neighbouring
chunks sharing a path further down is withdrawn: the in-page hit@1 drop is
explained by the defect.

**Run 1:** 2026-10-15, the script as first written, default embedding model
(`DEFAULT_EMBED_MODEL`), 10,000 bootstrap resamples, seed 0.

**Decision: NO.** The `full` variant did not clear the chunk-level bar: the
95% interval for the chunk-level MRR difference is [−0.0290, +0.0178],
which includes zero. The page-level guard passed (no metric fell by more
than one query's worth), but the rule needs both halves.

Coverage before the run: 2,566 chunks; 271 of 271 eligible questions
scored (0 page missing, 0 evidence not located, 0 evidence items missing;
14 unanswerable questions were not eligible); 74.6% of chunks have a
non-empty heading path, so breadcrumbs changed the input for three chunks
in four. 52 of the 59 quality-test queries were scored (the other 7 are
unanswerable).

### All 271 scored questions (means; differences with 95% page-clustered CI)

| Metric        | base   | full   | last   | full − base (95% CI)       | last − base (95% CI)       |
| ------------- | ------ | ------ | ------ | -------------------------- | -------------------------- |
| chunk_mrr     | 0.5205 | 0.5152 | 0.5297 | −0.0053 [−0.0290, +0.0178] | +0.0092 [−0.0170, +0.0337] |
| chunk_hit1    | 0.4207 | 0.4059 | 0.4354 | −0.0148 [−0.0569, +0.0266] | +0.0148 [−0.0284, +0.0560] |
| chunk_hit5    | 0.6236 | 0.6531 | 0.6384 | +0.0295 [+0.0036, +0.0561] | +0.0148 [−0.0144, +0.0424] |
| chunk_hit10   | 0.7232 | 0.7048 | 0.7085 | −0.0185 [−0.0382, +0.0000] | −0.0148 [−0.0358, +0.0039] |
| in_page_hit1  | 0.8229 | 0.7675 | 0.8118 | −0.0554 [−0.1036, −0.0077] | −0.0111 [−0.0545, +0.0320] |
| page_mrr      | 0.5965 | 0.5995 | 0.5983 | +0.0029 [−0.0182, +0.0223] | +0.0018 [−0.0222, +0.0255] |
| page_ndcg10   | 0.6423 | 0.6422 | 0.6394 | −0.0001 [−0.0177, +0.0160] | −0.0028 [−0.0223, +0.0163] |
| page_recall10 | 0.8007 | 0.7934 | 0.7897 | −0.0074 [−0.0268, +0.0144] | −0.0111 [−0.0352, +0.0121] |

### Quality-test subset (52 queries), page level

| Metric        | base   | full   | last   | full − base | last − base |
| ------------- | ------ | ------ | ------ | ----------- | ----------- |
| page_mrr      | 0.7006 | 0.6955 | 0.6809 | −0.0051     | −0.0197     |
| page_ndcg10   | 0.7382 | 0.7344 | 0.7226 | −0.0038     | −0.0156     |
| page_recall10 | 0.8654 | 0.8654 | 0.8654 | +0.0000     | +0.0000     |

The guard limit was −0.0192 (one query of 52).

### Cost

| Variant | Embed seconds | Breadcrumb chars added (mean / max, chunks with a path) |
| ------- | ------------- | ------------------------------------------------------- |
| base    | 167.2         | -                                                       |
| full    | 53.2          | 56 / 192                                                |
| last    | 48.3          | 31 / 119                                                |

The `base` time (167 s) is about 3 times the others. The model loads when
the queries are embedded, before `base`, so load time is not the reason; the
likely cause is batch shape (`base` sends batches of up to 16 chunks under one
title, `full` and `last` one or two per distinct path). The times are not
comparable. Breadcrumbs add 31 to 56
characters (roughly 8 to 15 tokens) to a chunk of up to 2,700 characters,
so cost is not what decides this.

### Reading

- Breadcrumbs did not move the primary metric. The point estimate for
  `full` is slightly negative (−0.005 MRR) and the interval (about ±0.02)
  fits anything from a small loss to a small gain.
- The significant secondary movements are mixed, and context rather than
  findings (eight metrics per variant, no multiplicity adjustment): `full`
  raises chunk hit@5 by +0.030 (interval just clear of zero) but lowers
  in-page hit@1 by −0.055 (interval clear of zero). The second isolates
  "which chunk of the right page". A plausible reading, not tested here:
  neighbouring chunks of a section share the same heading path, so adding it
  pulls them toward each other and blurs the difference between them.
- `last` (innermost heading only) shows nothing significant either way and
  cannot ship on its own under the rule.
- Page-level numbers are flat, as the earlier embedding-input change was.
- This is a vector-only result on one corpus, gov.uk pages with a clean
  `#`/`##`/`###` structure, which is the best case for breadcrumbs. It is
  no evidence that breadcrumbs help here, which is what the rule asked for;
  it is not strong evidence that they help nowhere.

### Follow-up

None required. Anyone revisiting should treat it as a new experiment
(for example breadcrumbs only in the FTS title field, or a corpus whose
chunks are long enough that the heading is far from the chunk text).
