# Heading breadcrumbs in embeddings — feature design

**Status:** Implemented (2026-10-15) as an opt-in feature; verification recorded below

**Amended 2026-10-15, after the hybrid-stage check:** breadcrumbs ship **off by default**. The check ("Amendment 1" at the end) was inconclusive, the user chose opt-in, and the reasons are in "Decision after the hybrid check" below. Where this document says the default is on, or that everyone re-embeds once, read it as applying to users who turn `PYQMD_EMBED_BREADCRUMBS=1` on.

**Related:** the experiment that justifies this, `2026-10-15-heading-breadcrumbs-experiment-design.md`
(Amendment 1 and Results; PR #20); the baseline format from
`2026-09-29-embedding-input-parity-design.md` (merged as #16).

## Problem

pyqmd embeds each chunk as `title: <doc title> | text: <chunk>`, Node's format.
A chunk from deep inside a long page says nothing about where it sits. A
preregistered experiment on ConditionalQA's train split (2,217 questions on 436
pages) found that adding the chunk's section-heading path to the title raises
vector retrieval: chunk-level MRR +0.024 (95% CI +0.014 to +0.035), in-page
hit@1 +0.031, and no page-level metric regressed. That result is vector-only.

This spec turns it into a product change.

## Decision

For `.md` files, each chunk is embedded as

`title: <doc title> > <h2> > <h3> … | text: <chunk>`

where the path is the heading path of the **section that holds most of the
chunk** (the `full` form the experiment validated). Everything else is unchanged:
non-markdown files, the chunk sizes, the model, query-time chunking and search.

Choices made here, so the review can overturn them:

1. **`full`, not `last`** (innermost heading only). `full` is the form the
   preregistered rule passed; `last` got about 80% of the chunk-level gain and
   was never decided on. The two were not tested against each other.
2. **Opt-in (amended; the first draft was default on).** The environment variable `PYQMD_EMBED_BREADCRUMBS` (`1`, `true` or `on`, case-insensitive, enables it; unset or anything else leaves it off), read when a `Store` is created, plus a `Store` keyword `breadcrumbs: bool | None = None` that overrides it (tests and the parity suite use the keyword). Off by default means the embedding input and fingerprint are exactly Node's and today's, so no one re-embeds unless they turn it on.
3. **A length cap of 256 characters on the added path.** The experiment's
   longest path was 219 characters, so the cap changes nothing on the tested
   corpus. It protects the 2,048-token embedding limit from a document with
   enormous headings, which would otherwise push the chunk text out of the
   window. Cap rule: drop outermost headings until the added text fits; if the
   innermost heading alone does not fit, truncate it and end it with `…`.
4. **Diverging from Node's embedding input is accepted.** It is the first
   retrieval-quality departure from Node, not just a formatting one. The
   index is separate from Node's, so nothing breaks; side-by-side results with
   Node drift slightly further. The `COMMAND_STATUS.md` `embed` row declares it.

## Design

### 1. Heading paths: `src/pyqmd_mlx/store/_breadcrumbs.py`

A new module, moved from the experiment script (one source of truth; the
script then imports it, which also keeps the script's reported numbers
reproducible because the behaviour is identical):

- `Heading(pos, level, text)` and `find_headings(content)`: ATX headings
  (`^#{1,6}[ \t]+…`) in document order, skipping lines inside code fences
  (`find_code_fences`), with an optional closing run of hashes needing a space
  before it so `# C#` keeps its `#`.
- `heading_path(headings, pos)`: the headings of the section containing `pos`,
  outermost first. A heading pops every entry of the same or deeper level, a
  heading starting at `pos` opens its own section, and level-1 headings (the
  page title) never appear in the path.
- `dominant_heading_path(headings, pos, length)`: the path of the section
  holding the most characters of `[pos, pos + length)`; a tie goes to the
  earlier section. This and not the path at `pos`, because a chunk starts 405
  characters before the heading its cut falls on, so its start usually lies in
  the previous section (the experiment's run 1 defect).
- `breadcrumb_title(doc_title, path, max_chars=256)`: the title string, applying
  the cap in decision 3. An empty path gives `doc_title` unchanged.

### 2. Per-chunk titles through the embedder

`llm.embed()` takes one `title` per call today. Chunks of one document now have
different titles, and one call per distinct title would shrink the batches. So:

- `embed(texts, model, kind, title)` and `format_embedding_inputs(...)` accept
  `title: str | list[str | None] | None`. A list has one entry per text (a
  mismatched length is a `ValueError`); a string or `None` behaves as today.
  `kind="query"` still ignores it.
- `Store.index_content` passes a list when breadcrumbs apply and the existing
  string otherwise, so with breadcrumbs off the call is identical to today's.
  Test fakes of `embed_fn` already accept `title=None`; none needs changing.

### 3. `Store.index_content`

For a chunk list from `embedding_chunks(...)`: if breadcrumbs are enabled and
the representative path's extension is `.md` (the same lower-cased last
extension rule `extract_title` uses), the title for each chunk is
`breadcrumb_title(extract_title(content, filepath), dominant_heading_path(
find_headings(content), pos, len(text)))`. Otherwise one title, as now. The
representative path (`MIN(path)` for the content hash) decides, exactly as it
does for the title today.

### 4. Fingerprint and re-embedding

`embedding_fingerprint(model, breadcrumbs=True)` gains one line when enabled:
`breadcrumbs:section-path-v1:256` (the format name and the cap). With
breadcrumbs off the fingerprint is byte-for-byte today's, so a user who opts
out never re-embeds. Consequences, all through mechanisms #16 already built:

- After upgrading, every stored vector has the old fingerprint, so
  `_count_hashes_needing_embedding` counts every document as pending.
  `pyqmd embed` re-embeds everything once, without `--force`; `status`,
  `vsearch` and `query` print the existing pending/stale warnings until then.
- Stale vectors keep working in the meantime; nothing breaks, the index is
  just in the old format until `embed` runs.
- Toggling the switch re-embeds, in either direction.

### 5. The experiment script

`scripts/bench_breadcrumbs.py` imports `find_headings` and
`dominant_heading_path` from the new module instead of defining them, and its
duplicated tests move to the module's tests. Its behaviour and the recorded
results do not change.

## Not in scope

- `last`, or any other path form; non-markdown files (`.org`, code); the cap
  beyond the one rule above.
- Query-time chunking, FTS, the reranker, or what `vsearch`/`query` do.
- A CLI flag or MCP parameter (the switch is an env var and a `Store` keyword).
- `embed --contextual` and the other ideas on the roadmap.

## Testing

Fast tests, no model:

- `_breadcrumbs`: the experiment's cases (nesting and popping siblings, a
  heading at the start position, `#` in code fences, closing hashes and
  `# C#`, no headings, level-1 headings, dominant-section choice and the tie
  rule), plus `breadcrumb_title` with and without the cap, a path of exactly
  256 characters, one over, and a single enormous heading.
- `embed`/`format_embedding_inputs`: per-text titles produce the expected
  strings for both embeddinggemma and Qwen formats; a list of the wrong length
  raises; `None` entries become `title: none`; `kind="query"` ignores a list.
- Store call site, with a recording fake `embed_fn`: a `.md` document with
  headings gets a list of per-chunk titles equal to `extract_title` plus the
  dominant path; a `.py` document, a markdown document with no headings, and
  breadcrumbs off each get the single `extract_title` string.
- Fingerprint: breadcrumbs on differs from off; off equals the value the
  repository's current `embedding_fingerprint` returns for the default model
  (pinned as a literal, so an accidental change fails); the cap is part of the
  on-fingerprint.
- Stale detection: a store whose vectors carry the off-fingerprint reports
  pending documents after the default turns on, `embed` re-embeds them, and a
  second `embed` is a no-op (the CLI's existing no-op check).

Parity suite: `tests/parity/test_embedding_inputs.py` formats strings through
`format_embedding_inputs` directly, never through a `Store`, so it is
unaffected and still proves the Node match. A new test checks that, on those
fixtures, the `Store`'s per-chunk titles with breadcrumbs on differ from
Node's only by the added path, and that with breadcrumbs off they equal it.
The quality test builds its own index with the default (on); the
verification below also runs it off.

Slow test (real model): two chunks with identical text under different heading
paths embed to different vectors.

## Verification before release

The experiment stops at vector search, so before the release:

- `just test-fast`, `just test-slow` and `just test-parity` green.
- The parity quality test on **ConditionalQA** with breadcrumbs on versus off
  (`PYQMD_EMBED_BREADCRUMBS=0`): no metric (MRR, nDCG@10, Recall@10) is lower
  with breadcrumbs by more than one query's worth (1/59); the numbers go in
  the spec. If end to end regresses past that, the default ships off and the
  result is recorded instead.
- The same on **SciFact**, which has no headings: the results must be
  identical (the strings do not change), which also shows the off-path and
  the on-path agree when no path exists.
- `embed` wall time on ConditionalQA before and after; per-text titles keep
  one batch per document, so a rise of more than about 10% means a regression
  to investigate. (The experiment's embed times were not comparable.)
- No command in this work writes to the user's own index; re-embedding it is
  the user's call after the release.

## Documentation and release

- `COMMAND_STATUS.md`: the `embed` row's declared differences gain
  "embeds `.md` chunks with their section-heading path (`PYQMD_EMBED_BREADCRUMBS=0`
  restores Node's input)"; the `status`/`vsearch`/`query` notes mention the
  one-time re-embed; the Roadmap tie-back records the change.
- `README.md`: a short "Upgrading" note (run `pyqmd embed` once) and the env
  var; `CLAUDE.md`'s Architecture bullets get the embedding-input line;
  `src/pyqmd_mlx/skills/pyqmd/SKILL.md` gets the env var if it describes
  embedding behaviour (the two command lists must not drift).
- `parity/README.md`: how the suite keeps matching Node (the keyword) and
  what the on-mode differences are.
- Release: one `feat(store)` minor release (a change to what `embed`
  produces, and everyone re-embeds). Commit scopes: `store` for the module and
  `index_content`, `llm` for the embedder, `config` only if `pyproject.toml`
  changes. The follow-up message to the qmd power user lists this among
  pyqmd's differences.

## Risks

- **The end-to-end gain may be smaller** than the vector-only one, since RRF
  and the reranker run after it. That is what the verification above checks.
- **One corpus.** The result is on well-structured gov.uk pages. Notes with
  flat or no headings will see little change; very uniform headings repeated
  across many pages will add little signal.
- **Heading text is embedded raw** (markdown emphasis and links included); the
  experiment did the same.
- **Mixed behaviour by file type** (`.md` gets paths, code does not) is a
  deliberate wart.

## Build order

1. After PR #20 merges, branch from `main` (the script and its tests must be
   there first to be refactored).
2. `_breadcrumbs.py` and its tests, moved out of the script.
3. Per-text titles in `embed`/`format_embedding_inputs`.
4. `index_content`, the fingerprint and the switch.
5. Parity changes and the end-to-end verification runs.
6. Docs. One PR, several `feat`/`test`/`docs` commits (each a changelog entry),
   with the commit tidy done before opening it.

## Verification results (2026-10-15)

Run with the finished feature: `uv run pytest parity/ -s --durations=5 [--dataset-config
parity/datasets/conditionalqa.yaml]`, breadcrumbs on (default) and off
(`PYQMD_EMBED_BREADCRUMBS=0`). Each run: 62 passed, 1 skipped.

### Quality, end to end (`query`: expansion, FTS, vector, RRF, rerank)

| Corpus (queries)   | Metric                   | Off    | On     | On − off | Node mean |
| ------------------ | ------------------------ | ------ | ------ | -------- | --------- |
| ConditionalQA (59) | MRR                      | 0.7374 | 0.7352 | −0.0022  | 0.5703    |
| ConditionalQA (59) | nDCG@10                  | 0.7686 | 0.7667 | −0.0019  | 0.6368    |
| ConditionalQA (59) | Recall@10                | 0.8644 | 0.8644 | 0.0000   | 0.8373    |
| ConditionalQA (59) | top-10 overlap with Node | 0.614  | 0.612  | −0.002   | n/a       |
| SciFact (30)       | MRR                      | 0.8034 | 0.8034 | 0        | 0.7579    |
| SciFact (30)       | nDCG@10                  | 0.8210 | 0.8210 | 0        | 0.7781    |
| SciFact (30)       | Recall@10                | 0.9000 | 0.9000 | 0        | 0.8722    |

The ConditionalQA runs were repeated in the opposite order and gave the same
numbers, so the metrics are deterministic.

### Ship rule

- **No regression on ConditionalQA:** every difference is above −1/59 (−0.0169).
  Met.
- **SciFact identical:** met (no headings, so the strings do not change).
- **Embed time within about 10%:** met. The suite's `setup` time (index + embed)
  was 178 s on and 58 s off when "on" ran first, 121 s off and 69 s on when "off"
  ran first: whichever run goes first pays 60 to 120 s of one-time cost, so
  those figures say nothing about breadcrumbs. Direct measurements:
  - tokens for the 652 documents' chunks as `embed()` sees them: raw +2.0%,
    padded +1.8% (1,361,481 to 1,388,942 and 1,606,867 to 1,636,574);
  - `embed()` on 80 documents, model warm, ABBA order (off, on, on, off, off,
    on, on, off): off 4.6 to 4.7 s, on 4.7 s per pass, a difference of about
    1%;
  - a six-pass alternating run over all documents drifted from 82 s to 132 s as
    the machine warmed, and was discarded as uninterpretable.

### What this does and does not show

- The feature does not regress anything measured end to end, and the cost is
  the expected extra 2% of tokens.
- **It also does not show a gain end to end on these 59 queries.** The experiment's
  vector-only gain (+0.024 chunk-level MRR on 2,217 train questions) is not
  visible after expansion, RRF and the reranker here: the ConditionalQA
  differences are −0.002 and the quality test's own noise margin is 0.04 to
  0.05, so it could not show a gain of that size either way. The benefit is
  therefore established at the retrieval stage, not yet at the final ranking.

## Amendment 1: hybrid-stage check (written 2026-10-15, before any run)

The end-to-end check above could not show the vector-stage gain (59 queries,
noise margin 0.04 to 0.05, and the reranker reorders the candidates anyway).
The question this amendment answers is narrower: **does the gain survive
fusion with full-text search?** It stops short of the reranker, which is the
stage a 59-query check cannot resolve.

- **Questions:** the 2,217 train questions the experiment scored with strict
  labelling (`--split train`, same selection code), compared at the page
  level: the gold page is the page named by the question.
- **Pipeline:** a real `Store` per mode, indexing the 652-page ConditionalQA
  corpus through the product's own scan and `index_content` (breadcrumbs on,
  then off via the `breadcrumbs` keyword), then `Store.query(q, limit=40,
skip_rerank=True)` with an `expand_fn` that returns no expansion lines.
  That is FTS + vector search + RRF exactly as the product fuses them, minus
  query expansion and reranking. Results are documents in fused-rank order;
  a page not in the top 40 scores 0.
- **Metrics:** page MRR, nDCG@10 and Recall@10, per question, `on` minus `off`,
  with a cluster bootstrap by gold page (10,000 resamples, seed 0).
- **Primary metric:** page nDCG@10. **Rule, fixed now:**
  1. Lower bound of the 95% interval above zero: the gain survives fusion;
     ship with the default on, as specified.
  2. Interval spans zero and every metric's point estimate is no lower than
     −0.01: inconclusive at this stage. The numbers go to the user, who decides
     the default; nothing is pre-decided.
  3. Any metric's point estimate below −0.01: ship with the default off (the
     switch and fingerprint stay, `breadcrumbs_enabled` returns False when
     the variable is unset), and say so in the README.
- **One run.** A defect in the script may be fixed and the run repeated once,
  with the defect recorded here. A result that merely disappoints may not.
- **Not in scope:** expansion and reranking (not measured here; the 59-query
  check is the only evidence for them), other corpora.
- **Precedence, added before any run:** if rules overlap (for example the nDCG@10 interval is above zero but another metric is below −0.01), 3 beats 1 beats 2.

### Amendment 1 result (2026-10-15, one run)

Run with `scripts/bench_breadcrumbs_hybrid.py` on the feature branch. Nothing
changed after the amendment was written except the precedence line, which was
added before the run. A 10-page smoke run printed numbers for 37 questions
beforehand (too few to mean anything; no choice depended on it).

2,217 train questions (strict labelling; 29 excluded as ambiguous), FTS +
vector + RRF over the 652-page corpus, no expansion, no reranker, top 40:

| Metric        | off    | on     | on − off (95% CI)          |
| ------------- | ------ | ------ | -------------------------- |
| page_mrr      | 0.5650 | 0.5694 | +0.0044 [−0.0034, +0.0120] |
| page_ndcg10   | 0.6056 | 0.6104 | +0.0048 [−0.0020, +0.0117] |
| page_recall10 | 0.7537 | 0.7600 | +0.0063 [−0.0031, +0.0159] |

**Outcome: rule 2, inconclusive at this stage; the user decides.** The primary
metric's interval (nDCG@10, −0.0020 to +0.0117) spans zero, and no metric's
point estimate is below −0.01 (all three are positive).

Reading:

- The point estimates are all positive but small: about +0.005 nDCG@10, +0.006
  Recall@10. The vector-only page-level gains were +0.011 nDCG@10 and +0.014
  Recall@10 (Results of Amendment 1 in the experiment spec), so fusion with
  full-text search carries over roughly half of them. That is what you would
  expect, since BM25 already finds many pages that the title path helps the
  vectors find.
- The interval is about ±0.007 wide: a real gain of up to +0.012 nDCG@10 is
  consistent with the data, and so is none.
- The reranker is the one stage not measured here. The 59-query end-to-end check
  could not resolve an effect this size either way.
- This is evidence about the size of the benefit. It does not change the cost: a
  one-time re-embed for everyone, and pyqmd's embedding input diverging from
  Node's.

## Decision after the hybrid check

The hybrid-stage result was rule 2 (inconclusive): every point estimate positive, none clear of zero. Options put to the user were default on, opt-in, or measuring on the user's own notes first (not enough notes yet). Opt-in was chosen:

- Fidelity to qmd is the stated goal, with deliberate differences reserved for a "clear improvement"; at the fusion stage the improvement is small and not clear, and the reranker stage is unmeasured.
- A default that later has to be reversed costs a re-embed twice; opt-in costs nobody anything now, and the default can be flipped later, with more evidence, in one release.
- The corpus is the best case for headings; flatter notes will see less.

Changes from the first draft: `breadcrumbs_enabled` is true only for `1`/`true`/`on`; the Store default is off, so `embedding_fingerprint(model)` (default off) and `Store` agree; the README section is "Optional: heading breadcrumbs" and the pending-embedding note applies only when the setting is toggled.
