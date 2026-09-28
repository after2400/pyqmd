# `pyqmd bench --samples N`: report sampling spread

**Date:** 2026-09-25
**Status:** Implemented (2026-09-26)
**Related:** `2026-09-10-python-mlx-rewrite-roadmap.md` (backlog entry
"`pyqmd bench --samples N`: report sampling spread");
`2026-09-25-seeded-expansion-sampling-design.md` (made expansion sample,
seeded per query); `scripts/replay_query.py` (`--expand seeded:<salt>`,
whose salting this design moves into `llm/expand.py`).

## Problem

Query expansion samples, seeded per query from a hash of the query and
model. A `bench` run's `hybrid` and `full` scores (`bench/runner.py`, both
via `Store.query`) are therefore one draw per query. On SciFact the same
pipeline scored Recall@10 anywhere from 0.833 to 0.900 depending on the
seed. A user comparing two configurations on their own corpus (say
`--chunk-strategy auto` vs. `regex`) can't tell a real difference from that
noise.

An env var to change the seed was considered and rejected in the backlog
entry: users would run bench once per value and average by hand, and
picking the best-scoring seed only fits the fixture's exact query strings.

## Goals

- `pyqmd bench --samples N` runs each expansion-using backend (`hybrid`,
  `full`) N times with differently salted seeds and reports each metric's
  mean and min–max, per query and in the summary, in text and `--json`.
- `--samples 1` (the default) runs today's code path unchanged; its text
  and JSON output are byte-identical to today's apart from the timestamp and
  latency values, which already vary between runs.
- Sample 0 is today's run. Sample _i_ (i ≥ 1) uses the same seed as
  `scripts/replay_query.py --expand seeded:<i>`.
- `bench` keeps measuring the real `Store.query` pipeline — no
  reimplementation of retrieval, fusion, or blending.

## Non-goals

- Resampling `bm25` or `vector`: neither expands, so they run once.
- An upper cap on N, parallel samples, or a latency distribution across
  samples.
- Any change to `pyqmd query`, the MCP tools, or `Store.query`'s signature.
- A seed/salt option for `query` itself.

## Approach

Bench wraps the store's model calls for samples 1..N−1. Rejected
alternatives:

- **Thread a salt through `Store.query`** (`expand_salt=` down through
  `_retrieve_and_fuse`, plus a rerank cache inside `Store`): pushes a
  bench-only concern into the core query path and adds a parameter the CLI
  `query` or MCP could later expose by accident.
- **Reimplement the pipeline in bench**, as `replay_query.py` does:
  duplicates retrieval/fusion/blending, so bench would silently measure
  something other than what `pyqmd query` does.

## Design

### 1. Salted seeding in `llm/expand.py`

- `_seed_for(query, model, salt=None)`: `salt=None` hashes
  `f"{model}\n{query}"` exactly as today; a salt hashes
  `f"{model}\n{query}\n{salt}"` (the replay harness's current formula).
  First 4 bytes of the sha256, big-endian, as today.
- `expand_query(query, model=DEFAULT_EXPAND_MODEL, salt=None)` passes the
  salt through. `salt=None` is today's behavior.
- `scripts/replay_query.py` drops its own `salted_seed` and uses the shared
  `_seed_for`, so sample _i_ and `seeded:<i>` can't drift apart.

**Contract:** an `expand_fn` used with `--samples` > 1 must accept an
optional `salt` keyword. The real `expand_query` does; the bench tests'
fakes gain it. No other caller passes a salt.

### 2. `Store.wrapping_llm_fns`

A context manager on `Store`:

```python
@contextmanager
def wrapping_llm_fns(self, expand=None, rerank=None):
    """Temporarily replace the expand/rerank functions with wrappers of the
    current ones. Each argument maps the current function to its
    replacement. Restored on exit, including on error."""
```

- Each argument is `current_fn -> replacement_fn`, so a caller can wrap
  without reading `Store`'s private attributes.
- Originals are restored in a `finally`; nesting works (inner restores to
  the outer wrapper).
- For single-threaded callers (bench). The MCP server never uses it.

`Store.query` and every other caller are unchanged.

### 3. Runner flow

`run_benchmark(store, fixture, collection, samples=1)`.

**`samples == 1`:** today's loop, untouched.

**`samples > 1`:**

1. **Sample 0** — today's single sweep over the queries, all 4 backends via
   `_score_backend`, with one addition: the reranker is wrapped in a
   record-only pass-through. It still calls the real reranker with the
   same full batch every time, and additionally stores each score under
   `(rerank_query, sha1(doc))` in a run-wide cache. Results and batches are
   unchanged.
2. **Samples 1..N−1** — outer loop over samples, inner over queries:

   ```python
   with store.wrapping_llm_fns(
       expand=lambda f: _salted_memo(f, salt=str(i)),
       rerank=lambda f: _cached(f, rerank_cache),
   ):
       for query in fixture.queries:
           hybrid = _score_backend(store, "hybrid", query, collection)
           full = _score_backend(store, "full", query, collection)
   ```

   - **Expander:** calls the wrapped function with `salt=str(i)`,
     memoized by `(query, model)` in a cache created fresh per sample —
     `hybrid` and `full` share one draw per query; samples differ.
   - **Reranker:** looks up the run-wide cache; calls the real reranker
     only for unseen docs, in one batch per call; returns scores in the
     original order.
   - `bm25` and `vector` are not re-run.

3. **Progress:** one line per sample to stderr (`bench: sample 2/5…`).
   Nothing is printed when N = 1; stdout and `--json` stay clean.

**Aggregation:**

- Per query: for `hybrid`/`full`, mean, min, and max of each metric across
  the N raw scores.
- Summary: for each sample, compute `_compute_summary`'s averages over just
  that sample's `hybrid`/`full` scores; report the mean, min, and max of
  those per-sample fixture averages. (Min/max of per-query scores would
  just be 0–1; the per-sample fixture averages are the 0.833–0.900 spread
  that tells a real difference from noise.)

**Caveat:** the reranker scores in batches, so a cached score can differ in
the last few bits from a fresh score computed in a different batch. The
replay harness already accepts this, and the seeded-expansion spike's
numbers were produced that way.

### 4. Data model and output

With `--samples 1` nothing new appears — no keys, columns, or header lines.
With N > 1:

**`QueryResult`**

- `backends[name]` keeps its `BackendScore` shape, so existing JSON readers
  keep working. For `hybrid`/`full`, `recall_at_k`/`mrr`/`ndcg_at_k` become
  the means across samples and `latency_ms` is sample 0's (the uncached
  run). `bm25`/`vector` are as today.
- New `samples: {"hybrid": [BackendScore × N], "full": [BackendScore × N]}`
  holds every raw draw, sample 0 first.

**`BenchResult`**

- New top-level `samples: N`.
- `summary[name]`: `avg_*` is the mean of the per-sample fixture averages
  (equal to the mean of per-query means). For `hybrid`/`full`, new
  `min_recall_at_k`/`max_recall_at_k`, `min_mrr`/`max_mrr`,
  `min_ndcg_at_k`/`max_ndcg_at_k`. `avg_latency_ms` is sample 0's.

The new keys are omitted from `result_to_dict` when N = 1.

**Text table (N > 1)**

- A header line before the table:
  `Samples: 5 (hybrid/full: mean min–max; ms from sample 0)`.
- `hybrid`/`full` metric cells become `mean min–max`, e.g.
  `0.80 0.60–1.00` — the range of that one query's score across samples.
  `bm25`/`vector` rows show the single value, padded so columns align.
- Summary lines append the range after each metric:
  `hybrid   Recall@k=0.884 (0.833–0.900) MRR=0.791 (0.760–0.812) nDCG@k=… Avg=840ms`.

### 5. CLI and errors

- `--samples N`, int, default 1, composes with `-c` and `--json`.
- Validated before the fixture loads: N < 1 raises
  `ValueError("--samples must be at least 1")` through the existing
  `run_or_exit` → one-line `Error:` + exit 1 (not Typer's `min=`, which
  exits 2 with a Click usage error no other pyqmd command uses).
- No new error types. `ExpansionModelError` from any sample aborts the run,
  as today. A backend raising within a sample scores 0 for that sample
  only, and the run continues. `wrapping_llm_fns` restores the originals
  when a sample raises.

## Testing

All fast, with the existing fake-model pattern; the seed is verified
without loading MLX, so no slow tests.

- **`llm/expand.py`:** `_seed_for(q, m)` is unchanged from today's value;
  `_seed_for(q, m, salt)` equals `sha256(f"{m}\n{q}\n{salt}")`'s first 4
  bytes big-endian; `replay_query.py` uses the shared function.
- **`Store.wrapping_llm_fns`:** swaps and restores; restores after an
  exception; nesting restores to the outer wrapper.
- **Runner:**
  - `samples=1` → today's dict shape, no `samples` keys.
  - `samples=3` with a salt-sensitive fake expander → correct per-query
    mean/min/max; summary min/max are over per-sample fixture averages.
  - Call counts: `bm25`/`vector` once per query; from sample 1 on, the
    expander runs once per (query, sample) shared by `hybrid`/`full`; the
    reranker's inner function only ever sees docs not already scored.
  - A backend raising in one sample → 0 in that sample only.
  - `ExpansionModelError` in sample 2 propagates.
- **CLI:**
  - `--samples 1` text and JSON match the no-flag output (latency and
    timestamp masked).
  - `--samples 3` shows the header line, `mean min–max` cells, summary
    ranges; JSON has `samples`.
  - `--samples 0` → `Error:` + exit 1.
  - Progress lines on stderr only.

## Docs

- `COMMAND_STATUS.md`'s `bench` row.
- `CLAUDE.md`'s command list.
- `src/pyqmd_mlx/skills/pyqmd/SKILL.md` (command surface).
- `src/pyqmd_mlx/skills/pyqmd-bench/SKILL.md` (comparing configurations is
  what this flag is for).
- The roadmap backlog entry, marked ✅ with a pointer to this spec.

## Done when

`bench --samples 5` on a fixture reports means and ranges for `hybrid` and
`full`; `--samples 1` output is byte-identical to today's (timestamps and
latency aside); and the docs above describe the flag.
