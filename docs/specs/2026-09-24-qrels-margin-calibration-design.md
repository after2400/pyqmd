# Qrels-mode quality margin: larger-N calibration

**Date:** 2026-09-24
**Status:** Implemented (2026-09-24)
**Related:** `2026-09-10-python-mlx-rewrite-roadmap.md` (backlog entry
"Larger-N calibration of the qrels-mode quality margin");
`2026-09-12-python-node-parity-suite-design.md` ("Quality test
methodology"); `parity/README.md` (the 2026-09-13 n=5 calibration note).

## Problem

`parity/test_quality.py::test_pyqmd_meets_qrels_mode_quality_bar` checks
that pyqmd's MRR, nDCG@10, and Recall@10 on SciFact's 30 queries stay
within `MARGIN = 0.05` of Node's baseline in
`parity/node_ref/scifact/quality/node_query_baseline.json`. Two things are
wrong with that:

1. **The baseline is a single random draw.** A 2026-09-13 run of 5 repeated
   captures against the same Node commit and corpus found `mrr` ranging
   0.746–0.789 and `ndcg_at_10` 0.768–0.795. `recall_at_10` held at 0.8667
   for 6 captures, then came in at 0.9 on the 7th. The committed baseline
   is whichever draw happened to land.
2. **One margin for three metrics with different noise.** Recall barely
   moves and MRR moves a lot, so a single 0.05 is either loose for recall,
   tight for MRR, or both. And the 0.05 was a reasoned guess, not a number
   measured from the data.

### Source of the nondeterminism

The README and roadmap attribute the noise to Node's LLM rerank step. The
more likely source is **query expansion**. Node calls
`session.prompt(..., {temperature: 0.7, topK: 20, topP: 0.8, ...})`
(`qmd/src/llm.ts`, `expandQuery`), with a comment saying greedy decoding
must not be used. Different expansions produce different FTS and vector
candidate pools, which is enough to move documents across the top-10
boundary.

pyqmd's `llm/expand.py` calls `mlx_lm.generate` with no sampler, so it
decodes greedily. Its reranker is a deterministic forward pass. pyqmd is
therefore expected to be run-to-run deterministic. That is an expectation,
not a verified fact; it gets verified during the calibration (see
"Calibration run").

The practical consequence: the test compares a (probably) fixed pyqmd value
against one random draw from Node's distribution. What the margin should
absorb is Node's run-to-run spread, and that can be measured.

### Why N=30 is cheap

The roadmap priced a 20–30-run calibration at around 2 hours because each
`capture_quality_baseline()` call re-runs `collection add` and `embed` over
all 5,183 documents. Embedding isn't where the variance is. The variance
arises at query time, so the corpus is indexed once and only the 30-query
pass repeats.

Node caches expansion (and rerank) results in the index's `llm_cache` table
(`qmd/src/store.ts`, `expandQuery`: `getCachedResult(db, cacheKey)`). Every
pass after the first would just replay the first pass's expansions and
measure zero variance. Node's `qmd cleanup` clears `llm_cache` (plus
orphan/inactive cleanup and a vacuum, which do nothing here), so the capture
runs it against the isolated capture index before each later pass.

## Design

### 1. Capture: `--quality-runs N`

`parity/capture_node_snapshots.py` gains `--quality-runs N` (default `30`,
must be ≥ 1). `capture_quality_baseline(profile, qmd_repo_root, output_dir,
runs=30)`:

- Runs `collection add` and `embed` once, as today.
- **Qrels mode:** runs the full query pass `runs` times. Before every pass
  after the first, it calls `run_node_cli(qmd_repo_root, ["cleanup"],
index_path)` and raises `RuntimeError` on a non-zero exit, matching the
  existing `add`/`embed` error handling. It computes the three metrics per
  pass (same code as today, pulled into a helper) and writes the aggregate
  file described below. It prints each pass's metrics as it goes, so a long
  run shows progress.
- **Agreement mode** (no qrels): unchanged, a single pass. Calibrating
  `OVERLAP_THRESHOLD` is out of scope; no second real dataset exists to
  tune it against.

`--quality-runs` only matters when the quality phase runs (`--phase quality`
or `all`).

### 2. Baseline file format (additive)

```json
{
  "mrr": 0.77,
  "ndcg_at_10": 0.78,
  "recall_at_10": 0.87,
  "calibration": {
    "runs": 30,
    "num_queries": 30,
    "stddev": { "mrr": 0.012, "ndcg_at_10": 0.008, "recall_at_10": 0.006 },
    "per_run": [{ "mrr": 0.0, "ndcg_at_10": 0.0, "recall_at_10": 0.0 }]
  }
}
```

- The top-level metric keys stay the baseline point, now the **mean** over
  `runs` passes, so anything reading them keeps working.
- `stddev` is the sample standard deviation (`statistics.stdev`, n−1). With
  `runs == 1` there's no stddev to compute, so the `calibration` block is
  omitted entirely and the file matches today's format exactly.
- `per_run` is kept so the margin can be recomputed later (for example with
  a different K) without re-capturing.
- A baseline file with no `calibration` block (today's scifact file, or any
  user's already-captured personal profile) still loads.

### 3. Test: per-metric margin

In `parity/test_quality.py`:

```python
MARGIN_SIGMAS = 3.0  # one-sided; ~0.13% false-fail per metric if Node's spread is ~normal
FALLBACK_MARGIN = 0.05  # used when the baseline has no calibration block


def _metric_margin(baseline: dict, metric: str) -> float:
    cal = baseline.get("calibration")
    if cal is None:
        return FALLBACK_MARGIN
    return max(MARGIN_SIGMAS * cal["stddev"][metric], 1 / cal["num_queries"])
```

- The **1/num_queries floor** is one query's worth of change: the smallest
  step Recall@10 can take when a query has a single relevant document
  (SciFact's usual case), and a meaningful unit for MRR/nDCG too. It stops a
  near-zero σ (recall's) from turning the check into "must match Node
  exactly", which a single legitimate ranking difference would fail.
- The assertion becomes `pyqmd_value >= node_mean - margin`. The failure
  message shows the metric, pyqmd's value, the Node mean, σ, K, and the
  margin actually applied (and says when the fallback was used).
- The test stays one-sided: pyqmd beating Node never fails it.
- `MARGIN` is renamed `FALLBACK_MARGIN`. Nothing outside this file
  references it (checked with grep during implementation).

The margin helper is a pure function, so it's unit-tested without models
(see Testing).

### 4. Calibration run (manual, once)

Following the standing rule for `capture_node_snapshots.py` (manual,
on-demand only), this is run deliberately, by hand, once:

1. Confirm the Node checkout's `HEAD` matches
   `parity/node_ref/scifact/COMMIT.txt` (`8262698`). `--phase quality`
   already enforces this.
2. `uv run python -m parity.capture_node_snapshots --qmd-repo-root <qmd>
--phase quality --quality-runs 30`
3. **pyqmd determinism check:** run pyqmd's 30-query pass 3 times against
   one indexed store, using a throwaway scratchpad script with the same
   calls the test makes, and compare the ranked lists. If they differ, stop
   and report back before finishing: pyqmd's own variance would then have
   to go into the margin, which changes this design.
4. Commit the new `node_query_baseline.json`, and record the observed
   means, σ, resulting margins, and pyqmd's current values in the README.

Note: pyqmd's query expansion only started working today (v0.6.0, the
GGUF-weights fix), so pyqmd's numbers quoted in the roadmap (0.7785 /
0.8105 / 0.9333) predate working expansion and are stale. The calibration
records fresh ones.

### 5. Docs

- `parity/README.md`: replace "A note on the qrels-mode quality test's
  margin" with the new method (mean + K·σ with the 1/N floor), the N=30
  numbers, and the corrected nondeterminism source (expansion sampling).
  Document `--quality-runs` next to the other capture flags.
- Roadmap: mark the backlog entry done, fix the "rerank nondeterminism"
  attribution in it and in the Qwen3-4B experiment note, and remove the item
  from the "Next step" list.
- `COMMAND_STATUS.md`: no row changes. The parity suite's methodology isn't
  a command's existence, behavior, or output text.
- CLAUDE.md's command list is untouched (no pyqmd CLI surface changes).

## Testing

Fast unit tests (no models, no Node):

- `_metric_margin`: returns K·σ when that's above the floor, the
  1/num_queries floor when σ is tiny or zero, and `FALLBACK_MARGIN` when
  there's no calibration block.
- Multi-pass aggregation in `capture_quality_baseline`, with `run_node_cli`
  monkeypatched to return canned `query --format json` output that varies
  per pass: checks `cleanup` is called exactly `runs − 1` times, only
  between passes, and in the right order; that the written file's
  means/stddev/per_run match hand-computed values; and that `runs=1` writes
  no `calibration` block. Also checks that a non-zero `cleanup` exit raises.
- Argument parsing: `--quality-runs` defaults to 30 and rejects values
  below 1.

The real N=30 capture is the calibration itself, run by hand once, and is
never part of any automated suite.

## Out of scope

- Calibrating agreement mode's `OVERLAP_THRESHOLD`.
- Making Node deterministic, for example by capturing with a fixed seed or
  greedy expansion. That would measure a Node configuration users don't
  run.
- Query-sampling uncertainty (whether 30 SciFact queries represent quality
  in general). The test compares both systems on the same fixed query set;
  generalizing beyond it is a benchmarking question, not a parity one.
