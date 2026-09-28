"""Qrels-mode quality metrics, the multi-run Node baseline summary, and the
per-metric margin the qrels-mode test allows. Shared by
capture_node_snapshots.py (writes the baseline) and test_quality.py (reads
it). See docs/specs/2026-09-24-qrels-margin-calibration-design.md.
"""

from __future__ import annotations

import statistics

from pyqmd_mlx.bench._metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k

METRICS = ("mrr", "ndcg_at_10", "recall_at_10")

# One-sided: pyqmd fails a metric only when it's more than MARGIN_SIGMAS
# of Node's own run-to-run stddev below Node's mean -- ~0.13% false-fail
# per metric if that spread is roughly normal.
MARGIN_SIGMAS = 3.0

# For a baseline with no calibration block (a single-run capture, e.g. an
# older personal profile's) -- the pre-calibration fixed margin.
FALLBACK_MARGIN = 0.05


def compute_metrics(
    ranked_lists: list[list[str]], relevant_sets: list[set[str]]
) -> dict[str, float]:
    n = len(ranked_lists)
    if n == 0:
        return dict.fromkeys(METRICS, 0.0)
    pairs = list(zip(ranked_lists, relevant_sets))
    return {
        "mrr": mean_reciprocal_rank(ranked_lists, relevant_sets),
        "ndcg_at_10": sum(ndcg_at_k(r, s, k=10) for r, s in pairs) / n,
        "recall_at_10": sum(recall_at_k(r, s, k=10) for r, s in pairs) / n,
    }


def summarize_runs(per_run: list[dict[str, float]], num_queries: int) -> dict:
    """The node_query_baseline.json content for one or more query passes:
    top-level per-metric means (so single-point readers keep working), plus
    a calibration block when there's more than one pass to take a stddev
    over."""
    if not per_run:
        raise ValueError("summarize_runs needs at least one run")
    summary: dict = {m: statistics.fmean(run[m] for run in per_run) for m in METRICS}
    if len(per_run) > 1:
        summary["calibration"] = {
            "runs": len(per_run),
            "num_queries": num_queries,
            "stddev": {m: statistics.stdev(run[m] for run in per_run) for m in METRICS},
            "per_run": per_run,
        }
    return summary


def metric_margin(baseline: dict, metric: str) -> float:
    """How far below Node's baseline pyqmd may fall on `metric`. The
    1/num_queries floor is one query's worth of change -- without it a
    near-zero stddev (recall's, typically) would demand an exact match."""
    cal = baseline.get("calibration")
    if cal is None:
        return FALLBACK_MARGIN
    return max(MARGIN_SIGMAS * cal["stddev"][metric], 1 / cal["num_queries"])
