import math

import pytest

from parity._quality_baseline import (
    FALLBACK_MARGIN,
    METRICS,
    compute_metrics,
    metric_margin,
    summarize_runs,
)


def test_compute_metrics_matches_hand_computed_values():
    # q1: the relevant doc is at rank 2 -> RR 0.5, nDCG 1/log2(3), recall 1.
    # q2: the relevant doc is missing -> all 0.
    metrics = compute_metrics([["a", "b"], ["c"]], [{"b"}, {"d"}])

    assert metrics["mrr"] == pytest.approx(0.25)
    assert metrics["ndcg_at_10"] == pytest.approx((1 / math.log2(3)) / 2)
    assert metrics["recall_at_10"] == pytest.approx(0.5)


def test_compute_metrics_on_no_queries_is_all_zero():
    assert compute_metrics([], []) == dict.fromkeys(METRICS, 0.0)


def test_summarize_single_run_is_the_run_itself_without_calibration():
    run = {"mrr": 0.7, "ndcg_at_10": 0.8, "recall_at_10": 0.9}

    assert summarize_runs([run], num_queries=30) == run


def test_summarize_multiple_runs_records_mean_sample_stddev_and_every_run():
    per_run = [
        {"mrr": 0.7, "ndcg_at_10": 0.8, "recall_at_10": 0.9},
        {"mrr": 0.8, "ndcg_at_10": 0.8, "recall_at_10": 0.9},
        {"mrr": 0.9, "ndcg_at_10": 0.8, "recall_at_10": 0.9},
    ]

    summary = summarize_runs(per_run, num_queries=30)

    assert summary["mrr"] == pytest.approx(0.8)
    assert summary["ndcg_at_10"] == pytest.approx(0.8)
    assert summary["recall_at_10"] == pytest.approx(0.9)
    cal = summary["calibration"]
    assert cal["runs"] == 3
    assert cal["num_queries"] == 30
    assert cal["stddev"]["mrr"] == pytest.approx(0.1)  # sample stddev (n-1)
    assert cal["stddev"]["ndcg_at_10"] == pytest.approx(0.0)
    assert cal["stddev"]["recall_at_10"] == pytest.approx(0.0)
    assert cal["per_run"] == per_run


def test_summarize_runs_rejects_an_empty_run_list():
    with pytest.raises(ValueError, match="at least one run"):
        summarize_runs([], num_queries=30)


def _calibrated(stddev: float, num_queries: int = 30) -> dict:
    return {
        "mrr": 0.77,
        "calibration": {
            "runs": 30,
            "num_queries": num_queries,
            "stddev": {"mrr": stddev},
            "per_run": [],
        },
    }


def test_metric_margin_is_three_sigma_when_above_the_floor():
    assert metric_margin(_calibrated(stddev=0.02), "mrr") == pytest.approx(0.06)


def test_metric_margin_floors_at_one_query_worth_of_change():
    assert metric_margin(_calibrated(stddev=0.001), "mrr") == pytest.approx(1 / 30)
    assert metric_margin(_calibrated(stddev=0.0, num_queries=10), "mrr") == pytest.approx(0.1)


def test_metric_margin_falls_back_without_a_calibration_block():
    assert metric_margin({"mrr": 0.77}, "mrr") == FALLBACK_MARGIN
