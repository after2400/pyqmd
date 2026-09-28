import math

from compare_rerank_fixtures import (
    compare_fixtures,
    missing_query_ids,
    print_summary,
    rank_by_score,
    score_ranking,
    spearman_correlation,
)


def test_spearman_correlation_identical_rankings_is_one():
    assert abs(spearman_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-9


def test_spearman_correlation_reversed_rankings_is_negative_one():
    assert abs(spearman_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) - (-1.0)) < 1e-9


def test_spearman_correlation_fewer_than_two_scores_is_nan():
    assert math.isnan(spearman_correlation([1.0], [1.0]))
    assert math.isnan(spearman_correlation([], []))


def test_spearman_correlation_zero_variance_is_nan_not_perfect_agreement():
    # A broken reranker that emits a constant score for every document must NOT
    # be reported as perfectly correlated with the baseline.
    assert math.isnan(spearman_correlation([1.0, 1.0, 1.0], [5.0, 3.0, 8.0]))


def test_rank_by_score_orders_descending():
    assert rank_by_score(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]


def test_score_ranking_perfect_ranking_scores_full_marks():
    result = score_ranking(["rel1", "rel2", "irrelevant"], {"rel1": 1, "rel2": 1}, top_k=10)

    assert result["recall"] == 1.0
    assert result["mrr"] == 1.0


def test_score_ranking_no_relevant_found_scores_zero():
    result = score_ranking(["irrelevant1", "irrelevant2"], {"rel1": 1}, top_k=10)

    assert result["recall"] == 0.0
    assert result["mrr"] == 0.0


def test_compare_fixtures_reports_per_query_metrics_and_latency():
    fixture = [
        {
            "query_id": "q1",
            "query": "test",
            "candidate_docs": [{"doc_id": "d1", "text": "t1"}, {"doc_id": "d2", "text": "t2"}],
            "qrels": {"d1": 1},
        }
    ]
    baseline_scores = {
        "q1": {
            "scores": [{"doc_id": "d1", "score": 0.9}, {"doc_id": "d2", "score": 0.1}],
            "latency_ms": 5.0,
        }
    }
    candidate_scores = {
        "q1": {
            "scores": [{"doc_id": "d1", "score": 0.8}, {"doc_id": "d2", "score": 0.2}],
            "latency_ms": 8.0,
        }
    }

    result = compare_fixtures(fixture, baseline_scores, candidate_scores)

    assert result["q1"]["baseline"]["mrr"] == 1.0
    assert result["q1"]["candidate"]["mrr"] == 1.0
    assert abs(result["q1"]["spearman"] - 1.0) < 1e-9
    assert result["q1"]["baseline_latency_ms"] == 5.0
    assert result["q1"]["candidate_latency_ms"] == 8.0


def test_compare_fixtures_recall_reflects_truncated_top_k_not_full_candidate_list():
    # 11 candidate docs, only "rel" is relevant. top_k = max(len(relevant), 10) = 10,
    # so "rel" must actually rank inside the top 10 to count as a hit -- being
    # present anywhere in the full 11-doc candidate list must NOT be enough.
    doc_ids = [f"d{i}" for i in range(1, 11)] + ["rel"]
    candidate_docs = [{"doc_id": doc_id, "text": doc_id} for doc_id in doc_ids]
    fixture = [
        {
            "query_id": "q1",
            "query": "test",
            "candidate_docs": candidate_docs,
            "qrels": {"rel": 1},
        }
    ]
    # Baseline ranks "rel" dead last (score 0.0, below every distractor).
    baseline_scores = {
        "q1": {
            "scores": [{"doc_id": doc_id, "score": 1.0} for doc_id in doc_ids[:-1]]
            + [{"doc_id": "rel", "score": 0.0}],
            "latency_ms": 1.0,
        }
    }
    # Candidate ranks "rel" first (score 1.0, above every distractor).
    candidate_scores = {
        "q1": {
            "scores": [{"doc_id": "rel", "score": 1.0}]
            + [{"doc_id": doc_id, "score": 0.0} for doc_id in doc_ids[:-1]],
            "latency_ms": 1.0,
        }
    }

    result = compare_fixtures(fixture, baseline_scores, candidate_scores)

    # "rel" falls outside baseline's truncated top-10 -> recall/f1 must be zero,
    # not 1.0 as it would be if scored against the untruncated 11-doc list.
    assert result["q1"]["baseline"]["recall"] == 0.0
    assert result["q1"]["baseline"]["f1"] == 0.0
    # "rel" survives inside candidate's top-10 -> recall/f1 must be nonzero.
    assert result["q1"]["candidate"]["recall"] == 1.0
    assert result["q1"]["candidate"]["f1"] > 0.0


def test_missing_query_ids_identifies_absent_query():
    fixture = [{"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q3"}]
    scores = {"q1": {"scores": [], "latency_ms": 1.0}, "q3": {"scores": [], "latency_ms": 1.0}}

    assert missing_query_ids(fixture, scores) == ["q2"]


def test_missing_query_ids_empty_when_all_present():
    fixture = [{"query_id": "q1"}]
    scores = {"q1": {"scores": [], "latency_ms": 1.0}}

    assert missing_query_ids(fixture, scores) == []


def test_print_summary_warns_about_missing_queries(capsys):
    per_query = {
        "q1": {
            "spearman": 1.0,
            "baseline": {"mrr": 1.0, "f1": 1.0, "recall": 1.0},
            "candidate": {"mrr": 1.0, "f1": 1.0, "recall": 1.0},
            "baseline_latency_ms": 1.0,
            "candidate_latency_ms": 1.0,
        }
    }

    print_summary(per_query, baseline_missing=["q2"], candidate_missing=[])

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "1 queries missing from baseline scores" in out
    assert "0 missing from candidate scores" in out


def test_print_summary_no_warning_when_nothing_missing(capsys):
    per_query = {
        "q1": {
            "spearman": 1.0,
            "baseline": {"mrr": 1.0, "f1": 1.0, "recall": 1.0},
            "candidate": {"mrr": 1.0, "f1": 1.0, "recall": 1.0},
            "baseline_latency_ms": 1.0,
            "candidate_latency_ms": 1.0,
        }
    }

    print_summary(per_query)

    out = capsys.readouterr().out
    assert "WARNING" not in out
