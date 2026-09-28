import math

from pyqmd_mlx.bench._metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    spearman_rank_correlation,
    top_k_overlap,
)


def test_reciprocal_rank_of_first_relevant_result():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_of_third_relevant_result():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3


def test_reciprocal_rank_returns_zero_when_no_relevant_result_present():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean_reciprocal_rank_averages_across_queries():
    ranked = [["a", "b"], ["x", "y"]]
    relevant = [{"a"}, {"y"}]  # MRR contributions: 1.0, 0.5

    assert mean_reciprocal_rank(ranked, relevant) == (1.0 + 0.5) / 2


def test_mean_reciprocal_rank_empty_input_is_zero():
    assert mean_reciprocal_rank([], []) == 0.0


def test_recall_at_k_counts_fraction_of_relevant_found():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}  # z is never retrieved

    assert recall_at_k(ranked, relevant, k=4) == 2 / 3


def test_recall_at_k_respects_k_cutoff():
    ranked = ["a", "b", "c"]
    relevant = {"c"}

    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert recall_at_k(ranked, relevant, k=3) == 1.0


def test_recall_at_k_with_no_relevant_docs_is_zero_not_a_crash():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = ["a", "b"]
    relevant = {"a", "b"}

    assert math.isclose(ndcg_at_k(ranked, relevant, k=2), 1.0, abs_tol=1e-9)


def test_ndcg_at_k_penalizes_relevant_doc_ranked_lower():
    perfect = ndcg_at_k(["a", "b"], {"a"}, k=2)
    worse = ndcg_at_k(["b", "a"], {"a"}, k=2)

    assert perfect > worse


def test_ndcg_at_k_credits_a_repeated_relevant_id_only_once():
    # Two ranked results canonicalizing to the same expected id must not
    # push DCG past the ideal (observed: nDCG 1.63 on a real bench run).
    assert math.isclose(ndcg_at_k(["a", "a"], {"a"}, k=2), 1.0, abs_tol=1e-9)
    assert math.isclose(
        ndcg_at_k(["b", "a", "a"], {"a"}, k=3), ndcg_at_k(["b", "a"], {"a"}, k=3), abs_tol=1e-9
    )


def test_ndcg_at_k_with_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_top_k_overlap_full_agreement_is_one():
    assert top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_top_k_overlap_no_agreement_is_zero():
    assert top_k_overlap(["a", "b"], ["x", "y"], k=2) == 0.0


def test_top_k_overlap_partial_agreement():
    assert top_k_overlap(["a", "b", "c"], ["a", "x", "y"], k=3) == 1 / 3


def test_top_k_overlap_full_agreement_with_fewer_than_k_results_is_one():
    # Both sides returned only 3 results for a k=10 comparison and agree
    # completely -- the denominator must shrink to 3, not stay fixed at k,
    # or perfect agreement on a sparse query becomes mathematically
    # unreachable.
    assert top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=10) == 1.0


def test_spearman_rank_correlation_identical_order_is_one():
    result = spearman_rank_correlation(["a", "b", "c"], ["a", "b", "c"])
    assert result is not None
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_spearman_rank_correlation_reversed_order_is_negative_one():
    result = spearman_rank_correlation(["a", "b", "c"], ["c", "b", "a"])
    assert result is not None
    assert math.isclose(result, -1.0, abs_tol=1e-9)


def test_spearman_rank_correlation_only_over_common_ids():
    # "z" appears only in `a`, "y" only in `b" -- correlation is computed
    # over just {"a", "b", "c"}, the ids both lists share.
    result = spearman_rank_correlation(["a", "b", "c", "z"], ["a", "b", "c", "y"])
    assert result is not None
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_spearman_rank_correlation_returns_none_with_fewer_than_two_common_ids():
    assert spearman_rank_correlation(["a"], ["b"]) is None
    assert spearman_rank_correlation(["a"], ["a"]) is None


def test_spearman_rank_correlation_ignores_position_of_unique_leading_id():
    # "z" appears only in `a`, ahead of the ids both lists share -- ranking
    # must be recomputed over just {"a", "b", "c"} so their relative order
    # (identical in both lists) scores as perfect correlation. Comparing
    # each id's position in the original (longer) list instead would shift
    # a/b/c's apparent ranks in the first list by one, understating the
    # correlation.
    result = spearman_rank_correlation(["z", "a", "b", "c"], ["a", "b", "c"])
    assert result is not None
    assert math.isclose(result, 1.0, abs_tol=1e-9)
