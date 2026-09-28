from pyqmd_mlx.store._rrf import get_hybrid_rrf_weights, reciprocal_rank_fusion
from pyqmd_mlx.store._types import RankedListMeta, RankedResult


def _result(file: str, score: float = 0.0) -> RankedResult:
    return RankedResult(file=file, display_path=file, title=file, body="body", score=score)


def test_single_list_preserves_order():
    results = [_result("a"), _result("b"), _result("c")]
    fused = reciprocal_rank_fusion([results])
    assert [r.file for r in fused] == ["a", "b", "c"]


def test_agreement_across_lists_boosts_score():
    list_a = [_result("a"), _result("b")]
    list_b = [_result("b"), _result("a")]
    fused = reciprocal_rank_fusion([list_a, list_b])
    # both appear in both lists at good ranks; b is rank 0 in list_b and
    # rank 1 in list_a, a is rank 0 in list_a and rank 1 in list_b --
    # symmetric, so scores should be equal and both should outscore an
    # item that only appears once.
    scores = {r.file: r.score for r in fused}
    assert scores["a"] == scores["b"]


def test_weights_favor_higher_weighted_list():
    list_a = [_result("only_in_a")]
    list_b = [_result("only_in_b")]
    fused = reciprocal_rank_fusion([list_a, list_b], weights=[2.0, 1.0])
    scores = {r.file: r.score for r in fused}
    assert scores["only_in_a"] > scores["only_in_b"]


def test_top_rank_bonus_applied():
    results = [_result("a")]
    fused = reciprocal_rank_fusion([results])
    # rank 0 gets +0.05 on top of the base 1/(60+1) contribution
    expected_base = 1.0 / 61
    assert fused[0].score == expected_base + 0.05


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([]) == []


def test_get_hybrid_rrf_weights_doubles_original_query_type():
    meta = [
        RankedListMeta(source="fts", query_type="original", query="q"),
        RankedListMeta(source="fts", query_type="lex", query="q lex"),
        RankedListMeta(source="vec", query_type="hyde", query="q hyde"),
    ]
    assert get_hybrid_rrf_weights(meta) == [2.0, 1.0, 1.0]
