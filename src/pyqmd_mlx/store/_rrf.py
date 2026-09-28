"""Reciprocal Rank Fusion, ported from store.ts's reciprocalRankFusion /
getHybridRrfWeights. Pure scoring logic, no DB.
"""

from ._types import RankedListMeta, RankedResult


def reciprocal_rank_fusion(
    result_lists: list[list[RankedResult]],
    weights: list[float] = (),
    k: int = 60,
) -> list[RankedResult]:
    """Merge ranked lists by reciprocal rank, with a small top-rank bonus."""
    scores: dict[str, dict] = {}

    for list_idx, result_list in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, result in enumerate(result_list):
            contribution = weight / (k + rank + 1)
            existing = scores.get(result.file)
            if existing:
                existing["rrf_score"] += contribution
                existing["top_rank"] = min(existing["top_rank"], rank)
            else:
                scores[result.file] = {
                    "result": result,
                    "rrf_score": contribution,
                    "top_rank": rank,
                }

    for entry in scores.values():
        if entry["top_rank"] == 0:
            entry["rrf_score"] += 0.05
        elif entry["top_rank"] <= 2:
            entry["rrf_score"] += 0.02

    ordered = sorted(scores.values(), key=lambda e: e["rrf_score"], reverse=True)
    return [
        RankedResult(
            file=e["result"].file,
            display_path=e["result"].display_path,
            title=e["result"].title,
            body=e["result"].body,
            score=e["rrf_score"],
        )
        for e in ordered
    ]


def get_hybrid_rrf_weights(ranked_list_meta: list[RankedListMeta]) -> list[float]:
    """Original-query lists get 2x weight; expansion-derived lists (lex/vec/
    hyde) stay at 1x regardless of position."""
    return [2.0 if meta.query_type == "original" else 1.0 for meta in ranked_list_meta]
