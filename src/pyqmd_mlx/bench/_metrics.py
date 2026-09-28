"""Pure IR-metric functions, no I/O. Two families:

- Absolute metrics (reciprocal_rank / mean_reciprocal_rank / recall_at_k /
  ndcg_at_k) score one system against a fixed set of relevant document ids
  -- used in qrels-mode, where curated relevance judgments exist.
- Agreement metrics (top_k_overlap / spearman_rank_correlation) compare two
  systems' own result lists to each other with no ground truth -- used in
  agreement-mode, for a dataset profile with no qrels.
"""

from __future__ import annotations

import math


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    per_query_ranked_ids: list[list[str]], per_query_relevant_ids: list[set[str]]
) -> float:
    if not per_query_ranked_ids:
        return 0.0
    scores = [
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(per_query_ranked_ids, per_query_relevant_ids)
    ]
    return sum(scores) / len(scores)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = set(ranked_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance nDCG: each relevant doc contributes gain 1, scored
    by 1/log2(rank+1), normalized against the ideal ordering (all relevant
    docs first). A relevant id repeated in ranked_ids earns gain only at
    its first occurrence, so the score can never exceed 1."""
    if not relevant_ids:
        return 0.0

    def dcg(ids: list[str]) -> float:
        total = 0.0
        credited: set[str] = set()
        for i, doc_id in enumerate(ids[:k]):
            if doc_id in relevant_ids and doc_id not in credited:
                credited.add(doc_id)
                total += 1.0 / math.log2(i + 2)  # rank is 1-indexed -> log2(rank+1)
        return total

    actual = dcg(ranked_ids)
    ideal_ids = list(relevant_ids)[:k] + [d for d in ranked_ids if d not in relevant_ids]
    ideal = dcg(ideal_ids)
    return actual / ideal if ideal > 0 else 0.0


def top_k_overlap(a_ids: list[str], b_ids: list[str], k: int) -> float:
    a_top = set(a_ids[:k])
    b_top = set(b_ids[:k])
    # Denominator shrinks with however few results either side actually
    # returned -- a fixed `k` made perfect agreement on a query with fewer
    # than k results (e.g. a sparse corpus) mathematically unreachable, so
    # the 0.7 threshold could fail queries that agreed completely.
    denom = min(k, len(a_ids), len(b_ids))
    if denom == 0:
        return 0.0
    return len(a_top & b_top) / denom


def spearman_rank_correlation(a_ids: list[str], b_ids: list[str]) -> float | None:
    """Spearman correlation over the ids both lists share -- ids unique to
    either list are ignored, since there's nothing to compare their rank
    against. Returns None if fewer than 2 ids are shared (correlation is
    undefined for 0 or 1 points)."""
    a_rank_full = {doc_id: i for i, doc_id in enumerate(a_ids)}
    b_rank_full = {doc_id: i for i, doc_id in enumerate(b_ids)}
    common = set(a_rank_full) & set(b_rank_full)

    n = len(common)
    if n < 2:
        return None

    # Ranks must span 0..n-1 over the common subset itself, not the ids'
    # positions in the original (longer) lists -- using original positions
    # let ids unique to one list shift the common ids' apparent rank gaps,
    # producing results outside [-1, 1] and occasionally the wrong sign.
    a_common_order = sorted(common, key=lambda d: a_rank_full[d])
    b_common_order = sorted(common, key=lambda d: b_rank_full[d])
    a_rank = {doc_id: i for i, doc_id in enumerate(a_common_order)}
    b_rank = {doc_id: i for i, doc_id in enumerate(b_common_order)}

    d_squared_sum = sum((a_rank[doc_id] - b_rank[doc_id]) ** 2 for doc_id in common)
    return 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
