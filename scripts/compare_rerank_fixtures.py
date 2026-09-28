"""Compare baseline (Node/GGUF) vs candidate (MLX) rerank scores: rank
correlation, P@k/R@k/MRR/F1 against BEIR qrels, and latency.

Usage: cd python && uv run scripts/compare_rerank_fixtures.py
  [--fixture data/scifact/rerank-fixture.json]
  [--baseline data/scifact/baseline-scores.json]
  [--candidate data/scifact/candidate-scores.json]
"""

import argparse
import json
import math
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def spearman_correlation(scores_a: list[float], scores_b: list[float]) -> float:
    n = len(scores_a)
    if n < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        indexed = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[indexed[k]] = avg_rank
            i = j + 1
        return result

    rank_a = ranks(scores_a)
    rank_b = ranks(scores_b)
    mean_a = sum(rank_a) / n
    mean_b = sum(rank_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    var_a = sum((a - mean_a) ** 2 for a in rank_a)
    var_b = sum((b - mean_b) ** 2 for b in rank_b)
    denom = math.sqrt(var_a * var_b)
    return cov / denom if denom > 0 else float("nan")


def rank_by_score(doc_ids: list[str], scores: list[float]) -> list[str]:
    return [
        doc_id for doc_id, _ in sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)
    ]


def relevant_doc_ids(qrels: dict[str, int], threshold: int = 1) -> list[str]:
    return [doc_id for doc_id, rel in qrels.items() if rel >= threshold]


def score_ranking(ranked_doc_ids: list[str], qrels: dict[str, int], top_k: int) -> dict:
    expected = relevant_doc_ids(qrels)
    if not expected:
        return {
            "precision_at_k": 0.0,
            "recall": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "f1": 0.0,
        }

    def hits_within(k: int) -> int:
        top_k_ids = set(ranked_doc_ids[:k])
        return sum(1 for doc_id in expected if doc_id in top_k_ids)

    hits_at_k = hits_within(top_k)
    matched = sum(1 for doc_id in expected if doc_id in ranked_doc_ids)

    mrr = 0.0
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in expected:
            mrr = 1 / (i + 1)
            break

    denominator = min(top_k, len(expected))
    precision_at_k = hits_at_k / denominator if denominator > 0 else 0.0
    recall = matched / len(expected)
    recall_at_1 = hits_within(1) / len(expected)
    recall_at_3 = hits_within(3) / len(expected)
    recall_at_5 = hits_within(5) / len(expected)
    f1 = (
        (2 * precision_at_k * recall / (precision_at_k + recall))
        if (precision_at_k + recall) > 0
        else 0.0
    )

    return {
        "precision_at_k": precision_at_k,
        "recall": recall,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "f1": f1,
    }


def missing_query_ids(fixture: list[dict], scores: dict) -> list[str]:
    """Query ids present in the fixture but absent from a scores dict.

    A missing entry silently defaults to an empty ranking and 0.0 latency in
    `compare_fixtures()` (so a partial/failed scoring run doesn't crash the
    comparison), but callers must surface this rather than let it pass
    unnoticed -- surface it via this helper instead of masking it.
    """
    return [q["query_id"] for q in fixture if q["query_id"] not in scores]


def compare_fixtures(fixture: list[dict], baseline_scores: dict, candidate_scores: dict) -> dict:
    per_query = {}
    for q in fixture:
        query_id = q["query_id"]
        qrels = {k: int(v) for k, v in q["qrels"].items()}
        baseline_entry = baseline_scores.get(query_id, {"scores": [], "latency_ms": 0.0})
        candidate_entry = candidate_scores.get(query_id, {"scores": [], "latency_ms": 0.0})
        baseline = {s["doc_id"]: s["score"] for s in baseline_entry["scores"]}
        candidate = {s["doc_id"]: s["score"] for s in candidate_entry["scores"]}
        doc_ids = [d["doc_id"] for d in q["candidate_docs"]]

        baseline_vec = [baseline.get(doc_id, 0.0) for doc_id in doc_ids]
        candidate_vec = [candidate.get(doc_id, 0.0) for doc_id in doc_ids]

        top_k = max(len(relevant_doc_ids(qrels)), 10)
        baseline_ranking = rank_by_score(doc_ids, baseline_vec)
        candidate_ranking = rank_by_score(doc_ids, candidate_vec)

        per_query[query_id] = {
            "spearman": spearman_correlation(baseline_vec, candidate_vec),
            "baseline": score_ranking(baseline_ranking[:top_k], qrels, top_k),
            "candidate": score_ranking(candidate_ranking[:top_k], qrels, top_k),
            "baseline_latency_ms": baseline_entry["latency_ms"],
            "candidate_latency_ms": candidate_entry["latency_ms"],
        }
    return per_query


def print_summary(
    per_query: dict,
    baseline_missing: list[str] | None = None,
    candidate_missing: list[str] | None = None,
) -> None:
    n = len(per_query)
    if n == 0:
        print("No queries to compare.")
        return

    baseline_missing = baseline_missing or []
    candidate_missing = candidate_missing or []
    if baseline_missing or candidate_missing:
        print(
            f"WARNING: {len(baseline_missing)} queries missing from baseline scores, "
            f"{len(candidate_missing)} missing from candidate scores"
        )

    spearman_values = [q["spearman"] for q in per_query.values()]
    valid_spearman = [v for v in spearman_values if not math.isnan(v)]
    excluded = len(spearman_values) - len(valid_spearman)
    avg_spearman = sum(valid_spearman) / len(valid_spearman) if valid_spearman else float("nan")
    for label in ("baseline", "candidate"):
        avg_mrr = sum(q[label]["mrr"] for q in per_query.values()) / n
        avg_f1 = sum(q[label]["f1"] for q in per_query.values()) / n
        avg_recall = sum(q[label]["recall"] for q in per_query.values()) / n
        print(f"{label:10s}  MRR={avg_mrr:.3f}  F1={avg_f1:.3f}  Recall={avg_recall:.3f}")

    baseline_latencies = sorted(q["baseline_latency_ms"] for q in per_query.values())
    candidate_latencies = sorted(q["candidate_latency_ms"] for q in per_query.values())
    median_baseline = baseline_latencies[n // 2]
    median_candidate = candidate_latencies[n // 2]
    ratio = median_candidate / median_baseline if median_baseline > 0 else float("inf")
    print(
        f"Median latency: baseline={median_baseline:.1f}ms  candidate={median_candidate:.1f}ms  ratio={ratio:.2f}x"
    )
    print(
        f"Mean Spearman correlation ({n} queries, {excluded} excluded as degenerate): "
        f"{avg_spearman:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_DATA_DIR / "rerank-fixture.json"))
    parser.add_argument("--baseline", default=str(DEFAULT_DATA_DIR / "baseline-scores.json"))
    parser.add_argument("--candidate", default=str(DEFAULT_DATA_DIR / "candidate-scores.json"))
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    baseline_scores = json.loads(Path(args.baseline).read_text())
    candidate_scores = json.loads(Path(args.candidate).read_text())

    baseline_missing = missing_query_ids(fixture, baseline_scores)
    candidate_missing = missing_query_ids(fixture, candidate_scores)

    per_query = compare_fixtures(fixture, baseline_scores, candidate_scores)
    print_summary(per_query, baseline_missing=baseline_missing, candidate_missing=candidate_missing)


if __name__ == "__main__":
    main()
