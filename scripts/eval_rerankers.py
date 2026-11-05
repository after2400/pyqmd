"""Reranker-only evaluation on SciFact, and the calibration fit for cross-encoders.

Each SciFact test query gets a candidate pool: the BM25 top-N from a small
in-memory index of the corpus (nothing is embedded, your pyqmd index is never
touched) plus every judged-relevant document, so every reranker sees the same
candidates. A reranker is scored by how well it orders that pool (MRR, nDCG@10,
recall@10).

For a cross-encoder the script also fits (bias, temperature) so that
sigmoid((logit - bias) / temperature) reproduces the reference reranker's
(Qwen3-Reranker's) score distribution: the median relevant and median other
document land on the reference's two medians. It prints the constants for
RERANK_CALIBRATION in src/pyqmd_mlx/llm/_constants.py, and how the calibrated
scores compare with the reference's. (Platt scaling was tried first and rejected:
it calibrates to SciFact's ~3% relevant prevalence, which would flatten the
rerank half of the RRF blend.)

Usage: uv run scripts/eval_rerankers.py [--data-dir data/scifact]
  [--models MODEL ...] [--reference MODEL] [--pool 30] [--max-queries N]
  [--intent TEXT] [--out PATH]
Needs data/scifact (uv run scripts/prepare_scifact_corpus.py) and the models in
the Hugging Face cache.
"""

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from pyqmd_mlx.bench._metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"
K = 10


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def match_medians(
    relevant_logits: list[float],
    other_logits: list[float],
    target_relevant: float,
    target_other: float,
) -> tuple[float, float]:
    """(bias, temperature) such that sigmoid((logit - bias) / temperature) maps
    the median relevant logit to `target_relevant` and the median other logit to
    `target_other`, i.e. a cross-encoder's score distribution is made to match a
    reference reranker's (Qwen3-Reranker's: relevant ~1, others ~0), which is the
    scale the RRF blend and --min-score assume."""
    if not relevant_logits or not other_logits:
        raise ValueError("match_medians needs both relevant and other examples")
    if not 0.0 < target_other < target_relevant < 1.0:
        raise ValueError(
            f"targets must satisfy 0 < other < relevant < 1, got {target_other}, {target_relevant}"
        )
    median_relevant = statistics.median(relevant_logits)
    median_other = statistics.median(other_logits)
    if median_relevant <= median_other:
        raise ValueError(
            f"median relevant logit {median_relevant:.3f} must exceed median other {median_other:.3f}"
        )
    temperature = (median_relevant - median_other) / (
        _logit(target_relevant) - _logit(target_other)
    )
    bias = median_relevant - _logit(target_relevant) * temperature
    return bias, temperature


def summarize(rankings: dict[str, list[str]], qrels: dict[str, set[str]], k: int = K) -> dict:
    """Mean MRR, nDCG@k and recall@k over the ranked queries."""
    ids = list(rankings)
    if not ids:
        return {"mrr": 0.0, f"ndcg_at_{k}": 0.0, f"recall_at_{k}": 0.0}
    ranked = [rankings[q] for q in ids]
    relevant = [qrels[q] for q in ids]
    return {
        "mrr": mean_reciprocal_rank(ranked, relevant),
        f"ndcg_at_{k}": sum(ndcg_at_k(r, g, k) for r, g in zip(ranked, relevant)) / len(ids),
        f"recall_at_{k}": sum(recall_at_k(r, g, k) for r, g in zip(ranked, relevant)) / len(ids),
    }


def load_scifact(data_dir: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """({query id: text}, {query id: relevant doc ids}) for the judged test queries."""
    qrels: dict[str, set[str]] = {}
    for line in (data_dir / "qrels-test.tsv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        query_id, doc_id, score = line.split("\t")
        if int(score) >= 1:
            qrels.setdefault(query_id, set()).add(doc_id)
    queries = {}
    for line in (data_dir / "queries.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]
    return queries, {q: qrels[q] for q in queries}


def read_doc(data_dir: Path, doc_id: str) -> str:
    return (data_dir / "corpus" / f"{doc_id}.md").read_text()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Bm25Index:
    """Okapi BM25 over whole documents, scoring a document by any of the
    query's terms. (pyqmd's own FTS joins query terms with AND, which a long
    claim almost never satisfies, so it can't pool candidates for SciFact.)"""

    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        for position, doc_id in enumerate(self.ids):
            counts: dict[str, int] = {}
            tokens = _tokens(docs[doc_id])
            self.lengths.append(len(tokens))
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                self.postings.setdefault(token, []).append((position, count))
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def top(self, query: str, n: int) -> list[str]:
        """The ids of the n best-scoring documents (those matching no term are omitted)."""
        scores: dict[int, float] = {}
        total = len(self.ids)
        for term in set(_tokens(query)):
            postings = self.postings.get(term, [])
            idf = math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, count in postings:
                norm = 1.0 - self.b + self.b * self.lengths[position] / self.average_length
                scores[position] = scores.get(position, 0.0) + idf * (
                    count * (self.k1 + 1.0) / (count + self.k1 * norm)
                )
        best = sorted(scores, key=lambda p: (-scores[p], p))[:n]
        return [self.ids[p] for p in best]


def build_index(data_dir: Path) -> Bm25Index:
    corpus_dir = data_dir / "corpus"
    return Bm25Index({path.stem: path.read_text() for path in sorted(corpus_dir.glob("*.md"))})


def build_pool(index: Bm25Index, query: str, gold_ids: set[str], size: int) -> list[str]:
    """BM25 top-`size` document ids, then any gold id the BM25 pass missed."""
    pool = index.top(query, size)
    pool.extend(sorted(gold_ids - set(pool)))
    return pool


def evaluate_model(
    scorer: Callable[[str, list[str]], list[float]],
    queries: dict[str, str],
    pools: dict[str, list[str]],
    texts: dict[str, str],
    qrels: dict[str, set[str]],
    intent: str | None = None,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Score every pooled document; return per-document rows and each query's ranking."""
    rows: list[dict] = []
    rankings: dict[str, list[str]] = {}
    for query_id, doc_ids in pools.items():
        query = f"{intent}\n\n{queries[query_id]}" if intent else queries[query_id]
        scores = scorer(query, [texts[d] for d in doc_ids])
        for doc_id, score in zip(doc_ids, scores):
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "label": int(doc_id in qrels[query_id]),
                    "score": score,
                }
            )
        order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])
        rankings[query_id] = [doc_ids[i] for i in order]
    return rows, rankings


def make_scorer(model_id: str) -> tuple[Callable[[str, list[str]], list[float]], bool]:
    """(scorer, returns_raw_logits). Cross-encoders are scored uncalibrated so
    their logits can be fitted; other rerankers return their own 0..1 scores."""
    from pyqmd_mlx.llm import _cross_encoder, rerank

    if _cross_encoder.is_cross_encoder(model_id):
        encoder = _cross_encoder.load_cross_encoder(model_id)
        return (lambda q, docs: _cross_encoder.score_logits(encoder, q, docs)), True
    return (lambda q, docs: rerank(q, docs, model_id)), False


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _share_above(values: list[float], threshold: float = 0.5) -> float:
    return sum(v > threshold for v in values) / len(values)


def main() -> None:
    from pyqmd_mlx.llm import DEFAULT_RERANK_MODEL
    from pyqmd_mlx.llm._constants import GRANITE_RERANK_MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_RERANK_MODEL, GRANITE_RERANK_MODEL])
    parser.add_argument(
        "--reference",
        default=DEFAULT_RERANK_MODEL,
        help="the reranker whose score distribution cross-encoders are calibrated to",
    )
    parser.add_argument("--pool", type=int, default=30, help="BM25 candidates per query")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--intent", default=None, help="prefix each query like --intent does")
    parser.add_argument("--out", type=Path, default=None, help="write per-document scores (JSONL)")
    args = parser.parse_args()

    queries, qrels = load_scifact(args.data_dir)
    query_ids = list(queries)[: args.max_queries]
    queries = {q: queries[q] for q in query_ids}
    index = build_index(args.data_dir)
    pools = {q: build_pool(index, queries[q], qrels[q], args.pool) for q in query_ids}
    texts = {d: read_doc(args.data_dir, d) for pool in pools.values() for d in pool}
    print(f"{len(queries)} queries, {sum(len(p) for p in pools.values())} pooled pairs", flush=True)

    results: dict[str, tuple[list[dict], bool]] = {}
    for model_id in args.models:
        scorer, raw = make_scorer(model_id)
        start = time.monotonic()
        rows, rankings = evaluate_model(scorer, queries, pools, texts, qrels, args.intent)
        seconds = time.monotonic() - start
        results[model_id] = (rows, raw)
        metrics = summarize(rankings, qrels)
        print(f"\n{model_id}")
        print(
            "  "
            + "  ".join(f"{name}={value:.4f}" for name, value in metrics.items())
            + f"  ({seconds / len(queries):.2f} s/query)"
        )

    def split(model_id: str) -> tuple[list[float], list[float]]:
        rows = results[model_id][0]
        return (
            [r["score"] for r in rows if r["label"]],
            [r["score"] for r in rows if not r["label"]],
        )

    if args.reference in results and not results[args.reference][1]:
        reference_relevant, reference_other = split(args.reference)
        target_relevant, target_other = _median(reference_relevant), _median(reference_other)
        print(f"\nreference {args.reference}")
        print(
            f"  median score: relevant={target_relevant:.4f} other={target_other:.4f}; "
            f"share above 0.5: relevant={_share_above(reference_relevant):.3f} "
            f"other={_share_above(reference_other):.3f}"
        )
        for model_id, (_rows, raw) in results.items():
            if not raw:
                continue
            relevant, other = split(model_id)
            bias, temperature = match_medians(relevant, other, target_relevant, target_other)
            print(f"\ncalibration for {model_id}")
            print(f'  "{model_id}": ({bias:.3f}, {temperature:.4f})')
            for label, values in (("relevant", relevant), ("other", other)):
                calibrated = [_sigmoid((x - bias) / temperature) for x in values]
                print(
                    f"  calibrated {label}: median={_median(calibrated):.4f} "
                    f"share above 0.5={_share_above(calibrated):.3f}"
                )
    else:
        print("\n(no calibration: the reference reranker was not among --models)")

    if args.out:
        out_rows = [{"model": m, **r} for m, (rows, _raw) in results.items() for r in rows]
        args.out.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
        print(f"\nwrote {len(out_rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
