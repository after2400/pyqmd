"""Orchestrates a bench run: for each fixture query, calls all 4 search
backends against the Store, scores results with pyqmd_mlx.bench._metrics, and
aggregates per-backend averages. Ported from Node's src/bench/bench.ts,
minus structured-query support (see design spec) and using pyqmd's own
existing IR metrics rather than Node's bespoke precision/recall@1,3,5/F1
set."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial

from pyqmd_mlx.bench._fixture import BenchFixture
from pyqmd_mlx.bench._metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from pyqmd_mlx.bench._pathmatch import canonicalize_ranked_ids
from pyqmd_mlx.bench._sampling import RerankCache, salted_memo
from pyqmd_mlx.llm import ExpansionModelError
from pyqmd_mlx.store import Store

BACKENDS = ("bm25", "vector", "hybrid", "full")
# Backends that expand the query, so their scores vary with the seed.
SAMPLED_BACKENDS = ("hybrid", "full")
_METRICS = ("recall_at_k", "mrr", "ndcg_at_k")


@dataclass
class BackendScore:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_ms: float


@dataclass
class QueryResult:
    id: str
    query: str
    backends: dict[str, BackendScore] = field(default_factory=dict)
    # bench --samples > 1 only: every draw for SAMPLED_BACKENDS, sample 0
    # first. backends then holds their means (latency from sample 0).
    samples: dict[str, list[BackendScore]] | None = None


@dataclass
class BenchResult:
    fixture: str
    collection: str | None
    timestamp: str
    results: list[QueryResult]
    summary: dict[str, dict[str, float]]
    samples: int = 1


def _dedupe(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _run_bm25(store: Store, query: str, limit: int, collection: str | None) -> list[str]:
    return [r.filepath for r in store.search_fts(query, limit=limit, collection=collection)]


def _run_vector(store: Store, query: str, limit: int, collection: str | None) -> list[str]:
    return [r.filepath for r in store.search_vec(query, limit=limit, collection=collection)]


def _run_hybrid(store: Store, query: str, limit: int, collection: str | None) -> list[str]:
    results = store.query(query, limit=limit, collection=collection, skip_rerank=True)
    return [r.file for r in results]


def _run_full(store: Store, query: str, limit: int, collection: str | None) -> list[str]:
    results = store.query(query, limit=limit, collection=collection, skip_rerank=False)
    return [r.file for r in results]


_BACKEND_RUNNERS = {
    "bm25": _run_bm25,
    "vector": _run_vector,
    "hybrid": _run_hybrid,
    "full": _run_full,
}


def check_collection_ready(store: Store, collection: str | None) -> None:
    """Fail fast, before running any queries, if there's nothing to search
    -- mirrors Node's assertBenchCollectionReady, avoiding a wall of
    all-zero scores when the real problem is an empty/missing index."""
    stats = store.get_collection_document_stats()

    if collection:
        names = [c["name"] for c in store.list_collections()]
        if collection not in names:
            hint = (
                f"Available: {', '.join(names)}. Run 'pyqmd ls' to inspect."
                if names
                else "Run 'pyqmd ls' to see available collections."
            )
            raise ValueError(f"Collection not found: {collection}\n{hint}")
        if stats.get(collection, {"count": 0})["count"] == 0:
            raise ValueError(
                f"Collection '{collection}' has no indexed documents.\n"
                f"Run 'pyqmd update', then 'pyqmd ls {collection}' to confirm "
                "files are indexed before bench."
            )
        return

    total = sum(s["count"] for s in stats.values())
    if total == 0:
        raise ValueError(
            "No indexed documents found.\n"
            "Index a collection with 'pyqmd collection add' / 'pyqmd update' "
            "before running bench."
        )


def _score_backend(store: Store, backend_name: str, query, collection: str | None) -> BackendScore:
    limit = max(query.expected_in_top_k, 10)
    start = time.monotonic()
    try:
        raw_paths = _BACKEND_RUNNERS[backend_name](store, query.query, limit, collection)
    except ExpansionModelError:
        # A misconfigured expansion model is a setup error, not one backend
        # being unavailable: scoring it 0 would silently skew every
        # hybrid/full result, so let the command report it and exit.
        raise
    except Exception:
        # A single backend being unavailable (e.g. no embeddings yet) must
        # not abort the whole run -- score it 0 and keep going, matching
        # Node's own try/catch-per-backend resilience.
        return BackendScore(0.0, 0.0, 0.0, (time.monotonic() - start) * 1000)

    latency_ms = (time.monotonic() - start) * 1000
    ranked_ids = canonicalize_ranked_ids(_dedupe(raw_paths), query.expected_files)
    relevant_ids = set(query.expected_files)

    return BackendScore(
        recall_at_k=recall_at_k(ranked_ids, relevant_ids, query.expected_in_top_k),
        mrr=reciprocal_rank(ranked_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevant_ids, query.expected_in_top_k),
        latency_ms=latency_ms,
    )


def _compute_summary(query_results: list[QueryResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in BACKENDS:
        scores = [qr.backends[name] for qr in query_results if name in qr.backends]
        if not scores:
            continue
        summary[name] = {
            "avg_recall_at_k": sum(s.recall_at_k for s in scores) / len(scores),
            "avg_mrr": sum(s.mrr for s in scores) / len(scores),
            "avg_ndcg_at_k": sum(s.ndcg_at_k for s in scores) / len(scores),
            "avg_latency_ms": sum(s.latency_ms for s in scores) / len(scores),
        }
    return summary


def _score_query(store: Store, query, collection: str | None) -> QueryResult:
    backends = {name: _score_backend(store, name, query, collection) for name in BACKENDS}
    return QueryResult(id=query.id, query=query.query, backends=backends)


def _mean_score(draws: list[BackendScore]) -> BackendScore:
    """Metric means across samples; latency from sample 0, the only run
    with no cache hits."""
    n = len(draws)
    return BackendScore(
        recall_at_k=sum(d.recall_at_k for d in draws) / n,
        mrr=sum(d.mrr for d in draws) / n,
        ndcg_at_k=sum(d.ndcg_at_k for d in draws) / n,
        latency_ms=draws[0].latency_ms,
    )


def _sample_spread(query_results: list[QueryResult], samples: int) -> dict[str, dict[str, float]]:
    """min_*/max_* of each sample's fixture-wide average. Per-query min/max
    would just be 0..1; this is the spread that tells a real difference
    between two configurations from seed noise."""
    spread: dict[str, dict[str, float]] = {}
    if not query_results:
        return spread
    for name in SAMPLED_BACKENDS:
        entry: dict[str, float] = {}
        for metric in _METRICS:
            averages = [
                sum(getattr(qr.samples[name][i], metric) for qr in query_results)
                / len(query_results)
                for i in range(samples)
            ]
            entry[f"min_{metric}"] = min(averages)
            entry[f"max_{metric}"] = max(averages)
        spread[name] = entry
    return spread


def _run_samples(
    store: Store,
    fixture: BenchFixture,
    collection: str | None,
    samples: int,
    on_sample: Callable[[int, int], None] | None,
) -> list[QueryResult]:
    """Sample 0 is a plain run whose rerank scores are recorded; samples
    1..N-1 re-run SAMPLED_BACKENDS with seed salt str(i) and cached rerank
    scores (see docs/specs/2026-09-25-bench-samples-design.md)."""
    rerank_cache = RerankCache()

    if on_sample:
        on_sample(1, samples)
    with store.wrapping_llm_fns(rerank=rerank_cache.recording):
        query_results = [_score_query(store, q, collection) for q in fixture.queries]
    for qr in query_results:
        qr.samples = {name: [qr.backends[name]] for name in SAMPLED_BACKENDS}

    for i in range(1, samples):
        if on_sample:
            on_sample(i + 1, samples)
        with store.wrapping_llm_fns(
            expand=partial(salted_memo, salt=str(i)), rerank=rerank_cache.caching
        ):
            for query, qr in zip(fixture.queries, query_results):
                for name in SAMPLED_BACKENDS:
                    qr.samples[name].append(_score_backend(store, name, query, collection))

    for qr in query_results:
        for name in SAMPLED_BACKENDS:
            qr.backends[name] = _mean_score(qr.samples[name])
    return query_results


def run_benchmark(
    store: Store,
    fixture: BenchFixture,
    collection: str | None,
    samples: int = 1,
    on_sample: Callable[[int, int], None] | None = None,
) -> BenchResult:
    if samples == 1:
        query_results = [_score_query(store, q, collection) for q in fixture.queries]
        summary = _compute_summary(query_results)
    else:
        query_results = _run_samples(store, fixture, collection, samples, on_sample)
        # Over the per-query means, avg_* is also the mean of the per-sample
        # fixture averages (every sample scores the same queries).
        summary = _compute_summary(query_results)
        for name, spread in _sample_spread(query_results, samples).items():
            summary[name].update(spread)

    return BenchResult(
        fixture="",
        collection=collection,
        timestamp=datetime.now(UTC).isoformat(),
        results=query_results,
        summary=summary,
        samples=samples,
    )


def all_zero(summary: dict[str, dict[str, float]]) -> bool:
    if not summary:
        return False
    return all(
        s["avg_recall_at_k"] == 0 and s["avg_mrr"] == 0 and s["avg_ndcg_at_k"] == 0
        for s in summary.values()
    )


def result_to_dict(result: BenchResult) -> dict:
    d = asdict(result)
    if result.samples == 1:
        # --samples 1 JSON stays byte-identical to before the flag existed.
        del d["samples"]
        for qr in d["results"]:
            del qr["samples"]
    return d
