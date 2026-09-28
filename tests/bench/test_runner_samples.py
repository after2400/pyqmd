from collections import Counter
from types import SimpleNamespace

import pytest

from pyqmd_mlx.bench._fixture import BenchFixture, BenchQuery
from pyqmd_mlx.bench.runner import SAMPLED_BACKENDS, result_to_dict, run_benchmark
from pyqmd_mlx.llm import ExpansionModelError
from pyqmd_mlx.store import Store


class _SamplingStore:
    """Stand-in for Store: query() expands (each lex: line names a file),
    then reranks unless skip_rerank, through swappable functions. A real
    Store can skip expansion on a strong BM25 hit; this one never does."""

    wrapping_llm_fns = Store.wrapping_llm_fns

    def __init__(self, expand_fn, rerank_fn):
        self._expand_fn = expand_fn
        self._rerank_fn = rerank_fn
        self.calls = Counter()

    def search_fts(self, query, limit, collection=None):
        self.calls["fts"] += 1
        return [SimpleNamespace(filepath="hit.md")]

    def search_vec(self, query, limit, collection=None):
        self.calls["vec"] += 1
        return [SimpleNamespace(filepath="miss.md")]

    def query(self, query, limit, collection=None, skip_rerank=False):
        files = [line.split(": ", 1)[1] for line in self._expand_fn(query, "fake-model")]
        if not skip_rerank:
            scores = self._rerank_fn(query, files, "fake-rerank")
            files = [f for _, f in sorted(zip(scores, files), key=lambda t: -t[0])]
        return [SimpleNamespace(file=f) for f in files]


def _fixture(*queries):
    return BenchFixture(
        description="test",
        version=1,
        collection=None,
        queries=[
            BenchQuery(
                id=q,
                query=q,
                type="exact",
                description="",
                expected_files=["hit.md"],
                expected_in_top_k=1,
            )
            for q in queries
        ],
    )


def _expander(log, miss_salts=("1",), miss_queries=("alpha",)):
    """Finds hit.md, except for a query in miss_queries under a salt in
    miss_salts, which finds miss.md."""

    def expand(query, model, salt=None):
        log.append((query, salt))
        target = "miss.md" if query in miss_queries and salt in miss_salts else "hit.md"
        return [f"lex: {target}"]

    return expand


def _reranker(log):
    def rerank(query, documents, model):
        log.append(list(documents))
        return [1.0] * len(documents)

    return rerank


def test_one_sample_keeps_todays_dict_shape():
    store = _SamplingStore(_expander([]), _reranker([]))

    result = run_benchmark(store, _fixture("alpha"), None)

    d = result_to_dict(result)
    assert list(d) == ["fixture", "collection", "timestamp", "results", "summary"]
    assert list(d["results"][0]) == ["id", "query", "backends"]
    assert list(d["summary"]["hybrid"]) == [
        "avg_recall_at_k",
        "avg_mrr",
        "avg_ndcg_at_k",
        "avg_latency_ms",
    ]


def test_one_sample_never_calls_on_sample():
    seen = []
    store = _SamplingStore(_expander([]), _reranker([]))

    run_benchmark(store, _fixture("alpha"), None, on_sample=lambda i, n: seen.append(i))

    assert seen == []


def test_per_query_scores_are_means_with_every_draw_kept():
    store = _SamplingStore(_expander([]), _reranker([]))

    result = run_benchmark(store, _fixture("alpha"), None, samples=3)

    qr = result.results[0]
    assert [s.recall_at_k for s in qr.samples["hybrid"]] == [1.0, 0.0, 1.0]
    assert qr.backends["hybrid"].recall_at_k == pytest.approx(2 / 3)
    assert qr.backends["hybrid"].latency_ms == qr.samples["hybrid"][0].latency_ms
    assert set(qr.samples) == set(SAMPLED_BACKENDS)
    assert qr.backends["bm25"].recall_at_k == 1.0


def test_summary_ranges_are_over_per_sample_fixture_averages():
    store = _SamplingStore(_expander([]), _reranker([]))

    result = run_benchmark(store, _fixture("alpha", "beta"), None, samples=3)

    hybrid = result.summary["hybrid"]
    # Per-sample fixture averages: 1.0, 0.5, 1.0 -- not alpha's own 0..1.
    assert hybrid["avg_recall_at_k"] == pytest.approx(5 / 6)
    assert hybrid["min_recall_at_k"] == 0.5
    assert hybrid["max_recall_at_k"] == 1.0
    assert "min_recall_at_k" not in result.summary["bm25"]
    assert result.samples == 3


def test_sampled_dict_has_the_new_keys():
    store = _SamplingStore(_expander([]), _reranker([]))

    d = result_to_dict(run_benchmark(store, _fixture("alpha"), None, samples=2))

    assert d["samples"] == 2
    assert len(d["results"][0]["samples"]["full"]) == 2
    assert "max_ndcg_at_k" in d["summary"]["full"]


def test_bm25_and_vector_run_once_per_query():
    store = _SamplingStore(_expander([]), _reranker([]))

    run_benchmark(store, _fixture("alpha", "beta"), None, samples=4)

    assert store.calls == Counter({"fts": 2, "vec": 2})


def test_later_samples_expand_once_per_query_with_their_salt():
    log = []
    store = _SamplingStore(_expander(log), _reranker([]))

    run_benchmark(store, _fixture("alpha"), None, samples=3)

    # Sample 0 is today's run: hybrid and full each expand, unsalted.
    assert log == [("alpha", None), ("alpha", None), ("alpha", "1"), ("alpha", "2")]


def test_later_samples_only_rerank_unseen_docs():
    log = []
    store = _SamplingStore(_expander([]), _reranker(log))

    run_benchmark(store, _fixture("alpha"), None, samples=3)

    # Sample 0 reranks hit.md; sample 1 finds miss.md (new); sample 2 finds
    # hit.md again, already scored.
    assert log == [["hit.md"], ["miss.md"]]


def test_on_sample_reports_each_sample():
    seen = []
    store = _SamplingStore(_expander([]), _reranker([]))

    run_benchmark(
        store, _fixture("alpha"), None, samples=3, on_sample=lambda i, n: seen.append((i, n))
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_backend_failing_in_one_sample_scores_zero_there_only():
    def expand(query, model, salt=None):
        if salt == "1":
            raise RuntimeError("flaky")
        return ["lex: hit.md"]

    store = _SamplingStore(expand, _reranker([]))

    result = run_benchmark(store, _fixture("alpha"), None, samples=3)

    assert [s.recall_at_k for s in result.results[0].samples["full"]] == [1.0, 0.0, 1.0]


def test_expansion_model_error_in_a_later_sample_propagates():
    def expand(query, model, salt=None):
        if salt == "2":
            raise ExpansionModelError("bad model")
        return ["lex: hit.md"]

    store = _SamplingStore(expand, _reranker([]))

    with pytest.raises(ExpansionModelError):
        run_benchmark(store, _fixture("alpha"), None, samples=3)
