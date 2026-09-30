import pytest

from pyqmd_mlx.bench._fixture import BenchFixture, BenchQuery
from pyqmd_mlx.bench.runner import BACKENDS, all_zero, check_collection_ready, run_benchmark
from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _fake_expand(query, model):
    return [f"lex: {query} setup", f"vec: how to {query}"]


def _fake_rerank(query, documents, model):
    return [1.0 if query.lower() in doc.lower() else 0.1 for doc in documents]


def _make_store():
    return Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand, rerank_fn=_fake_rerank)


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")


def _fixture_with_one_query(expected_files, expected_in_top_k=5):
    return BenchFixture(
        description="test",
        version=1,
        collection=None,
        queries=[
            BenchQuery(
                id="q1",
                query="authentication",
                type="exact",
                description="",
                expected_files=expected_files,
                expected_in_top_k=expected_in_top_k,
            )
        ],
    )


def test_check_collection_ready_raises_when_no_documents_indexed():
    store = Store(":memory:", embed_fn=_fake_embed)

    with pytest.raises(ValueError, match="No indexed documents"):
        check_collection_ready(store, None)


def test_check_collection_ready_raises_for_unknown_collection():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication guide")

    with pytest.raises(ValueError, match="Collection not found"):
        check_collection_ready(store, "nonexistent")


def test_check_collection_ready_passes_for_indexed_collection():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication guide")

    check_collection_ready(store, "notes")  # must not raise


def test_run_benchmark_scores_all_four_backends():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "pasta recipe instructions")
    fixture = _fixture_with_one_query(["auth.md"])

    result = run_benchmark(store, fixture, None)

    assert len(result.results) == 1
    assert set(result.results[0].backends.keys()) == set(BACKENDS)
    for name in BACKENDS:
        score = result.results[0].backends[name]
        assert 0.0 <= score.recall_at_k <= 1.0
        assert 0.0 <= score.mrr <= 1.0
        assert 0.0 <= score.ndcg_at_k <= 1.0
        assert score.latency_ms >= 0.0
    assert set(result.summary.keys()) == set(BACKENDS)


def test_run_benchmark_scores_full_backend_perfectly_for_a_clear_match():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "pasta recipe instructions")
    fixture = _fixture_with_one_query(["auth.md"], expected_in_top_k=1)

    result = run_benchmark(store, fixture, None)

    full_score = result.results[0].backends["full"]
    assert full_score.recall_at_k == 1.0
    assert full_score.mrr == 1.0


def test_run_benchmark_never_credits_one_expected_file_twice():
    # Regression for a real run that reported bm25 nDCG@k = 1.63: a decoy
    # file whose name merely ends with the expected filename was matched
    # too, crediting the single expected file twice.
    store = _make_store()
    store.add_collection("memory", "/memory")
    _seed_doc(store, "memory", "areas-alpha-notes-md.md", "Alpha", "authentication on alpha")
    _seed_doc(
        store,
        "memory",
        "projects-0a1b2c3d-areas-alpha-notes-md.md",
        "Alpha project",
        "authentication on alpha project notes",
    )
    fixture = _fixture_with_one_query(["areas-alpha-notes-md.md"])

    result = run_benchmark(store, fixture, None)

    for name in BACKENDS:
        score = result.results[0].backends[name]
        assert score.ndcg_at_k <= 1.0, name
        assert score.recall_at_k <= 1.0, name
        assert score.mrr <= 1.0, name


def test_run_benchmark_scores_zero_when_backend_raises(monkeypatch):
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")
    fixture = _fixture_with_one_query(["auth.md"])

    def _boom(*args, **kwargs):
        raise RuntimeError("embeddings unavailable")

    monkeypatch.setattr(store, "search_vec", _boom)

    result = run_benchmark(store, fixture, None)  # must not raise

    vector_score = result.results[0].backends["vector"]
    assert vector_score.recall_at_k == 0.0
    assert vector_score.mrr == 0.0
    assert vector_score.ndcg_at_k == 0.0


def test_all_zero_true_when_every_backend_scores_zero():
    summary = {
        "bm25": {
            "avg_recall_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_ndcg_at_k": 0.0,
            "avg_latency_ms": 1.0,
        }
    }
    assert all_zero(summary) is True


def test_all_zero_false_when_any_backend_scores_nonzero():
    summary = {
        "bm25": {
            "avg_recall_at_k": 1.0,
            "avg_mrr": 1.0,
            "avg_ndcg_at_k": 1.0,
            "avg_latency_ms": 1.0,
        }
    }
    assert all_zero(summary) is False


def test_all_zero_false_for_empty_summary():
    assert all_zero({}) is False
