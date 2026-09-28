import pytest

from pyqmd_mlx.bench._sampling import RerankCache, salted_memo


def test_salted_memo_passes_the_salt_and_memoizes_per_query_and_model():
    calls = []

    def expand(query, model, salt=None):
        calls.append((query, model, salt))
        return [f"lex: {query} {salt}"]

    wrapped = salted_memo(expand, salt="2")

    assert wrapped("a", "m") == ["lex: a 2"]
    assert wrapped("a", "m") == ["lex: a 2"]
    assert wrapped("b", "m") == ["lex: b 2"]
    assert wrapped("a", "other") == ["lex: a 2"]
    assert calls == [("a", "m", "2"), ("b", "m", "2"), ("a", "other", "2")]


def test_salted_memo_does_not_memoize_failures():
    calls = []

    def expand(query, model, salt=None):
        calls.append(query)
        raise RuntimeError("flaky")

    wrapped = salted_memo(expand, salt="1")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            wrapped("a", "m")
    assert calls == ["a", "a"]


def _recording_reranker(log):
    def rerank(query, documents, model):
        log.append(list(documents))
        return [float(len(d)) for d in documents]

    return rerank


def test_recording_always_calls_through_with_the_full_batch():
    log = []
    cache = RerankCache()
    rerank = cache.recording(_recording_reranker(log))

    assert rerank("q", ["aa", "b"], "m") == [2.0, 1.0]
    assert rerank("q", ["aa", "b"], "m") == [2.0, 1.0]
    assert log == [["aa", "b"], ["aa", "b"]]


def test_caching_serves_recorded_scores_and_sends_only_unseen_docs():
    log = []
    inner = _recording_reranker(log)
    cache = RerankCache()
    cache.recording(inner)("q", ["aa", "b"], "m")
    log.clear()

    caching = cache.caching(inner)
    assert caching("q", ["b", "cccc", "aa"], "m") == [1.0, 4.0, 2.0]
    assert log == [["cccc"]]

    assert caching("q", ["cccc", "aa"], "m") == [4.0, 2.0]
    assert log == [["cccc"]]


def test_caching_keys_on_the_query_too():
    log = []
    cache = RerankCache()
    caching = cache.caching(_recording_reranker(log))

    caching("q1", ["aa"], "m")
    caching("q2", ["aa"], "m")

    assert log == [["aa"], ["aa"]]


def test_caching_sends_a_duplicated_unseen_doc_once():
    log = []
    caching = RerankCache().caching(_recording_reranker(log))

    assert caching("q", ["aa", "aa"], "m") == [2.0, 2.0]
    assert log == [["aa"]]
