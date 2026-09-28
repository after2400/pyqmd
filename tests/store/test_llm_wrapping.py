import pytest

from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")


def _store(calls):
    def expand(query, model, salt=None):
        calls.append("base-expand")
        return [f"lex: {query}"]

    def rerank(query, documents, model):
        calls.append("base-rerank")
        return [0.5] * len(documents)

    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=expand, rerank_fn=rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")
    return store


def _tag(name, calls):
    """A wrapper factory that logs `name`, then calls through."""

    def factory(fn):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapped

    return factory


def _query(store):
    # An intent always disables the BM25 strong-signal shortcut, so
    # expansion and reranking are guaranteed to run.
    store.query("authentication", limit=5, intent="docs")


def test_wrappers_apply_inside_and_are_restored_after():
    calls = []
    store = _store(calls)

    with store.wrapping_llm_fns(expand=_tag("e", calls), rerank=_tag("r", calls)):
        _query(store)
    assert calls == ["e", "base-expand", "r", "base-rerank"]

    calls.clear()
    _query(store)
    assert calls == ["base-expand", "base-rerank"]


def test_only_the_given_function_is_wrapped():
    calls = []
    store = _store(calls)

    with store.wrapping_llm_fns(rerank=_tag("r", calls)):
        _query(store)

    assert calls == ["base-expand", "r", "base-rerank"]


def test_restored_when_the_body_raises():
    calls = []
    store = _store(calls)

    with pytest.raises(RuntimeError):
        with store.wrapping_llm_fns(expand=_tag("e", calls)):
            raise RuntimeError("boom")

    _query(store)
    assert calls == ["base-expand", "base-rerank"]


def test_restored_when_a_wrapper_factory_raises():
    calls = []
    store = _store(calls)

    def bad_factory(fn):
        raise ValueError("bad wrapper")

    with pytest.raises(ValueError):
        with store.wrapping_llm_fns(expand=_tag("e", calls), rerank=bad_factory):
            pass

    _query(store)
    assert calls == ["base-expand", "base-rerank"]


def test_nesting_wraps_the_outer_wrapper_and_restores_to_it():
    calls = []
    store = _store(calls)

    with store.wrapping_llm_fns(expand=_tag("outer", calls)):
        with store.wrapping_llm_fns(expand=_tag("inner", calls)):
            _query(store)
        assert calls == ["inner", "outer", "base-expand", "base-rerank"]

        calls.clear()
        _query(store)
        assert calls == ["outer", "base-expand", "base-rerank"]
