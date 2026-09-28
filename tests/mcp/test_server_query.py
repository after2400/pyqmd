import asyncio

import pytest

from pyqmd_mlx.mcp.server import _query_impl, build_server
from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] for _ in texts]


def _fake_rerank(query, documents, model):
    return [0.9 for _ in documents]


def _seed(store):
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    for name, word in (("a", "alpha"), ("b", "bravo")):
        content = f"# {word}\n\nThis document covers {word} topics in depth."
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        store.insert_document(
            name, "doc.md", word, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.index_content(h, content)


@pytest.mark.requires_expansion_weights
def test_query_impl_returns_structured_results():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)

    result = _query_impl(store, "alpha", 10, 0.0, None, None, True)

    assert result.is_error is not True
    assert len(result.structured_content["results"]) >= 1
    assert result.structured_content["results"][0]["docid"].startswith("#")


@pytest.mark.requires_expansion_weights
def test_query_impl_scopes_to_collections_list():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)

    result = _query_impl(store, "alpha OR bravo", 10, 0.0, ["a"], None, True)

    files = [r["file"] for r in result.structured_content["results"]]
    assert all(f.startswith("a/") for f in files)


@pytest.mark.requires_expansion_weights
def test_query_impl_summary_mentions_query_text():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)

    result = _query_impl(store, "alpha", 10, 0.0, None, None, True)

    assert "alpha" in result.content[0].text


@pytest.mark.requires_expansion_weights
def test_query_impl_no_results_returns_clean_summary_not_error():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)

    result = _query_impl(store, "nothing indexed", 10, 0.0, None, None, True)

    assert result.is_error is not True
    assert "No results" in result.content[0].text


@pytest.mark.requires_expansion_weights
def test_query_impl_rerank_false_skips_reranking():
    calls = []

    def _tracking_rerank(query, documents, model):
        calls.append(documents)
        return [0.9 for _ in documents]

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_tracking_rerank)
    _seed(store)

    _query_impl(store, "alpha", 10, 0.0, None, None, False)

    assert calls == []


@pytest.mark.requires_expansion_weights
def test_query_impl_valid_filter_narrows_results():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    for name, status in (("a", "published"), ("b", "draft")):
        content = f"---\nqmd:\n  metadata:\n    status: {status}\n---\n# alpha\n\nalpha content about alpha topics."
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        doc_id = store.insert_document(
            "notes",
            f"{name}.md",
            f"Alpha {status}",
            h,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        )
        store.sync_document_metadata(doc_id, content, f"{name}.md")
        store.index_content(h, content)

    result = _query_impl(
        store,
        "alpha",
        10,
        0.0,
        None,
        None,
        True,
        filter={"key": "status", "operator": "eq", "value": "published"},
    )

    assert result.is_error is not True
    titles = [r["title"] for r in result.structured_content["results"]]
    assert titles == ["Alpha published"]


def test_query_impl_invalid_filter_returns_is_error_not_exception():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)

    result = _query_impl(
        store,
        "alpha",
        10,
        0.0,
        None,
        None,
        True,
        filter={"key": "status", "operator": "bogus", "value": "x"},
    )

    assert result.is_error is True
    assert "unknown operator" in result.content[0].text


@pytest.mark.requires_expansion_weights
def test_query_impl_no_filter_behaves_as_before():
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)

    result = _query_impl(store, "alpha", 10, 0.0, None, None, True)

    assert result.is_error is not True


def _failing_expand(query, model):
    from pyqmd_mlx.llm import ExpansionModelError

    raise ExpansionModelError(f"Could not load query-expansion model '{model}': boom")


def test_query_impl_reports_expansion_model_error():
    store = Store(
        ":memory:",
        embed_fn=_fake_embed,
        rerank_fn=_fake_rerank,
        expand_fn=_failing_expand,
        expand_model="bad/model",
    )
    _seed(store)

    result = _query_impl(
        store, "zebra", limit=5, min_score=0.0, collections=None, intent=None, rerank=True
    )

    assert result.is_error is True
    assert "Could not load query-expansion model 'bad/model'" in result.content[0].text


def test_query_with_searches_fails_loudly():
    server = build_server(Store(":memory:"))
    result = asyncio.run(
        server.call_tool("query", {"query": "x", "searches": [{"type": "lex", "query": "x"}]})
    )
    assert result.is_error is True
    assert "mutually exclusive" in result.content[0].text


def test_query_with_candidate_limit_passes_through_to_store():
    store = Store(
        ":memory:",
        embed_fn=_fake_embed,
        expand_fn=lambda q, m: [q],
        rerank_fn=_fake_rerank,
    )
    _seed(store)
    server = build_server(store)
    result = asyncio.run(server.call_tool("query", {"query": "alpha", "candidateLimit": 1}))
    assert result.is_error is not True
    # candidateLimit bounds the retrieval pool: at most 1 result survives.
    assert len(result.structured_content["results"]) <= 1


@pytest.mark.parametrize("value", [True, False, "10"])
def test_query_candidate_limit_rejects_non_integer_json(monkeypatch, value):
    # CLAUDE.md bool-first rule: a JSON true/false must not coerce to 1/0
    # (Node's zod z.number() rejects both, and strings too).
    from mcp.server.mcpserver.exceptions import ToolError

    from pyqmd_mlx.mcp import server as server_mod

    monkeypatch.setattr(
        server_mod, "_query_impl", lambda *a, **k: pytest.fail("_query_impl must not run")
    )
    server = build_server(Store(":memory:"))
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("query", {"query": "x", "candidateLimit": value}))


@pytest.mark.parametrize("value", [0, -3])
def test_query_candidate_limit_rejects_non_positive(monkeypatch, value):
    # A negative LIMIT is "no limit" in SQLite -- never let it through.
    from pyqmd_mlx.mcp import server as server_mod

    monkeypatch.setattr(
        server_mod, "_query_impl", lambda *a, **k: pytest.fail("_query_impl must not run")
    )
    server = build_server(Store(":memory:"))
    result = asyncio.run(server.call_tool("query", {"query": "x", "candidateLimit": value}))
    assert result.is_error is True
    assert "'candidateLimit' must be a positive integer" in result.content[0].text


def test_query_with_searches_only_fails_loudly():
    server = build_server(Store(":memory:"))
    result = asyncio.run(server.call_tool("query", {"searches": [{"type": "lex", "query": "x"}]}))
    assert result.is_error is True
    assert "typed 'searches' are not supported" in result.content[0].text


def test_query_with_neither_query_nor_searches_fails_loudly():
    server = build_server(Store(":memory:"))
    result = asyncio.run(server.call_tool("query", {}))
    assert result.is_error is True
    assert "provide either 'query'" in result.content[0].text
