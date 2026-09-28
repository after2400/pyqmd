"""Fast, real-dispatch coverage for the MCP server.

Every other fast test in this suite calls the `_status_impl`/`_get_impl`/
`_multi_get_impl`/`_query_impl`/`_read_document_impl` functions directly,
bypassing the official mcp SDK's own request dispatch entirely: its
exception-wrapping (ToolError/ResourceError vs. Unexpected*Error), schema
generation, and result conversion never run. Only the slow, real-model
`test_e2e_real_models.py` ever drives a tool through genuine SDK dispatch
(and only the `query` tool, via a subprocess + stdio client).

This module closes that gap for all four tools plus the resource, in-process
and without any real MLX models: `MCPServer.call_tool()` and
`MCPServer.read_resource()` (both async, see `mcp.server.mcpserver.MCPServer`)
are the SDK's own dispatch entry points -- the same methods the stdio/HTTP
transports call after deserializing a JSON-RPC request. Driving them
directly here is a genuine round-trip through the SDK's tool/resource
managers, just without the transport and wire serialization on top.

This is also the regression test for the qmd://{+path} resource's not-found
path: before the fix, `_read_document_impl` raised a plain `ValueError`,
which `MCPServer.read_resource()` does not recognize as an anticipated
failure, so it got wrapped in `UnexpectedResourceError` -- discarding the
"Document not found" message and logging a full ERROR-level traceback for a
routine condition. `test_resource_missing_document_raises_resource_not_found_error`
below fails against that old behavior and passes against the fix (raising
`ResourceNotFoundError` instead); see the fix-report for the manual
before/after check.
"""

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from pyqmd_mlx.mcp.server import build_server
from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] for _ in texts]


def _fake_rerank(query, documents, model):
    return [0.9 for _ in documents]


def _make_store():
    return Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body)


def _build_seeded_server():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "Alpha", "# Alpha\n\nline one\nline two")
    return build_server(store)


def test_list_resources_is_empty_read_only_no_list_but_templates_are_not():
    server = _build_seeded_server()

    resources = asyncio.run(server.list_resources())
    templates = asyncio.run(server.list_resource_templates())

    assert resources == []
    assert len(templates) >= 1
    assert templates[0].uri_template == "qmd://{+path}"


def test_status_tool_via_real_dispatch_reports_no_error():
    server = _build_seeded_server()

    result = asyncio.run(server.call_tool("status", {}))

    assert result.is_error is not True
    assert "qmd Index Status" in result.content[0].text


def test_get_tool_via_real_dispatch_returns_resource_shaped_result():
    server = _build_seeded_server()

    result = asyncio.run(server.call_tool("get", {"file": "notes/a.md"}))

    assert result.is_error is not True
    assert result.content[0].type == "resource"
    assert result.content[0].resource.uri == "qmd://notes/a.md"
    assert "line one" in result.content[0].resource.text


def test_get_tool_via_real_dispatch_reports_clean_is_error_for_missing_document():
    server = _build_seeded_server()

    result = asyncio.run(server.call_tool("get", {"file": "notes/missing.md"}))

    assert result.is_error is True
    assert "not found" in result.content[0].text.lower()


def test_multi_get_tool_via_real_dispatch_returns_resource_shaped_result():
    server = _build_seeded_server()

    result = asyncio.run(server.call_tool("multi_get", {"pattern": "notes/a.md"}))

    assert result.is_error is not True
    assert len(result.content) == 1
    assert result.content[0].type == "resource"


@pytest.mark.requires_expansion_weights
def test_query_tool_via_real_dispatch_returns_structured_content():
    server = _build_seeded_server()

    result = asyncio.run(server.call_tool("query", {"query": "alpha"}))

    assert result.is_error is not True
    assert result.structured_content is not None
    assert len(result.structured_content["results"]) >= 1


@pytest.mark.requires_expansion_weights
def test_query_tool_via_real_dispatch_with_filter_returns_only_matching_results():
    store = _make_store()
    store.add_collection("notes", "/notes")
    published = "---\nqmd:\n  metadata:\n    status: published\n---\n# Alpha\n\nalpha content here"
    draft = "---\nqmd:\n  metadata:\n    status: draft\n---\n# Alpha\n\nalpha content here too"
    for path, content in (("a.md", published), ("b.md", draft)):
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        doc_id = store.insert_document(
            "notes", path, "Alpha", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.sync_document_metadata(doc_id, content, path)
        store.index_content(h, content)
    server = build_server(store)

    result = asyncio.run(
        server.call_tool(
            "query",
            {"query": "alpha", "filter": {"key": "status", "operator": "eq", "value": "published"}},
        )
    )

    assert result.is_error is not True
    files = [r["file"] for r in result.structured_content["results"]]
    assert files == ["notes/a.md"]


def test_query_tool_via_real_dispatch_with_invalid_filter_returns_clean_error():
    server = _build_seeded_server()

    result = asyncio.run(
        server.call_tool(
            "query", {"query": "alpha", "filter": {"key": "x", "operator": "bogus", "value": 1}}
        )
    )

    assert result.is_error is True
    assert "unknown operator" in result.content[0].text


def test_resource_via_real_dispatch_returns_document_content():
    server = _build_seeded_server()

    contents = list(asyncio.run(server.read_resource("qmd://notes/a.md")))

    assert len(contents) == 1
    assert "line one" in contents[0].content


def test_resource_missing_document_raises_resource_not_found_error():
    """Regression test for Finding 1: a missing document must surface as a
    clean ResourceNotFoundError (-32602, INFO-logged, message preserved),
    never as an UnexpectedResourceError (-32603, ERROR-logged traceback,
    message discarded). This is exactly the case that a plain `ValueError`
    from `_read_document_impl` used to get wrong."""
    server = _build_seeded_server()

    with pytest.raises(ResourceNotFoundError) as exc_info:
        asyncio.run(server.read_resource("qmd://notes/missing.md"))

    assert "Document not found" in str(exc_info.value)
    assert "notes/missing.md" in str(exc_info.value)
    # ResourceNotFoundError is a ResourceError, an anticipated failure -- NOT
    # the SDK's own UnexpectedResourceError crash wrapper, which would have
    # discarded this message and named only the URI.
    assert not isinstance(exc_info.value, ValueError)
