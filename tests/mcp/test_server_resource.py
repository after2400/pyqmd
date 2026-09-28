import pytest
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from pyqmd_mlx.mcp.server import _read_document_impl
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_read_document_impl_returns_line_numbered_body():
    store = Store(":memory:")
    store.add_collection("journals", "/journals")
    _seed_doc(store, "journals", "2025/note.md", "Note", "line one\nline two")

    text = _read_document_impl(store, "journals/2025/note.md")

    assert "1: line one" in text
    assert "2: line two" in text


def test_read_document_impl_raises_resource_not_found_error_when_not_found():
    store = Store(":memory:")
    with pytest.raises(ResourceNotFoundError, match="not found"):
        _read_document_impl(store, "journals/missing.md")
