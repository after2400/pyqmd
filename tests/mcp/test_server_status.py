from pyqmd_mlx.mcp.server import _status_impl
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_status_impl_reports_document_count():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    result = _status_impl(store)

    assert result.is_error is not True
    assert "1" in result.content[0].text
    assert result.structured_content["counts"]["active_documents"] == 1


def test_status_impl_lists_collections_in_structured_content():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")

    result = _status_impl(store)

    names = [c["name"] for c in result.structured_content["collections"]]
    assert names == ["notes"]


def test_status_impl_on_empty_store_does_not_crash():
    store = Store(":memory:")
    result = _status_impl(store)
    assert result.is_error is not True
    assert result.structured_content["counts"]["active_documents"] == 0
