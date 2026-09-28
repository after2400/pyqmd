from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body, created_at, modified_at):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, created_at)
    store.insert_document(collection, path, title, content_hash, created_at, modified_at)


def test_get_collection_document_stats_counts_and_latest_per_collection():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "X", "x", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    _seed_doc(store, "a", "y.md", "Y", "y", "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z")
    _seed_doc(store, "b", "z.md", "Z", "z", "2026-01-01T00:00:00Z", "2026-01-15T00:00:00Z")

    stats = store.get_collection_document_stats()

    assert stats["a"] == {"count": 2, "latest_modified": "2026-02-01T00:00:00Z"}
    assert stats["b"] == {"count": 1, "latest_modified": "2026-01-15T00:00:00Z"}
    store.close()


def test_get_collection_document_stats_excludes_inactive_documents():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    _seed_doc(store, "a", "x.md", "X", "x", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.deactivate_document("a", "x.md")

    stats = store.get_collection_document_stats()

    assert "a" not in stats
    store.close()


def test_get_collection_document_stats_omits_collections_with_no_documents():
    store = Store(":memory:")
    store.add_collection("empty", "/empty")

    stats = store.get_collection_document_stats()

    assert "empty" not in stats
    store.close()


def test_get_collection_document_stats_returns_empty_dict_when_no_collections():
    store = Store(":memory:")
    assert store.get_collection_document_stats() == {}
    store.close()
