from pyqmd_mlx.store import Store
from pyqmd_mlx.store._metadata_filter import parse_metadata_filter


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return content_hash


def test_search_fts_finds_matching_document():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Authentication", "how to configure authentication")
    _seed_doc(store, "notes", "docker.md", "Docker", "docker compose networking guide")

    results = store.search_fts("authentication")
    assert len(results) == 1
    assert results[0].title == "Authentication"
    assert results[0].source == "fts"
    store.close()


def test_search_fts_returns_empty_for_unmatched_query():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "something unrelated")
    assert store.search_fts("nonexistent_term_xyz") == []
    store.close()


def test_search_fts_returns_empty_for_negation_only_query():
    store = Store(":memory:")
    assert store.search_fts("-excluded") == []
    store.close()


def test_search_fts_populates_context_when_configured():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "notes", "auth.md", "Authentication", "how to configure authentication")

    results = store.search_fts("authentication")

    assert results[0].context == "Notes root context"
    store.close()


def test_search_fts_context_is_none_when_not_configured():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Authentication", "how to configure authentication")

    results = store.search_fts("authentication")

    assert results[0].context is None
    store.close()


def test_search_fts_respects_limit():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    for i in range(5):
        _seed_doc(store, "notes", f"doc{i}.md", f"Doc {i}", "shared keyword content")
    results = store.search_fts("shared", limit=2)
    assert len(results) == 2
    store.close()


def test_search_fts_scopes_to_collection():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "X", "matching keyword")
    _seed_doc(store, "b", "y.md", "Y", "matching keyword")

    results = store.search_fts("matching", collection="a")
    assert len(results) == 1
    assert results[0].collection_name == "a"
    store.close()


def test_search_fts_score_is_between_zero_and_one():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "keyword content here")
    results = store.search_fts("keyword")
    assert 0.0 < results[0].score < 1.0
    store.close()


def _seed_with_metadata(store, path, title, body, metadata_yaml):
    full_body = f"---\nqmd:\n  metadata:\n{metadata_yaml}\n---\n{body}"
    content_hash = store.hash_content(full_body)
    store.insert_content(content_hash, full_body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.sync_document_metadata(doc_id, full_body, path)
    return doc_id


def test_search_fts_filter_excludes_non_matching_documents():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_with_metadata(
        store,
        "a.md",
        "Auth Published",
        "authentication configuration guide",
        "    status: published",
    )
    _seed_with_metadata(
        store, "b.md", "Auth Draft", "authentication configuration guide", "    status: draft"
    )

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_fts("authentication", filter=filter_)

    assert [r.title for r in results] == ["Auth Published"]
    store.close()


def test_search_fts_filter_excludes_documents_without_extraction():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    body = "authentication configuration guide"
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "a.md", "Auth", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    # No sync_document_metadata call -- this document was never extracted.

    filter_ = parse_metadata_filter({"key": "status", "operator": "exists", "value": False})
    results = store.search_fts("authentication", filter=filter_)

    assert results == []  # exists:false must NOT match an unextracted document
    store.close()


def test_search_fts_filter_combines_with_collection_scope():
    # Both documents have status:published metadata (the filter alone would
    # match both) -- only the collection scope should exclude "Other Auth",
    # proving `collection` and `filter` compose rather than either one
    # accidentally doing all the work.
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_collection("other", "/other")
    _seed_with_metadata(
        store, "a.md", "Notes Auth", "authentication configuration guide", "    status: published"
    )
    other_body = "authentication configuration guide for another collection"
    full_other_body = f"---\nqmd:\n  metadata:\n    status: published\n---\n{other_body}"
    other_hash = store.hash_content(full_other_body)
    store.insert_content(other_hash, full_other_body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "other", "b.md", "Other Auth", other_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.sync_document_metadata(doc_id, full_other_body, "b.md")

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_fts("authentication", collection="notes", filter=filter_)

    assert [r.title for r in results] == ["Notes Auth"]
    store.close()
