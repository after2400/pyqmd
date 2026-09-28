from pyqmd_mlx.store import Store


def test_hash_content_is_stable_sha256():
    store = Store(":memory:")
    h1 = store.hash_content("hello world")
    h2 = store.hash_content("hello world")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest length
    store.close()


def test_insert_content_and_document_round_trip():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("# Hello\nbody text")
    store.insert_content(content_hash, "# Hello\nbody text", "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", "hello.md", "Hello", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    assert isinstance(doc_id, int)

    found = store.find_active_document("notes", "hello.md")
    assert found["hash"] == content_hash
    assert found["title"] == "Hello"
    store.close()


def test_find_active_document_returns_none_when_missing():
    store = Store(":memory:")
    assert store.find_active_document("notes", "missing.md") is None
    store.close()


def test_update_document_changes_hash_and_title():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    old_hash = store.hash_content("v1")
    store.insert_content(old_hash, "v1", "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", "a.md", "V1", old_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )

    new_hash = store.hash_content("v2")
    store.insert_content(new_hash, "v2", "2026-01-02T00:00:00Z")
    store.update_document(doc_id, "V2", new_hash, "2026-01-02T00:00:00Z")

    found = store.find_active_document("notes", "a.md")
    assert found["hash"] == new_hash
    assert found["title"] == "V2"
    store.close()


def test_deactivate_document_hides_it_from_find_active():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("gone")
    store.insert_content(content_hash, "gone", "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "gone.md", "Gone", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.deactivate_document("notes", "gone.md")
    assert store.find_active_document("notes", "gone.md") is None
    store.close()


def test_get_active_document_paths():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    for path in ("a.md", "b.md"):
        h = store.hash_content(path)
        store.insert_content(h, path, "2026-01-01T00:00:00Z")
        store.insert_document(
            "notes", path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
    paths = set(store.get_active_document_paths("notes"))
    assert paths == {"a.md", "b.md"}
    store.close()


def test_get_active_documents_with_size_returns_path_modified_and_byte_size():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello 🎉 world")
    store.insert_content(h, "hello 🎉 world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z")

    docs = store.get_active_documents_with_size("notes")

    assert len(docs) == 1
    assert docs[0]["path"] == "a.md"
    assert docs[0]["modified_at"] == "2026-02-01T00:00:00Z"
    assert docs[0]["size"] == len("hello 🎉 world".encode())
    store.close()


def test_get_active_documents_with_size_excludes_inactive_and_orders_by_path():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    for path in ("b.md", "a.md", "c.md"):
        h = store.hash_content(path)
        store.insert_content(h, path, "2026-01-01T00:00:00Z")
        store.insert_document(
            "notes", path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
    store.deactivate_document("notes", "c.md")

    docs = store.get_active_documents_with_size("notes")

    assert [d["path"] for d in docs] == ["a.md", "b.md"]
    store.close()


def test_get_active_documents_with_size_returns_empty_for_empty_collection():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    assert store.get_active_documents_with_size("notes") == []
    store.close()


def test_insert_document_same_collection_and_path_conflicts():
    import sqlite3

    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    h = store.hash_content("x")
    store.insert_content(h, "x", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "x.md", "X", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    try:
        store.insert_document(
            "notes", "x.md", "X2", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        raise AssertionError("expected sqlite3.IntegrityError")
    except sqlite3.IntegrityError:
        pass
    store.close()
