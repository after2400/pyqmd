from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return content_hash


def test_cleanup_orphaned_content_deletes_hash_with_no_active_document():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    content_hash = _seed_doc(store, "notes", "a.md", "A", "hello world")
    store.index_content(content_hash, "hello world", model="fake-model")
    store.deactivate_document("notes", "a.md")

    cleaned = store.cleanup_orphaned_content()

    assert cleaned == 1
    content_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert content_row["n"] == 0
    vectors_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content_vectors WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert vectors_row["n"] == 0
    store.close()


def test_cleanup_orphaned_content_preserves_hash_shared_with_active_document():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    shared_hash = _seed_doc(store, "a", "x.md", "X", "shared content")
    _seed_doc(store, "b", "y.md", "Y", "shared content")
    store.deactivate_document("a", "x.md")

    cleaned = store.cleanup_orphaned_content()

    assert cleaned == 0
    content_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content WHERE hash = ?", (shared_hash,)
    ).fetchone()
    assert content_row["n"] == 1
    store.close()


def test_cleanup_orphaned_content_returns_zero_when_nothing_orphaned():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    assert store.cleanup_orphaned_content() == 0
    store.close()
