from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return content_hash, doc_id


def _seed_metadata(store, doc_id):
    store.conn.execute(
        "INSERT INTO document_metadata "
        "(document_id, metadata_json, extraction_version, extracted_at) "
        "VALUES (?, '{}', 1, '2026-01-01T00:00:00Z')",
        (doc_id,),
    )
    store.conn.execute(
        "INSERT INTO document_metadata_values "
        "(document_id, key, ordinal, value_type, text_value) "
        "VALUES (?, 'tag', 0, 'string', 'example')",
        (doc_id,),
    )
    store.conn.commit()


def test_clear_llm_cache_deletes_all_rows_and_returns_count():
    store = Store(":memory:")
    store.conn.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)",
        ("h1", "result-1", "2026-01-01T00:00:00Z"),
    )
    store.conn.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)",
        ("h2", "result-2", "2026-01-01T00:00:00Z"),
    )
    store.conn.commit()

    assert store.count_llm_cache() == 2
    assert store.clear_llm_cache() == 2
    assert store.count_llm_cache() == 0
    store.close()


def test_clear_llm_cache_returns_zero_when_already_empty():
    store = Store(":memory:")
    assert store.count_llm_cache() == 0
    assert store.clear_llm_cache() == 0
    store.close()


def test_purge_inactive_documents_deletes_only_inactive_rows():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _, active_id = _seed_doc(store, "notes", "active.md", "Active", "keep me")
    _, inactive_id = _seed_doc(store, "notes", "inactive.md", "Inactive", "remove me")
    store.deactivate_document("notes", "inactive.md")

    assert store.count_inactive_documents() == 1
    purged = store.purge_inactive_documents()

    assert purged == 1
    assert store.count_inactive_documents() == 0
    assert store.find_active_document("notes", "active.md") is not None
    row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents WHERE id = ?", (inactive_id,)
    ).fetchone()
    assert row["n"] == 0
    store.close()


def test_purge_inactive_documents_removes_fts_and_metadata_rows():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _, doc_id = _seed_doc(store, "notes", "a.md", "A", "unique searchable body")
    _seed_metadata(store, doc_id)
    store.deactivate_document("notes", "a.md")

    store.purge_inactive_documents()

    fts_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents_fts WHERE documents_fts MATCH 'searchable'"
    ).fetchone()
    assert fts_row["n"] == 0
    metadata_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM document_metadata WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert metadata_row["n"] == 0
    values_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM document_metadata_values WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert values_row["n"] == 0
    store.close()


def test_purge_inactive_documents_returns_zero_when_nothing_inactive():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello")

    assert store.purge_inactive_documents() == 0
    store.close()


def test_count_orphaned_content_matches_cleanup_without_deleting():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    store.deactivate_document("notes", "a.md")

    assert store.count_orphaned_content() == 1
    assert store.count_orphaned_content() == 1  # read-only: calling it again changes nothing
    cleaned = store.cleanup_orphaned_content()
    assert cleaned == 1
    assert store.count_orphaned_content() == 0
    store.close()


def test_count_orphaned_content_returns_zero_when_nothing_orphaned():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    assert store.count_orphaned_content() == 0
    store.close()


def test_vacuum_runs_without_error_on_populated_store():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    store.vacuum()  # must not raise
    store.close()


def test_vacuum_runs_without_error_on_empty_store():
    store = Store(":memory:")
    store.vacuum()  # must not raise
    store.close()


def test_optimize_documents_fts_runs_without_error_on_populated_store():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    store.optimize_documents_fts()  # must not raise
    store.close()


def test_optimize_documents_fts_skips_quietly_when_table_missing():
    store = Store(":memory:")
    store.conn.execute("DROP TABLE documents_fts")
    store.conn.commit()

    store.optimize_documents_fts()  # must not raise
    store.close()
