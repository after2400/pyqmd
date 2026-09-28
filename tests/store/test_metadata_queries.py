from pyqmd_mlx.store import Store


def _seed(store, path, body, metadata_yaml=None):
    store_collection = "notes"
    if not store.get_collection(store_collection):
        store.add_collection(store_collection, "/notes")
    full_body = f"---\nqmd:\n  metadata:\n{metadata_yaml}\n---\n{body}" if metadata_yaml else body
    content_hash = store.hash_content(full_body)
    store.insert_content(content_hash, full_body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        store_collection,
        path,
        "Title",
        content_hash,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    if metadata_yaml:
        store.sync_document_metadata(doc_id, full_body, path)
    return doc_id


def test_get_metadata_by_filepath_returns_metadata_for_matching_paths():
    store = Store(":memory:")
    _seed(store, "a.md", "body", "    status: published")
    _seed(store, "b.md", "body", "    status: draft")

    result = store.get_metadata_by_filepath(["qmd://notes/a.md", "qmd://notes/b.md"])

    assert result["qmd://notes/a.md"] == {"status": "published"}
    assert result["qmd://notes/b.md"] == {"status": "draft"}
    store.close()


def test_get_metadata_by_filepath_returns_empty_dict_for_no_input():
    store = Store(":memory:")
    assert store.get_metadata_by_filepath([]) == {}
    store.close()


def test_get_metadata_by_filepath_omits_documents_without_extraction():
    store = Store(":memory:")
    _seed(store, "a.md", "body")  # no metadata_yaml -- never synced

    result = store.get_metadata_by_filepath(["qmd://notes/a.md"])

    assert "qmd://notes/a.md" not in result
    store.close()


def test_get_metadata_by_filepath_handles_more_than_one_sql_in_chunk(monkeypatch):
    import pyqmd_mlx.store.store as store_module

    monkeypatch.setattr(store_module, "SQL_IN_CHUNK_SIZE", 2)
    store = Store(":memory:")
    for i in range(5):
        _seed(store, f"doc{i}.md", "body", f"    status: s{i}")

    result = store.get_metadata_by_filepath([f"qmd://notes/doc{i}.md" for i in range(5)])

    assert len(result) == 5
    assert result["qmd://notes/doc3.md"] == {"status": "s3"}
    store.close()


def test_count_documents_pending_metadata_counts_unextracted_active_documents():
    store = Store(":memory:")
    _seed(store, "a.md", "body")  # never extracted
    _seed(store, "b.md", "body", "    status: ok")  # extracted, no error

    assert store.count_documents_pending_metadata() == 1
    store.close()


def test_count_documents_pending_metadata_counts_extraction_errors_as_pending():
    store = Store(":memory:")
    doc_id = _seed(store, "a.md", "body")
    store.sync_document_metadata(doc_id, "---\nqmd:\n  metadata: not-a-mapping\n---\nbody", "a.md")

    assert store.count_documents_pending_metadata() == 1
    store.close()


def test_count_documents_pending_metadata_is_zero_when_all_current():
    store = Store(":memory:")
    _seed(store, "a.md", "body", "    status: ok")

    assert store.count_documents_pending_metadata() == 0
    store.close()


def test_count_documents_pending_metadata_counts_stale_extraction_version(monkeypatch):
    store = Store(":memory:")
    _seed(store, "a.md", "body", "    status: ok")  # extracted under the current version

    import pyqmd_mlx.store._metadata as metadata_module

    monkeypatch.setattr(metadata_module, "METADATA_EXTRACTION_VERSION", 2)
    # count_documents_pending_metadata does `from ._metadata import
    # METADATA_EXTRACTION_VERSION` inside its own method body, so it picks
    # up the bumped value at call time without re-running extraction --
    # simulating "this row was extracted under the old version number."
    assert store.count_documents_pending_metadata() == 1
    store.close()
