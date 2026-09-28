import json

from pyqmd_mlx.store import Store
from pyqmd_mlx.store._metadata import MetadataExtractionResult


def _seed_document(store, path="a.md", body="body"):
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", path, "Title", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return doc_id


def test_sync_document_metadata_extracts_and_persists_scalar_values():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    content = "---\nqmd:\n  metadata:\n    status: published\n    priority: 3\n---\nbody"

    extraction = store.sync_document_metadata(doc_id, content, "a.md")

    assert extraction.metadata == {"status": "published", "priority": 3}
    row = store.conn.execute(
        "SELECT metadata_json, extraction_version, extraction_error FROM document_metadata WHERE document_id = ?",
        (doc_id,),
    ).fetchone()
    assert json.loads(row["metadata_json"]) == {"status": "published", "priority": 3}
    assert row["extraction_version"] == 1
    assert row["extraction_error"] is None

    values = store.conn.execute(
        "SELECT key, ordinal, value_type, text_value, number_value FROM document_metadata_values "
        "WHERE document_id = ? ORDER BY key",
        (doc_id,),
    ).fetchall()
    assert len(values) == 2
    priority_row = next(v for v in values if v["key"] == "priority")
    assert priority_row["value_type"] == "number"
    assert priority_row["number_value"] == 3
    store.close()


def test_sync_document_metadata_persists_array_values_with_ordinals():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    content = "---\nqmd:\n  metadata:\n    topics:\n      - a\n      - b\n---\nbody"

    store.sync_document_metadata(doc_id, content, "a.md")

    rows = store.conn.execute(
        "SELECT ordinal, text_value FROM document_metadata_values WHERE document_id = ? ORDER BY ordinal",
        (doc_id,),
    ).fetchall()
    assert [(r["ordinal"], r["text_value"]) for r in rows] == [(0, "a"), (1, "b")]
    store.close()


def test_sync_document_metadata_persists_extraction_error():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    content = "---\nqmd:\n  metadata: not-a-mapping\n---\nbody"

    extraction = store.sync_document_metadata(doc_id, content, "a.md")

    assert extraction.error is not None
    row = store.conn.execute(
        "SELECT extraction_error FROM document_metadata WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert row["extraction_error"] == extraction.error
    store.close()


def test_sync_document_metadata_replaces_prior_values_on_second_call():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    store.sync_document_metadata(
        doc_id, "---\nqmd:\n  metadata:\n    status: draft\n---\nbody", "a.md"
    )
    store.sync_document_metadata(
        doc_id, "---\nqmd:\n  metadata:\n    status: published\n---\nbody", "a.md"
    )

    rows = store.conn.execute(
        "SELECT text_value FROM document_metadata_values WHERE document_id = ?", (doc_id,)
    ).fetchall()
    assert [r["text_value"] for r in rows] == ["published"]
    store.close()


def test_sync_document_metadata_only_if_stale_skips_current_extraction():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    store.sync_document_metadata(
        doc_id, "---\nqmd:\n  metadata:\n    status: draft\n---\nbody", "a.md"
    )

    result = store.sync_document_metadata(
        doc_id,
        "---\nqmd:\n  metadata:\n    status: published\n---\nbody",
        "a.md",
        only_if_stale=True,
    )

    assert result is None  # skipped -- extraction row is already current
    row = store.conn.execute(
        "SELECT text_value FROM document_metadata_values WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert row["text_value"] == "draft"  # unchanged
    store.close()


def test_sync_document_metadata_only_if_stale_backfills_missing_extraction():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    # No document_metadata row exists yet -- only_if_stale must still extract.
    result = store.sync_document_metadata(
        doc_id,
        "---\nqmd:\n  metadata:\n    status: published\n---\nbody",
        "a.md",
        only_if_stale=True,
    )
    assert result is not None
    assert result.metadata == {"status": "published"}
    store.close()


def test_replace_document_metadata_can_be_called_directly():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    extraction = MetadataExtractionResult(metadata={"status": "published"}, extraction_version=1)

    store.replace_document_metadata(doc_id, extraction)

    row = store.conn.execute(
        "SELECT metadata_json FROM document_metadata WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert json.loads(row["metadata_json"]) == {"status": "published"}
    store.close()


def test_remove_collection_cleans_up_metadata_even_with_foreign_keys_off():
    store = Store(":memory:")
    # Simulate the real-world case: foreign key enforcement off for this
    # connection, as it always is for a freshly reopened on-disk database.
    store.conn.execute("PRAGMA foreign_keys = OFF")
    doc_id = _seed_document(store)
    store.sync_document_metadata(
        doc_id, "---\nqmd:\n  metadata:\n    status: published\n---\nbody", "a.md"
    )

    store.remove_collection("notes")

    assert (
        store.conn.execute(
            "SELECT COUNT(*) as n FROM document_metadata WHERE document_id = ?", (doc_id,)
        ).fetchone()["n"]
        == 0
    )
    assert (
        store.conn.execute(
            "SELECT COUNT(*) as n FROM document_metadata_values WHERE document_id = ?", (doc_id,)
        ).fetchone()["n"]
        == 0
    )
    store.close()
