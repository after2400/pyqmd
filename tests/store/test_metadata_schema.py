from pyqmd_mlx.store import Store


def _table_sql(store, name):
    row = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row["sql"] if row else None


def test_document_metadata_table_exists():
    store = Store(":memory:")
    sql = _table_sql(store, "document_metadata")
    assert sql is not None
    assert "document_id INTEGER PRIMARY KEY" in sql
    assert "extraction_version INTEGER NOT NULL" in sql
    store.close()


def test_document_metadata_values_table_exists():
    store = Store(":memory:")
    sql = _table_sql(store, "document_metadata_values")
    assert sql is not None
    assert "value_type TEXT NOT NULL" in sql
    store.close()


def test_document_metadata_values_rejects_wrong_type_combination():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("body")
    store.insert_content(content_hash, "body", "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", "a.md", "A", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.conn.execute(
        "INSERT INTO document_metadata (document_id, extraction_version, extracted_at) VALUES (?, 1, '2026-01-01T00:00:00Z')",
        (doc_id,),
    )
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, text_value, number_value) "
            "VALUES (?, 'k', 0, 'string', 'v', 1.0)",
            (doc_id,),
        )
    store.close()


def test_document_metadata_values_cascade_deletes_with_document_metadata():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("body")
    store.insert_content(content_hash, "body", "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", "a.md", "A", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.conn.execute("PRAGMA foreign_keys = ON")
    store.conn.execute(
        "INSERT INTO document_metadata (document_id, extraction_version, extracted_at) VALUES (?, 1, '2026-01-01T00:00:00Z')",
        (doc_id,),
    )
    store.conn.execute(
        "INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, text_value) "
        "VALUES (?, 'k', 0, 'string', 'v')",
        (doc_id,),
    )
    store.conn.commit()
    store.conn.execute("DELETE FROM document_metadata WHERE document_id = ?", (doc_id,))
    store.conn.commit()
    row = store.conn.execute(
        "SELECT COUNT(*) as n FROM document_metadata_values WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert row["n"] == 0
    store.close()


def test_metadata_indexes_exist():
    store = Store(":memory:")
    names = {
        row["name"]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert {
        "idx_metadata_text_lookup",
        "idx_metadata_number_lookup",
        "idx_metadata_boolean_lookup",
    } <= names
    store.close()


def test_schema_version_unchanged_at_one():
    store = Store(":memory:")
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1
    store.close()
