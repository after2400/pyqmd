"""Regression tests for Store methods that build a SQL WHERE clause with one
placeholder per hash/doc-id in a Python list (clear_embeddings,
remove_collection). Those clauses must be batched into chunks -- otherwise a
realistically-sized collection blows past SQLite's default expression-tree
depth limit of 1000 and `sqlite3.OperationalError: Expression tree is too
large (maximum depth 1000)` is raised."""

import sqlite_vec

from pyqmd_mlx.store import Store

# Comfortably over SQLite's default expression-tree depth limit (1000), so
# an unbatched " OR ".join(...)/",".join(...) clause over this many hashes
# or doc ids would raise before these tests were added.
BULK_DOC_COUNT = 1200


def _seed_many_documents(store: Store, collection: str, n: int) -> list[str]:
    """Insert `n` distinct small documents (distinct content -> distinct
    hashes) into `collection`. Returns the list of content hashes."""
    store.add_collection(collection, f"/{collection}")
    hashes = []
    for i in range(n):
        body = f"document body number {i}"
        content_hash = store.hash_content(body)
        store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
        store.insert_document(
            collection,
            f"doc{i}.md",
            f"Doc {i}",
            content_hash,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        )
        hashes.append(content_hash)
    return hashes


def _seed_fake_vectors(store: Store, hashes: list[str]) -> None:
    """Populate content_vectors/vectors_vec for each hash directly via SQL
    (bypassing index_content's embedding call) so bulk deletes have
    something real to chunk through."""
    store.ensure_vec_table(2)
    for h in hashes:
        store.conn.execute(
            """
            INSERT INTO content_vectors (hash, seq, pos, model, total_chunks, embedded_at)
            VALUES (?, 0, 0, 'fake-model', 1, '2026-01-01T00:00:00Z')
            """,
            (h,),
        )
        store.conn.execute(
            "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
            (f"{h}_0", sqlite_vec.serialize_float32([1.0, 0.0])),
        )
    store.conn.commit()


def test_clear_embeddings_handles_large_hash_list_without_raising():
    store = Store(":memory:")
    hashes = _seed_many_documents(store, "notes", BULK_DOC_COUNT)
    _seed_fake_vectors(store, hashes)

    cleared = store.clear_embeddings("notes")

    assert cleared == BULK_DOC_COUNT
    remaining_vectors = store.conn.execute("SELECT COUNT(*) AS n FROM content_vectors").fetchone()[
        "n"
    ]
    assert remaining_vectors == 0
    remaining_vec_rows = store.conn.execute("SELECT COUNT(*) AS n FROM vectors_vec").fetchone()["n"]
    assert remaining_vec_rows == 0
    store.close()


def test_clear_embeddings_unscoped_handles_large_hash_list_without_raising():
    store = Store(":memory:")
    hashes = _seed_many_documents(store, "notes", BULK_DOC_COUNT)
    _seed_fake_vectors(store, hashes)

    cleared = store.clear_embeddings()

    assert cleared == BULK_DOC_COUNT
    remaining_vec_rows = store.conn.execute("SELECT COUNT(*) AS n FROM vectors_vec").fetchone()["n"]
    assert remaining_vec_rows == 0
    store.close()


def test_remove_collection_handles_large_document_list_without_raising():
    store = Store(":memory:")
    hashes = _seed_many_documents(store, "notes", BULK_DOC_COUNT)
    _seed_fake_vectors(store, hashes)

    assert store.remove_collection("notes") is True

    doc_count = store.conn.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE collection = ?", ("notes",)
    ).fetchone()["n"]
    assert doc_count == 0
    fts_count = store.conn.execute("SELECT COUNT(*) AS n FROM documents_fts").fetchone()["n"]
    assert fts_count == 0
    content_count = store.conn.execute("SELECT COUNT(*) AS n FROM content").fetchone()["n"]
    assert content_count == 0
    content_vectors_count = store.conn.execute(
        "SELECT COUNT(*) AS n FROM content_vectors"
    ).fetchone()["n"]
    assert content_vectors_count == 0
    vec_count = store.conn.execute("SELECT COUNT(*) AS n FROM vectors_vec").fetchone()["n"]
    assert vec_count == 0
    store.close()


def test_remove_collection_large_list_preserves_other_collections():
    store = Store(":memory:")
    hashes = _seed_many_documents(store, "notes", BULK_DOC_COUNT)
    _seed_fake_vectors(store, hashes)
    other_hash = store.hash_content("other collection content")
    store.add_collection("keep", "/keep")
    store.insert_content(other_hash, "other collection content", "2026-01-01T00:00:00Z")
    store.insert_document(
        "keep", "a.md", "A", other_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )

    store.remove_collection("notes")

    assert store.get_collection("keep") is not None
    assert store.find_active_document("keep", "a.md") is not None
    content_row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM content WHERE hash = ?", (other_hash,)
    ).fetchone()
    assert content_row["n"] == 1
    store.close()
