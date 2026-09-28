from pyqmd_mlx.store.store import Store


def test_store_creates_all_core_tables():
    store = Store(":memory:")
    tables = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    for expected in ("content", "documents", "collections", "content_vectors", "llm_cache"):
        assert expected in tables
    store.close()


def test_store_creates_fts5_virtual_table():
    store = Store(":memory:")
    tables = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "documents_fts" in tables
    store.close()


def test_store_sets_user_version_to_current_schema_version():
    from pyqmd_mlx.store._schema import CURRENT_SCHEMA_VERSION

    store = Store(":memory:")
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == CURRENT_SCHEMA_VERSION
    store.close()


def test_store_migrate_is_idempotent_on_reopen():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        store1 = Store(db_path)
        store1.close()
        # Reopening an already-current-version DB should not raise or
        # recreate anything destructively.
        store2 = Store(db_path)
        version = store2.conn.execute("PRAGMA user_version").fetchone()[0]
        from pyqmd_mlx.store._schema import CURRENT_SCHEMA_VERSION

        assert version == CURRENT_SCHEMA_VERSION
        store2.close()


def test_sqlite_vec_extension_is_loaded():
    store = Store(":memory:")
    row = store.conn.execute("SELECT vec_version()").fetchone()
    assert row is not None
    store.close()


def test_store_injected_llm_functions_override_defaults():
    def fake_embed(texts, model, kind="query"):
        return [[0.0] * 4 for _ in texts]

    store = Store(":memory:", embed_fn=fake_embed)
    assert store._embed_fn is fake_embed
    store.close()
