from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.cleanup import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def _seed_everything(store):
    """Seed one row in every category cleanup touches: an llm_cache
    entry, and one inactive document (with metadata, and an FTS row
    still present since deactivate_document doesn't remove it) whose
    content hash becomes orphaned once it has no active reference."""
    store.conn.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)",
        ("cache-h1", "cached-result", "2026-01-01T00:00:00Z"),
    )
    store.conn.commit()
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("stale body")
    store.insert_content(content_hash, "stale body", "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        "notes", "a.md", "A", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.conn.execute(
        "INSERT INTO document_metadata "
        "(document_id, metadata_json, extraction_version, extracted_at) "
        "VALUES (?, '{}', 1, '2026-01-01T00:00:00Z')",
        (doc_id,),
    )
    store.conn.commit()
    store.deactivate_document("notes", "a.md")
    return doc_id, content_hash


def test_cleanup_real_run_removes_everything_and_reports_counts(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)
    doc_id, content_hash = _seed_everything(store)

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0
    assert "✓ Cleared 1 cached API responses" in result.output
    assert "✓ Removed 1 inactive document record(s)" in result.output
    assert "✓ Cleaned up 1 orphaned content hash(es)" in result.output
    assert "✓ FTS compacted, database vacuumed" in result.output

    assert store.count_llm_cache() == 0
    doc_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    assert doc_row["n"] == 0
    content_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert content_row["n"] == 0
    metadata_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM document_metadata WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert metadata_row["n"] == 0


def test_cleanup_real_run_on_clean_index_only_prints_vacuum_line(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0
    assert "✓ FTS compacted, database vacuumed" in result.output
    assert "Cleared" not in result.output
    assert "Removed" not in result.output
    assert "Cleaned up" not in result.output


def test_cleanup_dry_run_reports_counts_without_changing_anything(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)
    doc_id, content_hash = _seed_everything(store)

    result = runner.invoke(app, ["cleanup", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run — no changes made." in result.output
    assert "Would clear 1 cached API responses" in result.output
    assert "Would remove 1 inactive document record(s)" in result.output
    assert "Would clean up 1 orphaned content hash(es)" in result.output
    assert "Would compact FTS and vacuum the database" in result.output

    assert store.count_llm_cache() == 1
    doc_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    assert doc_row["n"] == 1
    content_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert content_row["n"] == 1
    metadata_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM document_metadata WHERE document_id = ?", (doc_id,)
    ).fetchone()
    assert metadata_row["n"] == 1


def test_cleanup_dry_run_on_clean_index_only_prints_vacuum_line(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["cleanup", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run — no changes made." in result.output
    assert "Would compact FTS and vacuum the database" in result.output
    assert "Would clear" not in result.output
    assert "Would remove" not in result.output
    assert "Would clean up" not in result.output


def test_cleanup_colors_only_the_checkmark_green_like_node(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["cleanup"], color=True)

    assert result.exit_code == 0
    assert "\x1b[32m✓\x1b[0m FTS compacted, database vacuumed" in result.stdout
