import json

from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.documents import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _seed_doc(store, path, body):
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document("notes", path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")


def test_multi_get_comma_separated_paths(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "content a")
    _seed_doc(store, "b.md", "content b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "notes/a.md,notes/b.md", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 2


def test_multi_get_resolves_bare_filenames_with_no_collection_prefix(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "content a")
    _seed_doc(store, "b.md", "content b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "a.md,b.md", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 2


def test_multi_get_reports_missing_but_continues(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "content a")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "notes/a.md,notes/missing.md", "--format", "json"])

    assert result.exit_code == 0
    assert "Not found: notes/missing.md" in result.output


def test_multi_get_all_missing_exits_nonzero(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "notes/missing.md"])

    assert result.exit_code == 1


def test_multi_get_respects_max_bytes(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "big.md", "x" * 100)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app, ["multi-get", "notes/big.md", "--max-bytes", "10", "--format", "json"]
    )

    parsed = json.loads(result.output)
    assert parsed[0]["skipped"] is True


def test_multi_get_truncates_at_max_lines(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "\n".join(f"line {i}" for i in range(10)))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "notes/a.md", "-l", "3", "--format", "json"])

    parsed = json.loads(result.output)
    assert "truncated" in parsed[0]["body"]


def test_multi_get_default_format_is_the_banner_renderer(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "content a")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["multi-get", "notes/a.md"])

    assert "File: qmd://notes/a.md" in result.output
    assert "=" * 60 in result.output
    assert "## notes/a.md" not in result.output


def test_multi_get_full_path_shows_on_disk_paths_without_docid(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("content a")
    (tmp_path / "b.md").write_text("content b")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    _seed_doc(store, "a.md", "content a")
    _seed_doc(store, "b.md", "content b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app, ["multi-get", "notes/a.md,notes/b.md", "--full-path", "--format", "json"]
    )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert {e["file"] for e in parsed} == {
        str((tmp_path / "a.md").resolve()),
        str((tmp_path / "b.md").resolve()),
    }
    assert all("docid" not in e for e in parsed)
