from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.documents import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _seed(store):
    store.add_collection("notes", "/notes")
    body = "line one\nline two\nline three\nline four\nline five"
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "a.md", "A", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return content_hash


def test_get_by_collection_slash_path(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md"])

    assert result.exit_code == 0
    assert "1: line one" in result.output


def test_get_by_bare_filename_with_no_collection_prefix(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "a.md"])

    assert result.exit_code == 0
    assert "1: line one" in result.output


def test_get_by_docid(monkeypatch):
    store = Store(":memory:")
    content_hash = _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", f"#{content_hash[:6]}"])

    assert result.exit_code == 0
    assert "line one" in result.output


def test_get_missing_document_exits_nonzero(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/missing.md"])

    assert result.exit_code == 1


def test_get_with_line_range_suffix(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md:2:2"])

    assert "2: line two" in result.output
    assert "3: line three" in result.output
    assert "line one" not in result.output
    assert "line four" not in result.output


def test_get_with_explicit_from_and_l_flags_override_suffix(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md:2:2", "--from", "1", "-l", "1"])

    assert "1: line one" in result.output
    assert "line two" not in result.output


def test_get_no_line_numbers_flag(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md", "--no-line-numbers"])

    assert "1: line one" not in result.output
    assert "line one" in result.output


def test_get_shows_folder_context_when_configured(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    store.add_context("notes", "", "Root context")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md"])

    assert result.exit_code == 0
    assert "Folder Context: Root context" in result.output


def test_get_omits_folder_context_line_when_none_configured(monkeypatch):
    store = Store(":memory:")
    _seed(store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md"])

    assert result.exit_code == 0
    assert "Folder Context:" not in result.output


def test_get_full_path_shows_on_disk_path_without_docid(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("line one\nline two")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    h = store.hash_content("line one\nline two")
    store.insert_content(h, "line one\nline two", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md", "--full-path"])

    assert result.exit_code == 0
    assert str((tmp_path / "a.md").resolve()) in result.output
    assert "qmd://" not in result.output.splitlines()[0]


def test_get_full_path_falls_back_and_warns_when_file_missing(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/nonexistent-dir")
    h = store.hash_content("body")
    store.insert_content(h, "body", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["get", "notes/a.md", "--full-path"])

    assert result.exit_code == 0
    assert "qmd://notes/a.md" in result.output
    assert "could not be resolved" in result.output
