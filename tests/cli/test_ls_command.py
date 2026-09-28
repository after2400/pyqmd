from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.documents import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _seed_doc(store, collection, path):
    h = store.hash_content(path)
    store.insert_content(h, path, "2026-01-01T00:00:00Z")
    store.insert_document(collection, path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")


def test_ls_no_arg_lists_collections(monkeypatch):
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls"])

    assert "a" in result.output
    assert "b" in result.output


def test_ls_collection_lists_its_documents(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md")
    _seed_doc(store, "notes", "b.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "notes"])

    assert "a.md" in result.output
    assert "b.md" in result.output


def test_ls_collection_slash_prefix_filters_by_path(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "journals/2025/a.md")
    _seed_doc(store, "notes", "journals/2024/b.md")
    _seed_doc(store, "notes", "other.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "notes/journals/2025"])

    assert "journals/2025/a.md" in result.output
    assert "journals/2024/b.md" not in result.output
    assert "other.md" not in result.output


def test_ls_missing_collection_exits_nonzero(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "nope"])

    assert result.exit_code == 1


def test_ls_empty_collection_prints_message(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "notes"])

    assert "No documents" in result.output


def test_ls_no_arg_shows_qmd_uri_and_file_count(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md")
    _seed_doc(store, "notes", "b.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls"])

    assert "qmd://notes/" in result.output
    assert "(2 files)" in result.output


def test_ls_no_arg_shows_zero_files_for_empty_collection(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls"])

    assert "(0 files)" in result.output


def test_ls_collection_shows_size_date_and_qmd_uri(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T12:00:00Z")
    # Noon UTC: format_ls_time converts to the system's local timezone
    # (matching Node's Date.getHours()), so a midnight timestamp would
    # land on a different calendar day depending on the machine running
    # this test. Noon stays "Jan 01" for any offset within +/-12h.
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "notes"])

    assert "qmd://notes/a.md" in result.output
    assert "11 B" in result.output
    assert "Jan 01" in result.output


def test_ls_collection_right_aligns_sizes(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    h1 = store.hash_content("x")
    store.insert_content(h1, "x", "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "small.md", "S", h1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    h2 = store.hash_content("x" * 2000)
    store.insert_content(h2, "x" * 2000, "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "big.md", "B", h2, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    monkeypatch.setattr("pyqmd_mlx.cli.commands.documents.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["ls", "notes"])

    lines = [line for line in result.output.splitlines() if line.strip()]
    # Both rows share the same modified_at, so format_ls_time produces an
    # identical, fixed-width date string for each -- meaning the qmd://
    # column lines up at the same index only if the (right-justified)
    # size column is also the same width on every row.
    qmd_uri_positions = {line.index("qmd://") for line in lines}
    assert len(qmd_uri_positions) == 1
