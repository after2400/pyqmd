from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.context import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def test_context_add_via_qmd_uri_is_stored(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", "qmd://notes/People", "Notes", "about", "people"])

    assert result.exit_code == 0
    assert "✓ Added context for: qmd://notes/People" in result.output
    assert "Context: Notes about people" in result.output
    assert store.get_context_for_path("notes", "People/wife.md") == "Notes about people"


def test_context_add_root_message_variant(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", "qmd://notes/", "Root", "context"])

    assert result.exit_code == 0
    assert "✓ Added context for: qmd://notes/ (collection root)" in result.output


def test_context_add_unknown_qmd_collection_errors_without_partial_write(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", "qmd://nope/People", "text"])

    assert result.exit_code == 1
    assert store.list_all_contexts() == []


def test_context_add_via_filesystem_path_resolves_cwd(tmp_path, monkeypatch):
    (tmp_path / "People").mkdir()
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)
    monkeypatch.chdir(tmp_path / "People")

    result = runner.invoke(app, ["add", ".", "People", "context"])

    assert result.exit_code == 0
    assert store.get_context_for_path("notes", "People/wife.md") == "People context"


def test_context_add_filesystem_path_outside_any_collection_errors(tmp_path, monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path / "notes"))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    result = runner.invoke(app, ["add", ".", "text"])

    assert result.exit_code == 1
    assert "is not in any indexed collection" in result.output
    assert "pyqmd status" in result.output


def test_context_list_empty_state(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "No contexts configured. Use 'pyqmd context add' to add one." in result.output


def test_context_list_groups_by_collection(monkeypatch):
    store = Store(":memory:")
    store.add_collection("reference", "/reference")
    store.add_context("reference", "", "Stable reference notes")
    store.add_context("reference", "People", "Notes about people")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "reference" in result.output
    assert "/ (root)" in result.output
    assert "Stable reference notes" in result.output
    assert "  People" in result.output
    assert "Notes about people" in result.output


def test_context_remove_success(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "People", "context")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "qmd://notes/People"])

    assert result.exit_code == 0
    assert "✓ Removed context for: qmd://notes/People" in result.output
    assert store.get_context_for_path("notes", "People/x.md") is None


def test_context_remove_no_context_set_errors(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "qmd://notes/People"])

    assert result.exit_code == 1
    assert "No context found for" in result.output


def test_context_remove_unknown_collection_errors(monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "qmd://nope/People"])

    assert result.exit_code == 1
    assert "Collection not found: nope" in result.output


def test_context_list_prints_node_header_and_layout(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "root context")
    store.add_context("notes", "sub.md", "file context")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert result.stdout == (
        "\nConfigured Contexts\n\nnotes\n  / (root)\n    root context\n  sub.md\n    file context\n"
    )


def test_context_add_root_via_virtual_path(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", "qmd://notes/", "root", "context"])

    assert result.exit_code == 0
    assert result.stdout == (
        "✓ Added context for: qmd://notes/ (collection root)\nContext: root context\n"
    )


def test_context_add_root_via_filesystem_path_has_no_root_suffix(tmp_path, monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "root"])

    assert result.exit_code == 0
    assert result.stdout.startswith("✓ Added context for: qmd://notes/\n")


def test_context_remove_echoes_virtual_path_argument(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "root context")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "qmd://notes/"])

    assert result.exit_code == 0
    assert result.stdout == "✓ Removed context for: qmd://notes/\n"


def test_context_remove_missing_prints_node_error(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "qmd://notes/"])

    assert result.exit_code == 1
    assert result.stderr == "No context found for: qmd://notes/\n"


def test_context_add_path_outside_collections_prints_node_error(tmp_path, monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "x"])

    assert result.exit_code == 1
    assert result.stderr == (
        f"Path is not in any indexed collection: {tmp_path}\n"
        "Run 'pyqmd status' to see indexed collections\n"
    )


def test_context_remove_path_outside_collections_has_no_hint_line(tmp_path, monkeypatch):
    store = Store(":memory:")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", str(tmp_path)])

    assert result.exit_code == 1
    assert result.stderr == f"Path is not in any indexed collection: {tmp_path}\n"


def test_context_colors_match_node(monkeypatch):
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.context.get_store", lambda db_path=None: store)

    added = runner.invoke(app, ["add", "qmd://notes/", "root"], color=True)
    listed = runner.invoke(app, ["list"], color=True)

    assert added.stdout.startswith("\x1b[32m✓\x1b[0m Added context for: qmd://notes/")
    assert "\x1b[2mContext: root\x1b[0m" in added.stdout
    assert "\x1b[1mConfigured Contexts\x1b[0m" in listed.stdout
    assert "\x1b[36mnotes\x1b[0m" in listed.stdout
    assert "    \x1b[2mroot\x1b[0m" in listed.stdout
