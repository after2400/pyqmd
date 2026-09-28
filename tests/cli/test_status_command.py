from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.status import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] for _ in texts]


def test_status_shows_document_and_vector_counts(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.index_content(h, "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "1 files indexed" in result.output
    assert "1 embedded" in result.output


def test_status_shows_pending_embed_count(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "1 need embedding" in result.output


def test_status_lists_collections_with_contexts(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "My personal notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "notes" in result.output
    assert "Contexts: 1" in result.output
    assert "My personal notes" in result.output


def test_status_omits_contexts_section_when_none_configured(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Contexts:" not in result.output


def test_status_truncates_long_context_preview(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    long_text = "x" * 100
    store.add_context("notes", "", long_text)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert ("x" * 57 + "...") in result.output
    assert long_text not in result.output


def test_status_no_collections(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "(none)" in result.output


def test_status_shows_pattern_and_files_lines_per_collection(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", pattern="**/*.md")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Pattern:  **/*.md" in result.output
    assert "Files:    1 (updated" in result.output
    assert "ago)" in result.output


def test_status_shows_files_zero_without_updated_when_empty(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Files:    0" in result.output
    assert "(updated" not in result.output


def test_status_shows_orphaned_hint_when_nonzero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.deactivate_document("notes", "a.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Orphaned: 1 content hash(es) — run 'pyqmd cleanup'" in result.output


def test_status_omits_orphaned_hint_when_zero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Orphaned:" not in result.output


def test_status_updated_line_uses_relative_time_not_raw_iso(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "ago" in result.output
    assert "2026-01-01T00:00:00Z" not in result.output


def test_status_shows_examples_section_when_collections_exist(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Examples" in result.output
    assert "pyqmd ls notes" in result.output
    assert "pyqmd get qmd://notes/path/to/file.md" in result.output
    assert 'pyqmd search "query" -c notes' in result.output


def test_status_omits_examples_section_when_no_collections(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Examples" not in result.output


def test_status_shows_models_section(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Models" in result.output
    assert "Embedding:   https://huggingface.co/" in result.output
    assert "Reranking:   https://huggingface.co/" in result.output
    assert "Generation:" in result.output


def test_status_tips_suggests_context_for_collections_missing_it(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Tips" in result.output
    assert "Add context to collections for better search results: notes" in result.output


def test_status_tips_omits_context_suggestion_when_all_have_context(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "My notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Add context to collections" not in result.output


def test_status_tips_suggests_update_command_only_with_multiple_collections(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "My notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Add update commands" not in result.output

    store.add_collection("other", "/other")
    store.add_context("other", "", "Other notes")

    result = runner.invoke(app, ["status"])

    assert "Add update commands to keep collections fresh: notes, other" in result.output


def test_status_tips_section_omitted_when_no_tips_apply(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", update_command="git pull")
    store.add_context("notes", "", "My notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert "Tips" not in result.output


def test_status_shows_ast_chunking_section(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "AST Chunking" in result.output
    assert "active" in result.output
    assert "python" in result.output
    assert "typescript" in result.output


def test_status_shows_unavailable_languages(monkeypatch):
    from pyqmd_mlx.store import _ast

    def _broken_loader():
        raise RuntimeError("no grammar for you")

    monkeypatch.setitem(_ast.GRAMMAR_LOADERS, "rust", _broken_loader)
    _ast._GRAMMAR_CACHE.pop("rust", None)
    _ast._FAILED_LANGUAGES.discard("rust")
    _ast._GRAMMAR_LOAD_ERRORS.pop("rust", None)

    try:
        store = Store(":memory:", embed_fn=_fake_embed)
        monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Unavailable: rust" in result.output
        assert "no grammar for you" in result.output
    finally:
        # _GRAMMAR_CACHE/_FAILED_LANGUAGES/_GRAMMAR_LOAD_ERRORS are plain
        # module-level state, not monkeypatch-tracked -- must be restored
        # explicitly (in finally, so a failed assertion above still
        # cleans up) or a broken "rust" entry leaks into every test that
        # runs afterward in the same process.
        _ast._FAILED_LANGUAGES.discard("rust")
        _ast._GRAMMAR_LOAD_ERRORS.pop("rust", None)
