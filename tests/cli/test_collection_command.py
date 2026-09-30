from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.collection import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def test_add_indexes_files_and_prints_summary(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Creating collection 'notes'...\n"
        f"Collection: {tmp_path} (**/*.md)\n"
        "\n"
        "Indexed: 1 new, 0 updated, 0 unchanged, 0 removed\n"
        "\n"
        "Run 'pyqmd embed' to update embeddings (1 unique hashes need vectors)\n"
        "✓ Collection 'notes' created successfully\n"
    )
    assert store.get_collection("notes") is not None


def test_add_does_not_print_orphan_line_for_a_fresh_collection(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    assert result.exit_code == 0
    assert "Cleaned up" not in result.output


def test_add_duplicate_name_exits_nonzero(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Collection 'notes' already exists.\nUse a different name with --name <name>\n"
    )


def test_add_duplicate_name_error_has_no_traceback(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    assert result.exit_code == 1
    assert "IntegrityError" not in result.output


def test_add_duplicate_path_and_pattern_under_different_name_errors_cleanly(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path), pattern="**/*.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "other"])

    assert result.exit_code == 1
    assert result.stderr == (
        "A collection already exists for this path and pattern:\n"
        "  Name: notes (qmd://notes/)\n"
        "  Pattern: **/*.md\n"
        "\n"
        "Use 'pyqmd update' to re-index it, or remove it first with "
        "'pyqmd collection remove notes'\n"
    )
    assert store.get_collection("other") is None


def test_add_duplicate_path_different_spelling_is_not_deduplicated(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", f"{tmp_path}/", pattern="**/*.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "other"])

    assert result.exit_code == 0
    assert store.get_collection("other") is not None


def test_add_persists_exclude_patterns_to_ignore_patterns_column(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    (tmp_path / "draft.md").write_text("# Draft\nskip me")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app,
        ["add", str(tmp_path), "--name", "notes", "--exclude", "draft.md", "--exclude", "*.tmp"],
    )

    assert result.exit_code == 0
    assert store.get_collection("notes")["ignore_patterns"] == "draft.md,*.tmp"
    assert store.find_active_document("notes", "draft.md") is None


def test_add_without_exclude_leaves_ignore_patterns_null(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    assert store.get_collection("notes")["ignore_patterns"] is None


def test_update_cmd_sets_multiword_command_unquoted(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update-cmd", "notes", "git", "pull"])

    assert result.exit_code == 0
    assert "Set update command for 'notes': git pull" in result.output
    assert store.get_collection("notes")["update_command"] == "git pull"


def test_update_cmd_with_no_command_clears_it(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", update_command="git pull")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update-cmd", "notes"])

    assert result.exit_code == 0
    assert "Cleared update command for 'notes'" in result.output
    assert store.get_collection("notes")["update_command"] is None


def test_update_cmd_unknown_collection_errors_cleanly(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update-cmd", "nope", "git", "pull"])

    assert result.exit_code == 1
    assert result.stderr == "Collection not found: nope\n"


def test_add_update_cmd_option_is_stored(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app, ["add", str(tmp_path), "--name", "notes", "--update-cmd", "git pull"]
    )

    assert result.exit_code == 0
    assert store.get_collection("notes")["update_command"] == "git pull"


def test_show_prints_update_line_when_set(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", update_command="git pull")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "notes"])

    assert "  Update:   git pull\n" in result.stdout


def test_show_omits_update_line_when_unset(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "notes"])

    assert "Update:" not in result.output


def test_list_shows_all_collections(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" in result.output


def test_list_empty_prints_message(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "No collections" in result.output


def test_list_shows_header_with_collection_count(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "Collections (2):" in result.output


def test_list_shows_pattern_files_and_updated_per_collection(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", pattern="**/*.md")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "notes (qmd://notes/)" in result.output
    assert "Pattern:  **/*.md" in result.output
    assert "Files:    1" in result.output
    assert "Updated:" in result.output
    assert "ago" in result.output


def test_list_shows_zero_files_and_updated_now_for_empty_collection(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "Files:    0" in result.output
    assert "Updated:  0s ago" in result.output


def test_list_shows_ignore_line_when_exclude_patterns_set(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", ignore_patterns="drafts/**,*.tmp.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "Ignore:   drafts/**, *.tmp.md" in result.output


def test_list_omits_ignore_line_when_not_set(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "Ignore:" not in result.output


def test_list_shows_excluded_tag_when_include_by_default_false(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", include_by_default=False)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "notes (qmd://notes/) [excluded]" in result.output


def test_list_omits_excluded_tag_by_default(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["list"])

    assert "[excluded]" not in result.output


def test_show_prints_collection_details(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "My notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "notes"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Collection: notes\n"
        "  Path:     /notes\n"
        "  Pattern:  **/*.md\n"
        "  Include:  yes (default)\n"
        "  Contexts: 1\n"
        "  Documents: 0\n"
    )


def test_show_missing_collection_exits_nonzero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "nope"])

    assert result.exit_code == 1
    assert result.stderr == "Collection not found: nope\n"


def test_remove_with_yes_flag_skips_confirmation(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "notes", "--yes"])

    assert result.exit_code == 0
    assert result.stdout == "✓ Removed collection 'notes'\n  Deleted 0 documents\n"
    assert store.get_collection("notes") is None


def test_remove_without_yes_prompts_and_respects_no(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "notes"], input="n\n")

    assert "Cancelled" in result.output
    assert store.get_collection("notes") is not None


def test_rename_updates_name(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("old", "/x")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["rename", "old", "new"])

    assert result.exit_code == 0
    assert store.get_collection("old") is None
    assert store.get_collection("new") is not None


def test_rename_missing_collection_exits_nonzero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["rename", "nope", "new"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Collection not found: nope\nRun 'pyqmd collection list' to see available collections.\n"
    )


def test_include_reports_success(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", include_by_default=False)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["include", "notes"])

    assert result.exit_code == 0
    assert "included in default queries" in result.output
    assert store.get_collection("notes")["include_by_default"] == 1


def test_exclude_reports_success(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["exclude", "notes"])

    assert result.exit_code == 0
    assert "excluded from default queries" in result.output
    assert store.get_collection("notes")["include_by_default"] == 0


def test_include_unknown_collection_errors_cleanly(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["include", "nope"])

    assert result.exit_code == 1


def test_exclude_unknown_collection_errors_cleanly(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["exclude", "nope"])

    assert result.exit_code == 1


def test_exclude_is_idempotent(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", include_by_default=False)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["exclude", "notes"])

    assert result.exit_code == 0
    assert store.get_collection("notes")["include_by_default"] == 0


def test_show_prints_include_yes_by_default(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "notes"])

    assert "  Include:  yes (default)\n" in result.stdout


def test_show_prints_include_no_when_excluded(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes", include_by_default=False)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["show", "notes"])

    assert "  Include:  no\n" in result.stdout


def test_remove_prints_deleted_and_cleaned_counts(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)
    runner.invoke(app, ["add", str(tmp_path), "--name", "notes"])

    result = runner.invoke(app, ["remove", "notes", "--yes"])

    assert result.exit_code == 0
    assert result.stdout == (
        "✓ Removed collection 'notes'\n"
        "  Deleted 1 documents\n"
        "  Cleaned up 1 orphaned content hashes\n"
    )


def test_remove_missing_collection_prints_node_error(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["remove", "ghost", "--yes"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Collection not found: ghost\nRun 'pyqmd collection list' to see available collections.\n"
    )


def test_rename_prints_virtual_path_line(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["rename", "a", "b"])

    assert result.exit_code == 0
    assert result.stdout == (
        "✓ Renamed collection 'a' to 'b'\n  Virtual paths updated: qmd://a/ → qmd://b/\n"
    )


def test_rename_onto_existing_name_prints_node_error(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["rename", "a", "b"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Collection name already exists: b\n"
        "Choose a different name or remove the existing collection first.\n"
    )
    assert store.get_collection("a") is not None


def test_add_success_colors_only_the_checkmark_green(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"], color=True)

    assert "\x1b[32m✓\x1b[0m Collection 'notes' created successfully" in result.stdout


def test_add_duplicate_name_first_line_is_yellow(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["add", str(tmp_path), "--name", "notes"], color=True)

    assert "\x1b[33mCollection 'notes' already exists.\x1b[0m" in result.stderr


def test_rename_colors_checkmark_green_and_uris_cyan(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["rename", "a", "b"], color=True)

    assert "\x1b[32m✓\x1b[0m Renamed collection 'a' to 'b'" in result.stdout
    assert "\x1b[36mqmd://a/\x1b[0m → \x1b[36mqmd://b/\x1b[0m" in result.stdout


def test_show_update_cmd_include_exclude_are_uncolored_like_node(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)

    for args in (
        ["show", "notes"],
        ["update-cmd", "notes", "git", "pull"],
        ["exclude", "notes"],
        ["include", "notes"],
    ):
        result = runner.invoke(app, args, color=True)
        assert result.exit_code == 0, args
        assert "\x1b[" not in result.stdout, args
