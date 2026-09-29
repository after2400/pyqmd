import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.collection import app as collection_app
from pyqmd_mlx.cli.commands.update import app
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import scan_and_register_collection

runner = CliRunner()

_GIT_AVAILABLE = shutil.which("git") is not None


def _init_git_repo(path, initial_content="# Doc\ninitial"):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "a.md").write_text(initial_content)
    subprocess.run(["git", "add", "a.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=path, check=True)


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def test_update_with_no_collections_prints_message_and_exits_zero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert result.stdout == (
        "No collections found. Run 'pyqmd collection add .' to index markdown files.\n"
    )


def test_update_all_collections_prints_framing_and_counts(tmp_path, monkeypatch):
    coll_a = tmp_path / "a"
    coll_a.mkdir()
    (coll_a / "x.md").write_text("# X\nhello")
    coll_b = tmp_path / "b"
    coll_b.mkdir()
    (coll_b / "y.md").write_text("# Y\nworld")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", str(coll_a))
    store.add_collection("b", str(coll_b))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Updating 2 collection(s)..." in result.output
    assert "[1/2] a (**/*.md)" in result.output
    assert "[2/2] b (**/*.md)" in result.output
    assert result.output.count("Indexed: 1 new, 0 updated, 0 unchanged, 0 removed") == 2
    assert "✓ All collections updated." in result.output
    assert "Run 'pyqmd embed'" in result.output


def test_update_prints_orphan_cleanup_line_when_content_hash_changes(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    scan_and_register_collection(store, str(coll), "**/*.md", "notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    (coll / "a.md").write_text("# A\nhello, changed!")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Cleaned up 1 orphaned content hash(es)" in result.output


def test_update_rescans_with_persisted_exclude_patterns_from_add(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\nhello")
    (tmp_path / "draft.md").write_text("# Draft\nskip me")

    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.collection.get_store", lambda db_path=None: store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    add_result = runner.invoke(
        collection_app, ["add", str(tmp_path), "--name", "notes", "--exclude", "draft.md"]
    )

    assert add_result.exit_code == 0
    assert store.get_collection("notes")["ignore_patterns"] == "draft.md"

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Indexed: 0 new, 0 updated, 1 unchanged, 0 removed" in result.output
    assert store.find_active_document("notes", "a.md") is not None
    assert store.find_active_document("notes", "draft.md") is None


def test_update_pending_embed_hint_omitted_when_zero(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    scan_and_register_collection(store, str(coll), "**/*.md", "notes")
    doc = store.find_active_document("notes", "a.md")
    store.index_content(doc["hash"], "# A\nhello")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Run 'pyqmd embed'" not in result.output


def test_update_scoped_to_named_collection(tmp_path, monkeypatch):
    coll_a = tmp_path / "a"
    coll_a.mkdir()
    (coll_a / "x.md").write_text("# X\nhello")
    coll_b = tmp_path / "b"
    coll_b.mkdir()
    (coll_b / "y.md").write_text("# Y\nworld")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", str(coll_a))
    store.add_collection("b", str(coll_b))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update", "-c", "a"])

    assert result.exit_code == 0
    assert "[1/1] a (**/*.md)" in result.output
    assert "b (**/*.md)" not in result.output


def test_update_unknown_collection_name_errors_before_touching_any_collection(
    tmp_path, monkeypatch
):
    coll_a = tmp_path / "a"
    coll_a.mkdir()
    (coll_a / "x.md").write_text("# X\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", str(coll_a))
    scan_and_register_collection(store, str(coll_a), "**/*.md", "a")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    (coll_a / "new.md").write_text("# New\nfile")

    result = runner.invoke(app, ["update", "-c", "a", "-c", "nope"])

    assert result.exit_code == 1
    assert store.find_active_document("a", "new.md") is None


def test_update_runs_hook_before_scan(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "existing.md").write_text("# Existing\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll), update_command=f"echo generated > {coll}/generated.md")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Running update command:" in result.output
    assert store.find_active_document("notes", "generated.md") is not None


def test_update_no_hook_output_when_update_command_unset(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert "Running update command:" not in result.output


def test_update_hook_nonzero_exit_stops_whole_command(tmp_path, monkeypatch):
    coll_a = tmp_path / "a"
    coll_a.mkdir()
    (coll_a / "x.md").write_text("# X\nhello")
    coll_b = tmp_path / "b"
    coll_b.mkdir()
    (coll_b / "y.md").write_text("# Y\nworld")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", str(coll_a), update_command="exit 3")
    store.add_collection("b", str(coll_b))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 3
    assert store.find_active_document("b", "y.md") is None


def test_update_deleted_collection_path_with_hook_errors_cleanly(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll), update_command="echo hi")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    coll.rmdir()

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Collection path no longer exists" in result.output
    assert "Traceback" not in result.output


def test_update_deleted_collection_path_no_hook_deactivates_all_documents(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    scan_and_register_collection(store, str(coll), "**/*.md", "notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    shutil.rmtree(coll)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert "1 removed" in result.output
    assert store.find_active_document("notes", "a.md") is None


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_update_pull_runs_real_git_pull_and_indexes_new_remote_commit(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _init_git_repo(remote)

    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(remote), str(local)], check=True)

    (remote / "b.md").write_text("# Second\nnew commit")
    subprocess.run(["git", "add", "b.md"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=remote, check=True)

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(local))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update", "--pull"])

    assert result.exit_code == 0
    assert "Pulling latest changes..." in result.output
    assert store.find_active_document("notes", "b.md") is not None


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_update_pull_skipped_quietly_without_git_directory(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update", "--pull"])

    assert result.exit_code == 0
    assert "Pulling latest changes..." not in result.output
    assert store.find_active_document("notes", "a.md") is not None


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_update_pull_failure_skips_collection_and_reports_in_trailer(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _init_git_repo(remote)

    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(remote), str(local)], check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=local, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=local, check=True)
    (local / "a.md").write_text("# Doc\nlocal divergent change")
    subprocess.run(["git", "add", "a.md"], cwd=local, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "local diverge"], cwd=local, check=True)

    (remote / "a.md").write_text("# Doc\nremote divergent change")
    subprocess.run(["git", "add", "a.md"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "remote diverge"], cwd=remote, check=True)

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(local))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update", "--pull"])

    assert result.exit_code == 0
    assert "git pull failed, skipping this collection" in result.output
    assert "✓ Updated 0 of 1 collection(s) (1 skipped)." in result.output


def test_update_pull_git_missing_from_path_warns_and_proceeds(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / ".git").mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(app, ["update", "--pull"])

    assert result.exit_code == 0
    assert "git unavailable, continuing without pull" in result.output
    assert store.find_active_document("notes", "a.md") is not None


def test_update_pull_dot_git_as_file_treated_as_non_repo(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / ".git").write_text("gitdir: /somewhere/else")
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update", "--pull"])

    assert result.exit_code == 0
    assert "Pulling latest changes..." not in result.output


def test_update_without_pull_flag_runs_no_git_commands(tmp_path, monkeypatch):
    coll = tmp_path / "notes"
    coll.mkdir()
    (coll / ".git").mkdir()
    (coll / "a.md").write_text("# A\nhello")

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(coll))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    calls = []
    real_run = subprocess.run

    def _recording_run(args, *a, **kw):
        calls.append(args)
        return real_run(args, *a, **kw)

    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.subprocess.run", _recording_run)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert not any(args[0] == "git" for args in calls)


def test_update_prints_node_layout(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\nalpha")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert result.stdout == (
        "Updating 1 collection(s)...\n"
        "\n"
        "[1/1] notes (**/*.md)\n"
        f"Collection: {tmp_path} (**/*.md)\n"
        "\n"
        "Indexed: 1 new, 0 updated, 0 unchanged, 0 removed\n"
        "\n"
        "✓ All collections updated.\n"
        "\n"
        "Run 'pyqmd embed' to update embeddings (1 unique hashes need vectors)\n"
    )


def test_update_hook_output_is_indented_and_announced(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path), update_command="echo one; echo two")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert "    Running update command: echo one; echo two\n    one\n    two\n" in result.stdout


def test_update_hook_failure_is_reported_on_stdout_like_node(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path), update_command="exit 3")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 3
    assert "✗ Update command failed with exit code 3\n" in result.stdout


def test_update_colors_match_node(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\nalpha")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["update"], color=True)

    assert "\x1b[1mUpdating 1 collection(s)...\x1b[0m" in result.stdout
    assert "\x1b[36m[1/1]\x1b[0m \x1b[1mnotes\x1b[0m \x1b[2m(**/*.md)\x1b[0m" in result.stdout
    assert "\x1b[32m✓ All collections updated.\x1b[0m" in result.stdout
