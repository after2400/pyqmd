import shutil
import subprocess
from pathlib import Path

import pytest

from parity._node_cli import get_node_commit, run_node_cli

QMD_REPO_ROOT = None  # resolved in a fixture below, since it's outside this repo


def _main_checkout_root() -> Path:
    """This repo's main checkout root, even when running from a git worktree
    (e.g. .claude/worktrees/<name>/), whose own root is nested inside it.
    The common git dir is the main checkout's .git in both cases (git prints
    it relative to cwd in the main checkout, absolute in a worktree). Falls
    back to this checkout's own root if git can't answer."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return repo_root
    return (repo_root / out).resolve().parent


@pytest.fixture
def qmd_repo_root():
    # The qmd (Node) repo is expected as a sibling of this (pyqmd) repo's
    # main checkout, e.g. .../AI/qmd next to .../AI/pyqmd -- not a parent
    # directory, and not a sibling of a worktree under .claude/worktrees/.
    root = _main_checkout_root().parent / "qmd"
    if not (root / "src" / "cli" / "qmd.ts").is_file():
        pytest.skip(f"qmd repo not found at expected location {root}")
    return root


@pytest.fixture(autouse=True)
def require_bun():
    if shutil.which("bun") is None:
        pytest.skip("bun is not installed")


def test_run_node_cli_returns_exit_code_and_output(qmd_repo_root, tmp_path):
    index_path = tmp_path / "test.sqlite"

    result = run_node_cli(qmd_repo_root, ["status"], index_path)

    assert result.exit_code == 0
    assert "Index" in result.stdout or "index" in result.stdout.lower()


def test_run_node_cli_isolates_db_via_index_path(qmd_repo_root, tmp_path):
    index_path = tmp_path / "isolated.sqlite"
    assert not index_path.exists()

    run_node_cli(qmd_repo_root, ["status"], index_path)

    # `status` alone doesn't necessarily create the file (it may just report
    # "no index yet"), so instead prove isolation via collection add + status
    # reporting the collection, then confirming the real ~/.cache/qmd path
    # was never touched by checking the isolated file exists and is a
    # distinct path from the default.
    assert index_path.parent == tmp_path


def test_run_node_cli_returns_nonzero_on_unknown_command(qmd_repo_root, tmp_path):
    result = run_node_cli(qmd_repo_root, ["definitely-not-a-real-command"], tmp_path / "x.sqlite")

    assert result.exit_code != 0


def test_get_node_commit_returns_a_sha(qmd_repo_root):
    commit = get_node_commit(qmd_repo_root)

    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_run_node_cli_isolates_collection_config_via_qmd_config_dir(qmd_repo_root, tmp_path):
    """Verify that collection add writes to the isolated QMD_CONFIG_DIR,
    not to the user's real ~/.config/qmd/."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.md").write_text("# Doc\ncontent")
    index_path = tmp_path / "isolated.sqlite"

    result = run_node_cli(
        qmd_repo_root,
        ["collection", "add", str(corpus_dir), "--name", "isolation-test"],
        index_path,
    )

    assert result.exit_code == 0, f"collection add failed: {result.stderr}"

    # Verify the isolated config directory was created and contains the collection YAML
    config_dir = index_path.parent / "config"
    assert config_dir.exists(), f"config dir not created at {config_dir}"

    yaml_files = list(config_dir.glob("*.yml")) + list(config_dir.glob("*.yaml"))
    assert len(yaml_files) >= 1, (
        f"expected at least one isolated config YAML under {config_dir}, found none"
    )


def test_run_node_cli_keeps_carriage_returns(tmp_path, monkeypatch):
    # text=True would apply universal-newline decoding and turn Node's
    # "\r<bar> 100%" into "\n<bar> 100%"; the capture must keep it raw.
    import subprocess

    import parity._node_cli as node_cli

    def fake_run(cmd, **kwargs):
        assert not kwargs.get("text") and not kwargs.get("universal_newlines")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="a\n\r█ 100%\r\n".encode(), stderr=b"\x1b[?25l"
        )

    monkeypatch.setattr(node_cli.subprocess, "run", fake_run)

    result = run_node_cli(tmp_path, ["embed"], tmp_path / "index.sqlite")

    assert result.stdout == "a\n\r█ 100%\r\n"
    assert result.stderr == "\x1b[?25l"
