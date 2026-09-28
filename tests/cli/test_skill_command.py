from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.skill import (
    _SKILLS_DIR,
    _skill_description,
    app,
    get_claude_skill_link_path,
    get_skill_install_dir,
)

runner = CliRunner()


def test_get_skill_install_dir_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert get_skill_install_dir(False) == tmp_path / ".agents" / "skills" / "pyqmd"


def test_get_skill_install_dir_global(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert get_skill_install_dir(True) == tmp_path / ".agents" / "skills" / "pyqmd"


def test_get_claude_skill_link_path_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert get_claude_skill_link_path(False) == tmp_path / ".claude" / "skills" / "pyqmd"


def test_get_claude_skill_link_path_global(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert get_claude_skill_link_path(True) == tmp_path / ".claude" / "skills" / "pyqmd"


def test_show_prints_skill_doc():
    result = runner.invoke(app, ["show"])

    assert result.exit_code == 0
    assert "# pyqmd" in result.output
    assert "pyqmd query" in result.output


def test_install_creates_stub_and_reference_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install"])

    assert result.exit_code == 0
    install_dir = tmp_path / ".agents" / "skills" / "pyqmd"
    stub = (install_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "pyqmd skill show" in stub
    assert "## Pitfalls" not in stub
    reference = (install_dir / "references" / "mcp-setup.md").read_text(encoding="utf-8")
    assert "MCP Server Setup" in reference


def test_install_without_force_errors_on_existing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)
    runner.invoke(app, ["install"])

    result = runner.invoke(app, ["install"])

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_install_with_force_overwrites(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)
    install_dir = tmp_path / ".agents" / "skills" / "pyqmd"
    install_dir.mkdir(parents=True)
    (install_dir / "stale.txt").write_text("old", encoding="utf-8")

    result = runner.invoke(app, ["install", "--force"])

    assert result.exit_code == 0
    assert not (install_dir / "stale.txt").exists()
    assert (install_dir / "SKILL.md").exists()


def test_install_global_uses_home_dir(monkeypatch, tmp_path):
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install", "--global"])

    assert result.exit_code == 0
    assert (tmp_path / "home" / ".agents" / "skills" / "pyqmd" / "SKILL.md").exists()
    assert not (tmp_path / "cwd" / ".agents").exists()


def test_install_yes_creates_claude_symlink(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["install", "--yes"])

    assert result.exit_code == 0
    link_path = tmp_path / ".claude" / "skills" / "pyqmd"
    assert link_path.is_symlink()
    assert link_path.resolve() == (tmp_path / ".agents" / "skills" / "pyqmd").resolve()


def test_install_non_interactive_without_yes_prints_tip_and_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install"])

    assert result.exit_code == 0
    assert "Tip:" in result.output
    assert not (tmp_path / ".claude" / "skills" / "pyqmd").exists()


def test_install_interactive_prompt_yes_creates_symlink(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: True)

    result = runner.invoke(app, ["install"], input="y\n")

    assert result.exit_code == 0
    assert (tmp_path / ".claude" / "skills" / "pyqmd").is_symlink()


def test_install_interactive_prompt_no_skips_symlink(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: True)

    result = runner.invoke(app, ["install"], input="n\n")

    assert result.exit_code == 0
    assert not (tmp_path / ".claude" / "skills" / "pyqmd").exists()


def test_install_second_run_detects_self_loop(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["install", "--yes"])

    result = runner.invoke(app, ["install", "--force", "--yes"])

    assert result.exit_code == 0
    assert "already sees the skill" in result.output


def test_install_existing_non_matching_claude_path_errors_without_force(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    claude_dir = tmp_path / ".claude" / "skills"
    claude_dir.mkdir(parents=True)
    (claude_dir / "pyqmd").symlink_to(elsewhere, target_is_directory=True)

    result = runner.invoke(app, ["install", "--yes"])

    assert result.exit_code == 1
    assert "Claude skill path already exists" in result.output


def test_show_includes_collection_routing_and_daemon_hint():
    result = runner.invoke(app, ["show"])

    assert result.exit_code == 0
    assert "## Collection routing" in result.output
    assert "pyqmd collection list" in result.output
    assert "qmd://<collection>/<path>" in result.output
    assert "--chunk-strategy auto" in result.output
    assert "MCP-first" in result.output


def test_install_reference_has_daemon_table():
    ref = (_SKILLS_DIR / "pyqmd" / "references" / "mcp-setup.md").read_text(encoding="utf-8")
    assert "pyqmd mcp --http --port 8181" in ref
    assert "QMD_ALLOWED_HOSTS" in ref
    assert "Claude Code" in ref and "OpenCode" in ref


def test_show_namespaced_skill_and_unknown_errors():
    ok = runner.invoke(app, ["show", "pyqmd"])
    assert ok.exit_code == 0
    assert "# pyqmd" in ok.output

    bad = runner.invoke(app, ["show", "nope"])
    assert bad.exit_code == 1
    assert "Unknown skill" in bad.output
    assert "pyqmd" in bad.output


def test_install_namespaced_skill_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install", "pyqmd"])

    assert result.exit_code == 0
    assert (tmp_path / ".agents" / "skills" / "pyqmd" / "SKILL.md").exists()


def test_librarian_skill_shows_and_installs(monkeypatch, tmp_path):
    shown = runner.invoke(app, ["show", "pyqmd-librarian"])
    assert shown.exit_code == 0
    assert "confirm-before-write" in shown.output
    assert "cleanup --dry-run" in shown.output

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)
    installed = runner.invoke(app, ["install", "pyqmd-librarian"])
    assert installed.exit_code == 0
    stub = (tmp_path / ".agents" / "skills" / "pyqmd-librarian" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "pyqmd skill show pyqmd-librarian" in stub


@pytest.mark.parametrize("name", ["pyqmd", "pyqmd-librarian", "pyqmd-researcher", "pyqmd-bench"])
def test_install_stub_carries_the_skills_own_description(monkeypatch, tmp_path, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install", name])

    assert result.exit_code == 0
    stub = (tmp_path / ".agents" / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    assert f"description: {_skill_description(name)}" in stub


def test_install_tip_names_the_installed_skill(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)

    result = runner.invoke(app, ["install", "pyqmd-bench"])

    assert result.exit_code == 0
    assert "pyqmd skill install pyqmd-bench --force --yes" in result.output


def test_list_shows_bundled_skills():
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "pyqmd" in result.output
    assert "pyqmd-librarian" in result.output


def test_researcher_skill_shows_lists_and_installs(monkeypatch, tmp_path):
    shown = runner.invoke(app, ["show", "pyqmd-researcher"])
    assert shown.exit_code == 0
    assert "never trust snippets" in shown.output
    assert "--intent" in shown.output

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "pyqmd-researcher" in listed.output

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)
    installed = runner.invoke(app, ["install", "pyqmd-researcher"])
    assert installed.exit_code == 0
    assert (tmp_path / ".agents" / "skills" / "pyqmd-researcher" / "SKILL.md").exists()


def test_bench_skill_shows_lists_and_installs(monkeypatch, tmp_path):
    shown = runner.invoke(app, ["show", "pyqmd-bench"])
    assert shown.exit_code == 0
    assert "expected_in_top_k" in shown.output
    assert "nDCG" in shown.output

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "pyqmd-bench" in listed.output

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.skill._is_interactive", lambda: False)
    installed = runner.invoke(app, ["install", "pyqmd-bench"])
    assert installed.exit_code == 0
    assert (tmp_path / ".agents" / "skills" / "pyqmd-bench" / "SKILL.md").exists()
