"""Tests for `pyqmd skills` (plural) — Node's generic multi-skill discovery
surface (`list`/`get`/`path`, `--json`), ported from `runSkillsCommand` in
Node's `src/cli/qmd.ts`."""

import json
from pathlib import Path

from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app as root_app
from pyqmd_mlx.cli.commands.skills import _SKILLS_DIR, app

runner = CliRunner()

BUNDLED = ["pyqmd", "pyqmd-bench", "pyqmd-librarian", "pyqmd-researcher"]


def _skill_md(name: str) -> str:
    return (_SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def test_list_shows_all_bundled_skills():
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    for name in BUNDLED:
        assert name in result.output


def test_list_pads_names_like_node():
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    longest = max(len(n) for n in BUNDLED)
    for name in BUNDLED:
        assert f"  {name.ljust(longest)}  " in result.output


def test_bare_invocation_defaults_to_list():
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    for name in BUNDLED:
        assert name in result.output


def test_bare_invocation_with_json_flag_lists_as_json():
    result = runner.invoke(app, ["--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert [s["name"] for s in payload["data"]] == sorted(BUNDLED)


def test_list_json_envelope():
    result = runner.invoke(app, ["list", "--json"])

    assert result.exit_code == 0
    assert result.output.lstrip().startswith("{")
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert [s["name"] for s in payload["data"]] == sorted(BUNDLED)
    assert all(set(s) == {"name", "description"} for s in payload["data"])


def test_group_level_json_flag_applies_to_list():
    result = runner.invoke(app, ["--json", "list"])

    assert result.exit_code == 0
    assert result.output.lstrip().startswith("{")
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert [s["name"] for s in payload["data"]] == sorted(BUNDLED)


def test_group_level_json_flag_applies_to_get():
    result = runner.invoke(app, ["--json", "get", "pyqmd"])

    assert result.exit_code == 0
    assert result.output.lstrip().startswith("{")
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["data"][0]["name"] == "pyqmd"


def test_group_level_json_flag_applies_to_path():
    result = runner.invoke(app, ["--json", "path", "pyqmd"])

    assert result.exit_code == 0
    assert result.output.lstrip().startswith("{")
    payload = json.loads(result.output)
    assert payload == {
        "success": True,
        "data": {"name": "pyqmd", "path": str(_SKILLS_DIR / "pyqmd")},
    }


def test_get_prints_skill_verbatim():
    expected = _skill_md("pyqmd")

    result = runner.invoke(app, ["get", "pyqmd"])

    assert result.exit_code == 0
    assert result.output == (expected if expected.endswith("\n") else expected + "\n")


def test_get_full_includes_reference_files():
    result = runner.invoke(app, ["get", "pyqmd", "--full"])

    assert result.exit_code == 0
    assert _skill_md("pyqmd") in result.output
    assert "\n--- references/mcp-setup.md ---\n" in result.output
    reference = (_SKILLS_DIR / "pyqmd" / "references" / "mcp-setup.md").read_text(encoding="utf-8")
    assert reference in result.output


def test_get_all_prints_every_skill_separated():
    result = runner.invoke(app, ["get", "--all"])

    assert result.exit_code == 0
    for name in BUNDLED:
        assert _skill_md(name) in result.output
    assert "\n---\n" in result.output


def test_get_all_json_envelope():
    result = runner.invoke(app, ["get", "--all", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert [s["name"] for s in payload["data"]] == sorted(BUNDLED)
    assert all(set(s) == {"name", "content"} for s in payload["data"])


def test_get_full_json_includes_files():
    result = runner.invoke(app, ["get", "pyqmd", "--full", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    (entry,) = payload["data"]
    assert entry["name"] == "pyqmd"
    assert entry["content"] == _skill_md("pyqmd")
    assert {"path": "references/mcp-setup.md", "content": entry["files"][0]["content"]} == {
        "path": "references/mcp-setup.md",
        "content": (_SKILLS_DIR / "pyqmd" / "references" / "mcp-setup.md").read_text(
            encoding="utf-8"
        ),
    }


def test_get_unknown_skill_errors():
    result = runner.invoke(app, ["get", "nope"])

    assert result.exit_code == 1
    assert "Skill not found: nope" in result.output


def test_get_without_name_errors():
    result = runner.invoke(app, ["get"])

    assert result.exit_code == 1
    assert "No skill name provided. Usage: pyqmd skills get <name>" in result.output


def test_get_unknown_skill_json_envelope():
    result = runner.invoke(app, ["get", "nope", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {"success": False, "error": "Skill not found: nope"}


def test_path_without_name_prints_search_dirs():
    result = runner.invoke(app, ["path"])

    assert result.exit_code == 0
    assert result.output.strip() == str(_SKILLS_DIR)


def test_path_with_name_prints_skill_dir():
    result = runner.invoke(app, ["path", "pyqmd-librarian"])

    assert result.exit_code == 0
    assert result.output.strip() == str(_SKILLS_DIR / "pyqmd-librarian")


def test_path_json_without_name():
    result = runner.invoke(app, ["path", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"success": True, "data": {"paths": [str(_SKILLS_DIR)]}}


def test_path_json_with_name():
    result = runner.invoke(app, ["path", "pyqmd-bench", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "success": True,
        "data": {"name": "pyqmd-bench", "path": str(_SKILLS_DIR / "pyqmd-bench")},
    }


def test_path_unknown_skill_errors():
    result = runner.invoke(app, ["path", "nope"])

    assert result.exit_code == 1
    assert "Skill not found: nope" in result.output


def test_unknown_subcommand_exits_nonzero():
    result = runner.invoke(app, ["frobnicate"])

    assert result.exit_code != 0


def test_help_prints_usage():
    result = runner.invoke(app, ["help"])

    assert result.exit_code == 0
    assert "Usage: pyqmd skills <list|get|path> [options]" in result.output
    assert "--json" in result.output


def _write_skill(directory: Path, name: str, hidden: bool = False) -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True)
    hidden_line = "\nhidden: true" if hidden else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} does things.{hidden_line}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def test_empty_skills_dir_lists_nothing_found(monkeypatch, tmp_path):
    monkeypatch.setenv("PYQMD_SKILLS_DIR", str(tmp_path))

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert result.output.strip() == "No skills found"


def test_skills_dir_env_override(monkeypatch, tmp_path):
    _write_skill(tmp_path, "custom-skill")
    monkeypatch.setenv("PYQMD_SKILLS_DIR", str(tmp_path))

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "custom-skill" in result.output
    assert "pyqmd" not in result.output


def test_hidden_skills_excluded_from_list_but_gettable(monkeypatch, tmp_path):
    _write_skill(tmp_path, "visible-skill")
    _write_skill(tmp_path, "secret-skill", hidden=True)
    monkeypatch.setenv("PYQMD_SKILLS_DIR", str(tmp_path))

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "visible-skill" in listed.output
    assert "secret-skill" not in listed.output

    all_skills = runner.invoke(app, ["get", "--all"])
    assert all_skills.exit_code == 0
    assert "secret-skill" not in all_skills.output

    direct = runner.invoke(app, ["get", "secret-skill"])
    assert direct.exit_code == 0
    assert "# secret-skill" in direct.output


def test_skills_group_mounted_on_root_app():
    result = runner.invoke(root_app, ["skills", "list"])

    assert result.exit_code == 0
    for name in BUNDLED:
        assert name in result.output
