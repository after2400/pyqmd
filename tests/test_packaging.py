"""Pins the distribution/import/command names from the 2026-09-24
import-package-rename spec, so a future edit can't reintroduce a
top-level `qmd` package or drop the `pyqmd` command."""

import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_distribution_name():
    assert PYPROJECT["project"]["name"] == "pyqmd-mlx"


def test_command_is_pyqmd_and_targets_pyqmd_mlx():
    assert PYPROJECT["project"]["scripts"] == {"pyqmd": "pyqmd_mlx.cli.app:main"}


def test_wheel_ships_only_pyqmd_mlx():
    assert PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/pyqmd_mlx"]


def test_docs_commits_do_not_release():
    # docs/ changes never ship in the wheel; see the 2026-09-28 PyPI
    # publishing spec. Anything under src/ (incl. SKILL.md) is fix/feat.
    options = PYPROJECT["tool"]["semantic_release"]["commit_parser_options"]
    assert options["patch_tags"] == ["fix", "perf", "chore"]
    assert "docs" in options["allowed_tags"]


def test_doh_and_review_commits_never_release_or_reach_the_changelog():
    import re

    release = PYPROJECT["tool"]["semantic_release"]
    options = release["commit_parser_options"]
    excludes = release["changelog"]["exclude_commit_patterns"]
    for tag in ("doh", "review"):
        assert tag in options["allowed_tags"]
        assert tag not in options["patch_tags"] + options["minor_tags"]
        for subject in (f"{tag}: x", f"{tag}(llm): x"):
            assert any(re.match(p, subject) for p in excludes), subject
