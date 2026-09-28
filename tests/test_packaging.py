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
