"""Subprocess wrapper for invoking the pyqmd CLI via `uv run pyqmd`, with
its database isolated from the user's real index via the PYQMD_DB env var
(see pyqmd_mlx/cli/_db.py). Mirrors _node_cli.py's isolation approach so
benchmark.py can drive both sides the same way. Used only by benchmark.py
-- never by test_structural.py or test_quality.py.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PYQMD_REPO_ROOT = Path(__file__).resolve().parent.parent


def pyqmd_cli_env(db_path: Path) -> dict[str, str]:
    return {**os.environ, "PYQMD_DB": str(db_path)}


def pyqmd_cli_cmd(args: list[str]) -> list[str]:
    """Prefers an installed `pyqmd` on PATH (via `just install`) so the
    benchmark works with no install step either way; falls back to
    `uv run pyqmd` from the repo root. Measured directly (three back-to-
    back `search` runs each way on a synced project): the two are within
    noise of each other, ~0.44-0.50s vs. ~0.46-0.47s -- `uv run`'s own
    overhead here is negligible next to Python/Typer's own import cost, so
    this preference is for convenience, not measurement fairness. See
    pyqmd_invocation_kind() for reporting which path was used."""
    pyqmd_bin = shutil.which("pyqmd")
    if pyqmd_bin:
        return [pyqmd_bin, *args]
    return ["uv", "run", "pyqmd", *args]


def pyqmd_invocation_kind() -> str:
    if shutil.which("pyqmd"):
        return "installed `pyqmd` on PATH"
    return "`uv run pyqmd` (measured equivalent to the installed binary -- see pyqmd_cli_cmd)"
