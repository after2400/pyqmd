"""Shared snapshot-file loading for test_structural.py -- also usable by
test_quality.py for the quality-mode files, since the on-disk layout is
consistent (parity/node_ref/<profile>/...)."""

from __future__ import annotations

import json
from pathlib import Path

_NODE_REF_DIR = Path(__file__).resolve().parent / "node_ref"


def load_snapshot(profile_name: str, category: str, scenario_name: str) -> dict:
    path = _NODE_REF_DIR / profile_name / category / f"{scenario_name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No captured snapshot at {path}. Run "
            f"`uv run python -m parity.capture_node_snapshots --qmd-repo-root <path>` first."
        )
    return json.loads(path.read_text())


def load_quality_file(profile_name: str, filename: str) -> dict:
    path = _NODE_REF_DIR / profile_name / "quality" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"No captured quality file at {path}. Run capture_node_snapshots.py first."
        )
    return json.loads(path.read_text())


def load_flow_raw(profile_name: str, flow_name: str) -> dict | None:
    """Raw Node stdout/stderr per flow step (capture_node_snapshots.py's
    cli_flow_raw/). None when not captured yet -- the text-parity test
    skips rather than fails, so the suite stays green until someone runs
    a `--phase cli-flow` capture."""
    path = _NODE_REF_DIR / profile_name / "cli_flow_raw" / f"{flow_name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
