"""Subprocess wrapper for invoking the reference Node qmd CLI via `bun`,
with its database and collection config fully isolated from the user's real
index. Used only by capture_node_snapshots.py (manual, on-demand) -- never by
test_structural.py or test_quality.py, which compare against already-captured
JSON files and have no runtime dependency on bun or a working Node checkout.

Isolation mechanisms:
- INDEX_PATH (env var): Controls the SQLite database location
  (verified against src/store.ts:663-680 -- overrides all other DB sources)
- QMD_CONFIG_DIR (env var): Controls the collection registry YAML location
  (verified against src/collections.ts -- Node qmd stores the registry in
  <QMD_CONFIG_DIR>/<indexName>.yml, not in the SQLite DB. Without this
  isolation, collection operations would write to the user's real
  ~/.config/qmd/index.yml and merge in pre-existing collections.)

Both are set automatically by run_node_cli from the index_path argument.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NodeCliResult:
    exit_code: int
    stdout: str
    stderr: str


def node_cli_env(index_path: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Builds the isolated env for one Node invocation, creating its
    QMD_CONFIG_DIR as a side effect. Factored out of run_node_cli so
    benchmark.py can drive the same isolated env through a different
    subprocess wrapper (parity/_timing.py's run_timed, for wall-clock/RSS
    measurement) without duplicating the isolation logic."""
    config_dir = index_path.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "INDEX_PATH": str(index_path),
        "QMD_CONFIG_DIR": str(config_dir),
    }
    if extra_env:
        env.update(extra_env)
    return env


def node_cli_cmd(args: list[str]) -> list[str]:
    return ["bun", "src/cli/qmd.ts", *args]


def run_node_cli(
    qmd_repo_root: Path,
    args: list[str],
    index_path: Path,
    extra_env: dict[str, str] | None = None,
) -> NodeCliResult:
    """Run `bun src/cli/qmd.ts <args>` from qmd_repo_root, with both the
    database and collection config isolated via INDEX_PATH and QMD_CONFIG_DIR.
    See module docstring for isolation mechanism details."""
    env = node_cli_env(index_path, extra_env)

    # Bytes, decoded without universal-newline translation (which text=True
    # would apply): Node's in-place output starts with "\r" (e.g. embed's
    # final "\r<bar> 100%" line), and a raw capture must keep it.
    result = subprocess.run(
        node_cli_cmd(args),
        cwd=qmd_repo_root,
        env=env,
        capture_output=True,
        timeout=600,
    )
    return NodeCliResult(
        exit_code=result.returncode,
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
    )


def get_node_commit(qmd_repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=qmd_repo_root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()
