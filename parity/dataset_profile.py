"""Loads a dataset profile: a YAML file naming a corpus directory, a
queries file, and an optional relevance-judgments (qrels) file. Every
other module in this suite reads from the active profile rather than
hardcoding a dataset -- see docs/specs/2026-09-12-python-node-
parity-suite-design.md's "Dataset profiles" section for the full rationale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_BUILTIN_DEFAULT = Path(__file__).resolve().parent / "datasets" / "scifact.yaml"
_ENV_VAR = "PARITY_DATASET_CONFIG"


@dataclass
class DatasetProfile:
    name: str
    corpus_dir: Path
    queries_file: Path
    qrels_file: Path | None

    @property
    def has_qrels(self) -> bool:
        return self.qrels_file is not None


def load_dataset_profile(config_path: str | Path) -> DatasetProfile:
    """Load and validate a dataset profile YAML file. Paths inside it
    resolve relative to the config file's own directory, not the caller's
    cwd, so a profile is portable regardless of where pytest is invoked
    from."""
    config_path = Path(config_path).resolve()
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.parent

    for required in ("name", "corpus_dir", "queries_file"):
        if required not in raw:
            raise ValueError(f"dataset profile {config_path}: missing required field '{required}'")

    corpus_dir = (base_dir / raw["corpus_dir"]).resolve()
    if not corpus_dir.is_dir():
        hint = (
            " Run 'uv run scripts/prepare_scifact_corpus.py' to generate it "
            "-- it downloads BEIR SciFact (~33MB) and isn't tracked in git, "
            "so a fresh clone starts without it."
            if config_path == _BUILTIN_DEFAULT
            else ""
        )
        raise ValueError(
            f"dataset profile {config_path}: corpus_dir '{corpus_dir}' is not a directory.{hint}"
        )

    queries_file = (base_dir / raw["queries_file"]).resolve()
    if not queries_file.is_file():
        raise ValueError(
            f"dataset profile {config_path}: queries_file '{queries_file}' does not exist"
        )

    qrels_raw = raw.get("qrels_file")
    qrels_file = (base_dir / qrels_raw).resolve() if qrels_raw else None
    if qrels_file is not None and not qrels_file.is_file():
        raise ValueError(f"dataset profile {config_path}: qrels_file '{qrels_file}' does not exist")

    return DatasetProfile(
        name=raw["name"], corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=qrels_file
    )


def resolve_active_profile_path(cli_arg: str | None = None) -> Path:
    """Resolution order: explicit CLI argument, then PARITY_DATASET_CONFIG
    env var, then the built-in scifact profile."""
    if cli_arg is not None:
        return Path(cli_arg)
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return Path(env_value)
    return _BUILTIN_DEFAULT
