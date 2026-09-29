"""Shared pytest fixtures for the parity suite. Registers --dataset-config
and provides the active DatasetProfile plus a fully-indexed pyqmd Store
built from its corpus_dir -- session-scoped, since indexing (and
especially embedding) the full corpus is expensive and should happen at
most once per test run, not once per test.
"""

from __future__ import annotations

import pytest

from parity.dataset_profile import DatasetProfile, load_dataset_profile, resolve_active_profile_path
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import scan_and_register_collection


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-config",
        action="store",
        default=None,
        help="Path to a parity dataset profile YAML file (default: the built-in scifact profile)",
    )


@pytest.fixture(scope="session")
def active_profile(pytestconfig) -> DatasetProfile:
    config_path = resolve_active_profile_path(pytestconfig.getoption("--dataset-config"))
    return load_dataset_profile(config_path)


@pytest.fixture(scope="session")
def indexed_pyqmd_store(active_profile, tmp_path_factory) -> Store:
    """A real Store, with active_profile.corpus_dir fully scanned,
    registered, and embedded via the real pyqmd_mlx.llm pipeline (real MLX
    models -- this is the expensive-once operation)."""
    db_path = tmp_path_factory.mktemp("parity_pyqmd") / "index.sqlite"
    store = Store(str(db_path))
    store.add_collection(active_profile.name, str(active_profile.corpus_dir))
    scan_and_register_collection(
        store, str(active_profile.corpus_dir), "**/*.md", active_profile.name
    )
    content = store.get_indexable_content(active_profile.name)
    for row in content:
        store.index_content(row["hash"], row["doc"], filepath=row["path"])
    yield store
    store.close()
