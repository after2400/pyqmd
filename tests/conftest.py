import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _expansion_weights_present() -> bool:
    """True when the resolved expansion model loads without a network
    download: a local MLX model dir, or a Hub repo already in the HF cache.
    Keeps CI and fresh worktrees from pulling ~934 MB just to run tests."""
    from pyqmd_mlx.llm import resolve_expand_model

    model = resolve_expand_model()
    if Path(model).is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        # Only the files mlx_lm.load itself fetches: it skips README.md and
        # .gitattributes, which huggingface_hub would otherwise count as an
        # incomplete snapshot.
        snapshot_download(
            model, local_files_only=True, allow_patterns=["*.json", "model*.safetensors"]
        )
    except Exception:
        return False
    return True


def _scifact_corpus_present() -> bool:
    from parity.dataset_profile import resolve_active_profile_path

    try:
        from parity.dataset_profile import load_dataset_profile

        load_dataset_profile(resolve_active_profile_path(None))
    except ValueError:
        return False
    return True


def pytest_runtest_setup(item):
    if "requires_expansion_weights" in item.keywords and not _expansion_weights_present():
        pytest.skip(
            "expansion model not available locally -- set PYQMD_EXPAND_MODEL to a "
            "local MLX model dir, or pre-download the default Hub repo into the HF cache"
        )
    if "requires_scifact_corpus" in item.keywords and not _scifact_corpus_present():
        pytest.skip(
            "scifact corpus missing (data/scifact is gitignored) -- run "
            "'uv run scripts/prepare_scifact_corpus.py' to generate it"
        )
