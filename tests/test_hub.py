"""load_quietly_if_cached(): HF progress bars off for cached models only."""

import huggingface_hub
import pytest
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)

from pyqmd_mlx.llm._hub import is_cached, load_quietly_if_cached


@pytest.fixture(autouse=True)
def _bars_enabled():
    enable_progress_bars()
    yield
    enable_progress_bars()


def test_local_directory_counts_as_cached(tmp_path):
    assert is_cached(str(tmp_path)) is True


def test_hub_repo_cached_when_config_is_in_hf_cache(monkeypatch):
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo, name: "/c/config.json"
    )
    assert is_cached("org/model") is True


def test_hub_repo_not_cached_when_config_missing(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: None)
    assert is_cached("org/model") is False


def test_invalid_repo_id_is_not_cached(monkeypatch):
    def boom(repo, name):
        raise ValueError("bad repo id")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", boom)
    assert is_cached("not a repo") is False


def test_cached_load_runs_with_bars_disabled_then_restores(tmp_path):
    seen = []

    def loader(model_id):
        seen.append(are_progress_bars_disabled())
        return "model"

    assert load_quietly_if_cached(str(tmp_path), loader) == "model"
    assert seen == [True]
    assert are_progress_bars_disabled() is False


def test_uncached_load_keeps_bars(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: None)
    seen = []

    def loader(model_id):
        seen.append(are_progress_bars_disabled())
        return "model"

    load_quietly_if_cached("org/model", loader)
    assert seen == [False]


def test_bars_restored_when_loader_raises(tmp_path):
    def loader(model_id):
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        load_quietly_if_cached(str(tmp_path), loader)
    assert are_progress_bars_disabled() is False


def test_bars_stay_disabled_if_caller_disabled_them(tmp_path):
    disable_progress_bars()
    load_quietly_if_cached(str(tmp_path), lambda model_id: "model")
    assert are_progress_bars_disabled() is True
