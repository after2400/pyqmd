"""Hide Hugging Face's "Fetching N files" bar when a model is already cached.

mlx_lm/mlx_embeddings load through huggingface_hub.snapshot_download, which
shows that bar on every load -- even when every file is already in the HF
cache and nothing downloads. Real download progress (a first run) stays.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def is_cached(model_id: str) -> bool:
    """True for a local model directory, or a Hub repo whose config.json is in the HF cache."""
    if Path(model_id).is_dir():
        return True
    import huggingface_hub

    try:
        return isinstance(huggingface_hub.try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


def load_quietly_if_cached(model_id: str, loader: Callable[[str], T]) -> T:
    """Run loader(model_id), with HF progress bars off if nothing needs downloading."""
    if not is_cached(model_id):
        return loader(model_id)
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        return loader(model_id)
    finally:
        if not was_disabled:
            enable_progress_bars()
