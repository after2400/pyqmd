"""Generic model-id-keyed lazy cache, shared by embed/rerank/expand_query."""

from typing import Callable, TypeVar

T = TypeVar("T")


def get_or_load(cache: dict[str, T], model_id: str, loader: Callable[[str], T]) -> T:
    """Return cache[model_id], loading and storing it via loader() on first access."""
    if model_id not in cache:
        cache[model_id] = loader(model_id)
    return cache[model_id]
