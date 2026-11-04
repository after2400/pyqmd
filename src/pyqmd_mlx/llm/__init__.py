"""MLX-based LLM layer for qmd: embed, rerank, query expansion."""

from ._constants import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXPAND_MODEL,
    DEFAULT_RERANK_MODEL,
    EXPAND_MODEL_ENV_VAR,
    RERANK_MODEL_ENV_VAR,
)
from .embed import embed
from .expand import ExpansionModelError, expand_query, resolve_expand_model
from .rerank import RerankModelError, rerank, resolve_rerank_model

__all__ = [
    "embed",
    "rerank",
    "expand_query",
    "resolve_expand_model",
    "resolve_rerank_model",
    "ExpansionModelError",
    "RerankModelError",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_EXPAND_MODEL",
    "EXPAND_MODEL_ENV_VAR",
    "RERANK_MODEL_ENV_VAR",
]
