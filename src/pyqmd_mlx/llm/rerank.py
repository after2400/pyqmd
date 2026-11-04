"""rerank(): Qwen3-Reranker yes/no-logit scoring, run directly on mlx-lm,
or a cross-encoder checkpoint scored through mlx_embeddings.

mlx_embeddings cannot do the yes/no scoring for the plain-text Qwen3-Reranker
family (see the design spec's "Research finding" section) -- mlx-lm keeps the
LM head intact, so we implement the scoring ourselves.
"""

import os

from . import _cross_encoder
from ._cache import get_or_load
from ._constants import DEFAULT_RERANK_MODEL, RERANK_MODEL_ENV_VAR
from ._hub import load_quietly_if_cached
from ._prompts import build_rerank_prompt, score_from_logits

_rerank_cache: dict = {}
_cross_encoder_cache: dict = {}


def resolve_rerank_model() -> str:
    """The effective rerank model: $PYQMD_RERANK_MODEL when set and
    non-empty, else DEFAULT_RERANK_MODEL."""
    return os.environ.get(RERANK_MODEL_ENV_VAR) or DEFAULT_RERANK_MODEL


class RerankModelError(RuntimeError):
    """The rerank model couldn't be loaded (bad repo id or path, not
    downloadable, ...). Lets the CLI and MCP server report one clean line
    that names the override."""


def _load(model_id: str):
    # Deferred -- see embed.py's _load for why: a module-level mlx_lm import
    # would tax every CLI command via pyqmd_mlx.store.Store, not just rerank().
    from mlx_lm import load as _load_lm_model

    return load_quietly_if_cached(model_id, _load_lm_model)


def _last_position_logits(rerank_model, inputs):
    """The next-token logits at the prompt's last position only.

    Calling the model directly applies the output layer to every position:
    a vocab-sized (~152k) row per prompt token, ~1 GB for a long candidate,
    of which scoring reads two values. Running the transformer body and
    applying the output layer to just the last hidden state gives the same
    logits there, bit for bit.

    Batching candidates (right-padded) was tried and rejected: no faster on
    Apple Silicon, and padding shifted scores enough to reorder results."""
    hidden = rerank_model.model(inputs)[:, -1:, :]
    if rerank_model.args.tie_word_embeddings:
        return rerank_model.model.embed_tokens.as_linear(hidden)[0, 0, :]
    return rerank_model.lm_head(hidden)[0, 0, :]


def _load_error(model_id: str, exc: Exception) -> RerankModelError:
    return RerankModelError(
        f"Could not load rerank model '{model_id}': {str(exc).rstrip('.')}. "
        f"Set {RERANK_MODEL_ENV_VAR} to a Hugging Face repo id or a local "
        "MLX model directory to override."
    )


def _load_yes_no(model_id: str):
    try:
        return _load(model_id)
    except Exception as exc:
        raise _load_error(model_id, exc) from exc


def _load_cross_encoder(model_id: str):
    try:
        return _cross_encoder.load_cross_encoder(model_id)
    except Exception as exc:
        raise _load_error(model_id, exc) from exc


def rerank(query: str, documents: list[str], model: str = DEFAULT_RERANK_MODEL) -> list[float]:
    """Score each document's relevance to query, in the same order as documents.

    The backend follows the checkpoint, not the model name: a config.json
    listing a `*ForSequenceClassification` architecture is a cross-encoder
    (scored from the pair, see _cross_encoder); anything else is a causal LM
    scored by its yes/no logits (the Qwen3-Reranker family). A model already
    loaded for yes/no scoring is never re-checked."""
    if not documents:
        return []
    if model not in _rerank_cache and (
        model in _cross_encoder_cache or _cross_encoder.is_cross_encoder(model)
    ):
        encoder = get_or_load(_cross_encoder_cache, model, _load_cross_encoder)
        logits = _cross_encoder.score_logits(encoder, query, documents)
        return [_cross_encoder.calibrate(x, encoder.bias, encoder.temperature) for x in logits]
    return _rerank_yes_no(query, documents, model)


def _rerank_yes_no(query: str, documents: list[str], model: str) -> list[float]:
    import mlx.core as mx

    rerank_model, tokenizer = get_or_load(_rerank_cache, model, _load_yes_no)

    true_token_id = tokenizer.convert_tokens_to_ids("yes")
    false_token_id = tokenizer.convert_tokens_to_ids("no")

    scores = []
    for document in documents:
        prompt = build_rerank_prompt(query, document)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        last_logits = _last_position_logits(rerank_model, mx.array([token_ids]))
        true_logit = float(last_logits[true_token_id].item())
        false_logit = float(last_logits[false_token_id].item())
        scores.append(score_from_logits(true_logit, false_logit))

    return scores
