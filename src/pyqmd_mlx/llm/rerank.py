"""rerank(): Qwen3-Reranker yes/no-logit scoring, run directly on mlx-lm.

mlx_embeddings cannot do this for the plain-text Qwen3-Reranker family (see
the design spec's "Research finding" section) -- mlx-lm keeps the LM head
intact, so we implement the scoring ourselves.
"""

from ._cache import get_or_load
from ._constants import DEFAULT_RERANK_MODEL
from ._prompts import build_rerank_prompt, score_from_logits

_rerank_cache: dict = {}


def _load(model_id: str):
    # Deferred -- see embed.py's _load for why: a module-level mlx_lm import
    # would tax every CLI command via pyqmd_mlx.store.Store, not just rerank().
    from mlx_lm import load as _load_lm_model

    return _load_lm_model(model_id)


def _last_position_logits(rerank_model, inputs):
    """The next-token logits at the prompt's last position only.

    Calling the model directly applies the output layer to every position:
    a vocab-sized (~152k) row per prompt token, ~1 GB for a long candidate,
    of which scoring reads two values. Running the transformer body and
    applying the output layer to just the last hidden state gives the same
    logits there, bit for bit."""
    hidden = rerank_model.model(inputs)[:, -1:, :]
    if rerank_model.args.tie_word_embeddings:
        return rerank_model.model.embed_tokens.as_linear(hidden)[0, -1, :]
    return rerank_model.lm_head(hidden)[0, -1, :]


def rerank(query: str, documents: list[str], model: str = DEFAULT_RERANK_MODEL) -> list[float]:
    """Score each document's relevance to query, in the same order as documents."""
    import mlx.core as mx

    rerank_model, tokenizer = get_or_load(_rerank_cache, model, _load)

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
