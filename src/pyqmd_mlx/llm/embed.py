"""embed(): text embeddings via mlx_embeddings."""

from ._cache import get_or_load
from ._constants import DEFAULT_EMBED_MODEL, EMBED_CONTEXT_TOKENS
from ._hub import load_quietly_if_cached
from ._prompts import format_embedding_inputs

_embed_cache: dict = {}


def _load(model_id: str):
    # Deferred: mlx_embeddings drags in mlx_vlm -> transformers (~700ms) --
    # a module-level import here would tax every CLI command that touches
    # pyqmd_mlx.store.Store (i.e. all of them, via pyqmd_mlx.llm's re-export), including
    # search/status/ls, which never call embed() at all.
    from mlx_embeddings import load as _load_embed_model

    return load_quietly_if_cached(model_id, _load_embed_model)


def _activation_dtype(embed_tokens):
    """The dtype a model's activations run in. A quantized embedding packs
    its weight into uint32; its scales carry the activation dtype."""
    scales = getattr(embed_tokens, "scales", None)
    return scales.dtype if scales is not None else embed_tokens.weight.dtype


def _additive_padding_mask(attention_mask, dtype):
    """(batch, 1, query, key) additive mask: 0 where the key is a real
    token, -inf where it is padding. Every query sees every real token
    (embeddinggemma attends bidirectionally)."""
    import mlx.core as mx

    batch, length = attention_mask.shape
    keys = attention_mask[:, None, None, :].astype(mx.bool_)
    mask = mx.where(keys, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))
    return mx.broadcast_to(mask, (batch, 1, length, length))


def _gemma3_text_embeds(embed_model, input_ids, attention_mask):
    """mlx_embeddings' gemma3_text Model.__call__, but with the padding mask
    built in the activation dtype. mlx-embeddings 0.1.0 casts it to
    embed_tokens.weight.dtype, the packed uint32 of a quantized model, so
    -inf became 0 and padding was never masked: in a padded batch, every
    text but the longest attended to padding tokens."""
    from mlx_embeddings.models.base import mean_pooling, normalize_embeddings

    dtype = _activation_dtype(embed_model.model.embed_tokens)
    hidden = embed_model.model(input_ids, _additive_padding_mask(attention_mask, dtype))
    pooled = mean_pooling(hidden, attention_mask)
    for dense in embed_model.dense:
        pooled = dense(pooled)
    return normalize_embeddings(pooled)


def embed(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    kind: str = "query",
    title: str | None = None,
) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input text.

    `kind` selects the formatter: "query" (default) applies the query
    prefix; "document" the document format, with `title` (the document's
    title, shared by every chunk in the batch; "none" when absent, as in
    Node). embeddinggemma is asymmetric by training, so documents must not
    receive the query prefix. Inputs are truncated at
    min(EMBED_CONTEXT_TOKENS, the tokenizer's model_max_length), Node's
    embedding context.
    """
    formatted = format_embedding_inputs(texts, model, kind, title)
    embed_model, tokenizer = get_or_load(_embed_cache, model, _load)
    max_length = min(EMBED_CONTEXT_TOKENS, tokenizer.model_max_length)
    inputs = tokenizer.batch_encode_plus(
        formatted, return_tensors="mlx", padding=True, truncation=True, max_length=max_length
    )
    if getattr(embed_model, "model_type", None) == "gemma3_text":
        vectors = _gemma3_text_embeds(embed_model, inputs["input_ids"], inputs["attention_mask"])
    else:
        vectors = embed_model(
            inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).text_embeds
    return vectors.tolist()
