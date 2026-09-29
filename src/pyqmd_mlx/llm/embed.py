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
    outputs = embed_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.text_embeds.tolist()
