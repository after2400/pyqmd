"""embed(): text embeddings via mlx_embeddings."""

from ._cache import get_or_load
from ._constants import DEFAULT_EMBED_MODEL
from ._hub import load_quietly_if_cached
from ._prompts import format_doc_for_embedding, format_query_for_embedding

_embed_cache: dict = {}


def _load(model_id: str):
    # Deferred: mlx_embeddings drags in mlx_vlm -> transformers (~700ms) --
    # a module-level import here would tax every CLI command that touches
    # pyqmd_mlx.store.Store (i.e. all of them, via pyqmd_mlx.llm's re-export), including
    # search/status/ls, which never call embed() at all.
    from mlx_embeddings import load as _load_embed_model

    return load_quietly_if_cached(model_id, _load_embed_model)


def embed(
    texts: list[str], model: str = DEFAULT_EMBED_MODEL, kind: str = "query"
) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input text.

    `kind` selects the formatter: "query" (default, preserves prior behavior)
    applies the query-prefix formatting; "document" applies the document
    formatting instead. embeddinggemma is asymmetric by training, so documents
    must not receive the query prefix.
    """
    if kind not in ("query", "document"):
        raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")
    embed_model, tokenizer = get_or_load(_embed_cache, model, _load)

    if kind == "document":
        formatted = [format_doc_for_embedding(text, model) for text in texts]
    else:
        formatted = [format_query_for_embedding(text, model) for text in texts]
    inputs = tokenizer.batch_encode_plus(
        formatted, return_tensors="mlx", padding=True, truncation=True, max_length=512
    )
    outputs = embed_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.text_embeds.tolist()
