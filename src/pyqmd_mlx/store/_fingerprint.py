"""Embedding fingerprint, ported from store.ts's getEmbeddingFingerprint
(store.ts:141): a short hash of everything that shapes a stored vector's
input (model id, the formatted query/document probes, chunk sizes).
Stored on every content_vectors row; a row without the current value
counts as pending, so any change to the embedding input re-embeds on the
next `pyqmd embed` without --force. Lives in store, not llm, because it
combines llm's formatters with store's chunk constants (llm importing
store would be circular)."""

import hashlib

from pyqmd_mlx.llm._prompts import format_doc_for_embedding, format_query_for_embedding

from ._chunking import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

PROBE_QUERY = "__qmd_embedding_query_probe__"
PROBE_TITLE = "__qmd_embedding_title_probe__"
PROBE_DOC = "__qmd_embedding_document_probe__"


def embedding_fingerprint(model: str) -> str:
    significant = "\n".join(
        [
            f"model:{model}",
            f"query:{format_query_for_embedding(PROBE_QUERY, model)}",
            f"doc:{format_doc_for_embedding(PROBE_DOC, model, PROBE_TITLE)}",
            f"chunk_tokens:{CHUNK_SIZE_TOKENS}",
            f"chunk_overlap_tokens:{CHUNK_OVERLAP_TOKENS}",
        ]
    )
    return hashlib.sha256(significant.encode("utf-8")).hexdigest()[:6]
