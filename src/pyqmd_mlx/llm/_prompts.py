"""Prompt/text formatting for embedding and reranking, ported from src/llm.ts."""

import math
import re

_QWEN_EMBED_RE = re.compile(r"qwen.*embed", re.IGNORECASE)
_EMBED_QWEN_RE = re.compile(r"embed.*qwen", re.IGNORECASE)


def is_qwen3_embedding_model(model_id: str) -> bool:
    """Detect if a model id uses the Qwen3-Embedding format (ported from isQwen3EmbeddingModel)."""
    return bool(_QWEN_EMBED_RE.search(model_id)) or bool(_EMBED_QWEN_RE.search(model_id))


def format_query_for_embedding(query: str, model_id: str) -> str:
    """Format a query for embedding (ported from formatQueryForEmbedding).

    Uses nomic-style task prefix for embeddinggemma (default), or Qwen3-Embedding
    instruct format when a Qwen embedding model is active.
    """
    if is_qwen3_embedding_model(model_id):
        return f"Instruct: Retrieve relevant documents for the given query\nQuery: {query}"
    return f"task: search result | query: {query}"


def format_doc_for_embedding(text: str, model_id: str, title: str | None = None) -> str:
    """Format a document for embedding (ported from formatDocForEmbedding).

    Uses nomic-style title/text fields for embeddinggemma (default), or raw text
    (optionally prefixed with a title line) when a Qwen embedding model is active.
    """
    if is_qwen3_embedding_model(model_id):
        return f"{title}\n{text}" if title else text
    return f"title: {title or 'none'} | text: {text}"


RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def build_rerank_prompt(
    query: str, document: str, instruction: str = DEFAULT_RERANK_INSTRUCTION
) -> str:
    """Build the Qwen3-Reranker prompt (ported from the official reference implementation)."""
    pair = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return (
        f"<|im_start|>system\n{RERANK_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{pair}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def score_from_logits(true_logit: float, false_logit: float) -> float:
    """Qwen3-Reranker's yes/no scoring: sigmoid(true_logit - false_logit)."""
    return 1.0 / (1.0 + math.exp(-(true_logit - false_logit)))
