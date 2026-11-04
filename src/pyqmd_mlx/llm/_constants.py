"""Default model ids for each pyqmd_mlx.llm role."""

DEFAULT_EMBED_MODEL = "mlx-community/embeddinggemma-300m-8bit"
DEFAULT_RERANK_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"

# Mixed 4/6-bit MLX conversion (llama.cpp Q4_K_M layout) of
# tobil/qmd-query-expansion-1.7B-gguf (the fine-tuned expansion model Node
# qmd ships), built by scripts/convert_expand_gguf.py and re-hosted on the Hub
# because no upstream MLX build exists -- see
# docs/specs/2026-09-24-expand-model-gguf-weights-design.md and
# docs/specs/2026-09-25-expand-model-requantize-design.md.
DEFAULT_EXPAND_MODEL = "after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6"

# Overrides DEFAULT_EXPAND_MODEL with a Hub repo id or a local MLX model dir.
EXPAND_MODEL_ENV_VAR = "PYQMD_EXPAND_MODEL"

# Overrides DEFAULT_RERANK_MODEL with a Hub repo id or a local MLX model dir.
RERANK_MODEL_ENV_VAR = "PYQMD_RERANK_MODEL"

# IBM's Apache-2.0 ModernBERT cross-encoder, the tested alternative reranker
# (English only). Select it with PYQMD_RERANK_MODEL.
GRANITE_RERANK_MODEL = "ibm-granite/granite-embedding-reranker-english-r2"

# Cross-encoder rerankers score a (query, document) pair in one sequence; the
# document is truncated to fit this many tokens (the checkpoint allows 8192).
CROSS_ENCODER_MAX_TOKENS = 2048

# Hub repo id -> (bias b, temperature t) mapping a cross-encoder's raw logit to
# a 0..1 score, sigmoid((logit - b) / t). Unlisted checkpoints use (0.0, 1.0).
RERANK_CALIBRATION: dict[str, tuple[float, float]] = {}

# Embedding context in tokens, Node's EMBED_CONTEXT_SIZE (llm.ts). embed()
# truncates at min(this, the tokenizer's model_max_length), mirroring
# resolveEmbedTokenLimit(); both are 2048 for the default model.
EMBED_CONTEXT_TOKENS = 2048
