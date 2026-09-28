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
