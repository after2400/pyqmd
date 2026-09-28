# CHANGELOG


## v0.1.0 (2026-09-28)

### ✨ Features

- **llm**: In-process MLX embeddings, reranking and query expansion
  ([#1](https://github.com/after2400/pyqmd/pull/1),
  [`bf3ecc4`](https://github.com/after2400/pyqmd/commit/bf3ecc4de182d8fc5b2282080626a7df13ebe782))

- Embeddings through mlx-embeddings, reranking and query expansion
  through mlx-lm, all in process: no subprocess or HTTP bridge
- Query expansion with a fine-tuned 1.7B model (mixed 4/6-bit MLX
  conversion), grammar-constrained output and per-query seeded
  sampling; overridable with PYQMD_EXPAND_MODEL
- Models download from the Hugging Face Hub on first use

Co-authored-by: Chuck Lunskis <17775963+after2400@users.noreply.github.com>

Co-authored-by: Claude Haiku 4.5 <noreply@anthropic.com>


## v0.0.1 (2026-09-28)

- Initial Release
