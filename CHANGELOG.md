# CHANGELOG


## v0.3.0 (2026-09-28)

### ✨ Features

- **cli**: The pyqmd command line ([#3](https://github.com/after2400/pyqmd/pull/3),
  [`c52aaa8`](https://github.com/after2400/pyqmd/commit/c52aaa881b148a8a75ee25025e733b61b3ec8031))

Collections (add/list/show/remove/rename/include/exclude/update-cmd), embed, update, cleanup, pull, status, context, search/vsearch/query, get/multi-get/ls, output formats (cli/json/csv/md/xml/files), line numbers, full paths and --version.


## v0.2.0 (2026-09-28)

### ✨ Features

- **store**: Hybrid SQLite search with metadata filtering
  ([#2](https://github.com/after2400/pyqmd/pull/2),
  [`93a6e4a`](https://github.com/after2400/pyqmd/commit/93a6e4a91c3cbf0995c90ea61df9069f65e22688))

- FTS5 (BM25) full-text search and sqlite-vec vector search
- Hybrid query: expansion, reciprocal rank fusion and LLM reranking
- Front-matter metadata extraction with a recursive filter language
  (and/or/not, comparisons, in/nin/all, exists)
- AST-aware chunking for code files (Python, TypeScript, JavaScript,
  Go, Rust) and folder context

### 🏗 Chores

- **release**: V0.2.0 [skip ci]
  ([`da947fa`](https://github.com/after2400/pyqmd/commit/da947fa352a0c97f4a5fefe5a24cc9aba5792a45))


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

### 🏗 Chores

- **release**: V0.1.0 [skip ci]
  ([`f5d57e6`](https://github.com/after2400/pyqmd/commit/f5d57e6a6055730590f9357a31e167bcbd50ce3f))


## v0.0.1 (2026-09-28)

- Initial Release
