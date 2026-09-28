# CHANGELOG


## v0.6.2 (2026-09-28)

### 📖 Documentation

- **specs**: Publish the design specs ([#9](https://github.com/after2400/pyqmd/pull/9),
  [`15c8344`](https://github.com/after2400/pyqmd/commit/15c8344ef1c6355f7fa1af067b72ab606e05c909))

The design history behind each subsystem, including spike results and rejected alternatives. Superseded designs are kept, marked with their successor.


## v0.6.1 (2026-09-28)

### ✅ Testing

- **parity**: Golden-snapshot parity suite against Node qmd
  ([#7](https://github.com/after2400/pyqmd/pull/7),
  [`1ae9883`](https://github.com/after2400/pyqmd/commit/1ae98832a1e3eadf10482a428eddcbb8d3705a7b))

Golden-snapshot comparisons of pyqmd's CLI and MCP output against a frozen Node qmd reference on the public SciFact dataset, plus a quality baseline and performance benchmark.

### 🏗 Chores

- **release**: V0.6.1 [skip ci]
  ([`e934639`](https://github.com/after2400/pyqmd/commit/e93463998511941c2a118b2ad2ba0b8f5be23197))

- **scripts**: Development and model-maintenance scripts
  ([#8](https://github.com/after2400/pyqmd/pull/8),
  [`fdf84e7`](https://github.com/after2400/pyqmd/commit/fdf84e7d1fb08633a6d640496d2f389bcb38844e))

Expansion-model GGUF conversion, rerank fixture scoring, SciFact corpus preparation, query replay and store validation.


## v0.6.0 (2026-09-28)

### ✨ Features

- **skills**: Bundled agent skills and skill commands
  ([#6](https://github.com/after2400/pyqmd/pull/6),
  [`1937466`](https://github.com/after2400/pyqmd/commit/193746693a377414e26e9138ebca6527bfacbdcd))

pyqmd, pyqmd-librarian, pyqmd-researcher and pyqmd-bench skills, with `pyqmd skill list|show|install` and `pyqmd skills list|get|path`.

### 🏗 Chores

- **release**: V0.6.0 [skip ci]
  ([`eb64e32`](https://github.com/after2400/pyqmd/commit/eb64e32dcc5e407900e95f54b897ca593c165734))


## v0.5.0 (2026-09-28)

### ✨ Features

- **bench**: Search-quality benchmarks against your own corpus
  ([#5](https://github.com/after2400/pyqmd/pull/5),
  [`8c5a227`](https://github.com/after2400/pyqmd/commit/8c5a22719b42e55e48ee5e2a6030a9090513522b))

`pyqmd bench <fixture.json>` scores bm25, vector, hybrid and full (reranked) retrieval with Recall@k, MRR and nDCG@k; --samples N reports the spread across expansion seeds.

### 🏗 Chores

- **release**: V0.5.0 [skip ci]
  ([`35e3180`](https://github.com/after2400/pyqmd/commit/35e31809d1c1b69e8f98133814fda2f299590ff4))


## v0.4.0 (2026-09-28)

### ✨ Features

- **mcp**: Serve pyqmd over MCP (stdio and HTTP) ([#4](https://github.com/after2400/pyqmd/pull/4),
  [`04c3538`](https://github.com/after2400/pyqmd/commit/04c3538d839b8c43e0d9457ada7dcd79d84b4dcd))

`pyqmd mcp` serves query, get, multi_get and status tools plus a qmd:// document resource to MCP clients, over stdio or Streamable HTTP (with an origin guard).

### 🏗 Chores

- **release**: V0.4.0 [skip ci]
  ([`129f8f3`](https://github.com/after2400/pyqmd/commit/129f8f3c19b5da16f3cf68011a1cff21d2cba83a))


## v0.3.0 (2026-09-28)

### ✨ Features

- **cli**: The pyqmd command line ([#3](https://github.com/after2400/pyqmd/pull/3),
  [`c52aaa8`](https://github.com/after2400/pyqmd/commit/c52aaa881b148a8a75ee25025e733b61b3ec8031))

Collections (add/list/show/remove/rename/include/exclude/update-cmd), embed, update, cleanup, pull, status, context, search/vsearch/query, get/multi-get/ls, output formats (cli/json/csv/md/xml/files), line numbers, full paths and --version.

### 🏗 Chores

- **release**: V0.3.0 [skip ci]
  ([`108d5d4`](https://github.com/after2400/pyqmd/commit/108d5d4dac186004b92e8f6ee5053629c79da6f1))


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
