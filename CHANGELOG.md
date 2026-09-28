# CHANGELOG


## v0.6.4 (2026-09-28)

### 🐞 Bug Fixes

- **config**: Keep release commits and trailers out of the changelog
  ([#11](https://github.com/after2400/pyqmd/pull/11),
  [`caf62b3`](https://github.com/after2400/pyqmd/commit/caf62b389b8d00095eab6f4f377ccbef31797dcc))

- Skip chore(release) commits: PSR's exclude_commit_patterns keeps any
  commit that bumps the version, and chore is a patch tag
- Match co-author and sign-off trailers case-insensitively, so GitHub's
  "Co-authored-by" lines from squash merges are dropped too
- Lint PR titles with commitlint: a multi-commit squash merge takes the
  PR title as its subject, which the commit-msg hook never sees


## v0.6.3 (2026-09-28)

### 📖 Documentation

- The README, agent guide and command parity status
  ([#10](https://github.com/after2400/pyqmd/pull/10),
  [`ceb075b`](https://github.com/after2400/pyqmd/commit/ceb075bc0fade9e3b5b39365b7f428b1d7496fb0))

Install from a release tag, beta feedback note, the CLAUDE.md/AGENTS.md agent guide and COMMAND_STATUS.md (command-by-command parity with Node qmd).


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


## v0.5.0 (2026-09-28)

### ✨ Features

- **bench**: Search-quality benchmarks against your own corpus
  ([#5](https://github.com/after2400/pyqmd/pull/5),
  [`8c5a227`](https://github.com/after2400/pyqmd/commit/8c5a22719b42e55e48ee5e2a6030a9090513522b))

`pyqmd bench <fixture.json>` scores bm25, vector, hybrid and full (reranked) retrieval with Recall@k, MRR and nDCG@k; --samples N reports the spread across expansion seeds.


## v0.4.0 (2026-09-28)

### ✨ Features

- **mcp**: Serve pyqmd over MCP (stdio and HTTP) ([#4](https://github.com/after2400/pyqmd/pull/4),
  [`04c3538`](https://github.com/after2400/pyqmd/commit/04c3538d839b8c43e0d9457ada7dcd79d84b4dcd))

`pyqmd mcp` serves query, get, multi_get and status tools plus a qmd:// document resource to MCP clients, over stdio or Streamable HTTP (with an origin guard).


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


## v0.0.1 (2026-09-28)

- Initial Release
