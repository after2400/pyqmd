# CHANGELOG


## v0.8.1 (2026-10-02)

### 🐞 Bug Fixes

- **store**: Correct stale _indexing docstring on metadata sync
  ([#19](https://github.com/after2400/pyqmd/pull/19),
  [`e8fba6d`](https://github.com/after2400/pyqmd/commit/e8fba6d65a6de97a169409bbca5dab43e5f85520))

The module docstring claimed syncDocumentMetadata was deliberately not ported, but scan_and_register_collection syncs frontmatter qmd.metadata for every scanned document on both `collection add` and `update`, re-extracting when content changed or the stored extraction is stale.


## v0.8.0 (2026-10-01)

### ✨ Features

- **config**: Support Python 3.11 through 3.14 ([#18](https://github.com/after2400/pyqmd/pull/18),
  [`db2a2a8`](https://github.com/after2400/pyqmd/commit/db2a2a8f571c3b5e2ca71ab90332f0072354662b))

Lower requires-python to 3.11 and list 3.11-3.14 as supported. The only 3.12-only code was an f-string in update that reused its outer quote; ruff now targets py311 so lint catches the next one. On 3.11 the lock resolves numpy 2.4 and scipy 1.17, whose newer releases dropped 3.11.

### ✅ Testing

- **parity**: Scrub repo paths from structural and MCP raw captures
  ([#17](https://github.com/after2400/pyqmd/pull/17),
  [`87c9243`](https://github.com/after2400/pyqmd/commit/87c9243690fdec03bf4f0954d7d06a7f2918fb0f))

The structural (cli_raw/) and MCP (mcp_raw/) phases wrote Node's raw output as-is, so a capture committed absolute local paths (the corpus under data/, the isolated index under node_ref/) that had to be hand-scrubbed afterwards. Both now replace the pyqmd checkout root with <pyqmd-repo> and the Node checkout root with <qmd-repo> before writing, longest first. From a git worktree, the main checkout (which holds the gitignored data/) also maps to <pyqmd-repo>.

### 🔁 Continuous Integration

- Test and smoke-test the wheel on Python 3.11 through 3.14
  ([#18](https://github.com/after2400/pyqmd/pull/18),
  [`db2a2a8`](https://github.com/after2400/pyqmd/commit/db2a2a8f571c3b5e2ca71ab90332f0072354662b))

The test job runs every supported Python on Ubuntu and macOS. build.yml splits into one build job, which uploads dist once, and a per-version smoke job that installs the wheel with the newest dependencies.

### 📖 Documentation

- **specs**: Publish the Python 3.11–3.14 support spec
  ([#18](https://github.com/after2400/pyqmd/pull/18),
  [`db2a2a8`](https://github.com/after2400/pyqmd/commit/db2a2a8f571c3b5e2ca71ab90332f0072354662b))


## v0.7.1 (2026-09-30)

### 🐞 Bug Fixes

- **cli**: Warn on vsearch/query when embeddings are pending or stale
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Like Node, vsearch and query now print a warning on stderr when 10% or more of documents need embeddings (a tip below that), and a tip when the index hasn't been updated for 14 days or more. After upgrading, this is how an existing index reports that it needs 'pyqmd embed'.

- **config**: Pin mlx-embeddings to exactly 0.1.0
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

pyqmd's embed() now runs embeddinggemma through its own copy of mlx-embeddings 0.1.0's forward pass, to fix that release's padding mask. A newer release could change what that copy relies on, so the version is pinned until a new release has been checked against it.

- **llm**: Embed chunks with their document title and up to 2048 tokens
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Every chunk was embedded as 'title: none | text: ...' and cut at 512 tokens, so vector search never saw the rest of a full chunk. Chunks now carry their document's title and are cut at 2048 tokens, as in Node. Existing vectors are re-embedded by 'pyqmd embed' after upgrading.

- **llm**: Embed the same input Node does ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

- **llm**: Mask padding when embedding a batch of chunks
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

mlx-embeddings 0.1.0 casts embeddinggemma's padding mask to the dtype of its embedding weight, which in the 8-bit model is packed uint32, so the mask's -inf became 0 and padding was never masked. Every chunk of a multi-chunk document except the longest was embedded while attending to padding (cosine 0.92 against the same chunk embedded alone). pyqmd now builds the mask in the model's activation dtype. Queries, embedded one at a time, were unaffected.

- **store**: Chunk for embedding at Node's embed-time size
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Node embeds 2700-character chunks (it estimates 3 characters per token when chunking for embedding) and uses 3600-character chunks only to pick each result's best chunk at query time. pyqmd used 3600 for both, so different text landed in each vector. Code-fence rules stay pyqmd's and are now listed as deliberate differences.

- **store**: Extract titles by file type, as Node does
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

The markdown heading rule ran on every file, so a Python file starting with '# setup helpers' was titled 'setup helpers'. Titles now follow Node's per-extension rules: markdown headings for .md, #+TITLE or the first heading for .org, else the file name (stripped from the last dot in the path, as Node does). The patterns use JS's line-break classes, so \r and U+2028 end a line as they do in Node. Run 'pyqmd update' after upgrading to correct stored titles.

- **store**: Fingerprint embeddings so format changes re-embed
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Each stored vector now records a fingerprint of the embedding input format (Node's getEmbeddingFingerprint). Vectors from an older format, including every vector in an existing pyqmd index, count as pending: status reports them, and 'pyqmd embed' re-embeds them without --force. After upgrading, run 'pyqmd update' and then 'pyqmd embed'.

- **store**: Refresh stored titles when only the title changes
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

update left a document's old title in place when its content was unchanged, so a changed title rule never reached existing documents. It now updates the title, as Node does, and counts the file as updated.

### ✅ Testing

- **parity**: Add ConditionalQA as a second parity corpus
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

A prep script converts ConditionalQA's gov.uk pages to markdown with real section headings, and picks one question per dev page (59), with its scenario as the query's intent. Scifact documents are one short abstract each, so it barely sees long-document or chunking behaviour.

- **parity**: Capture Node's embedding inputs for fixture documents
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

A new --phase inputs runs a small bun script against the pinned Node checkout and records, for each fixture, the title, the embed-time chunks and the exact strings Node embeds, plus Node's embedding fingerprint. The fixtures each hit one title or chunking rule. No pre-commit hook or ruff run touches them, so they reach both sides byte for byte.

- **parity**: Compare embedding inputs with Node's, string for string
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Checks pyqmd's per-chunk embedding inputs (title, text, and chunk boundaries), query inputs and fingerprint against the inputs Node produced for the same fixture documents.

- **parity**: Pass query intents through the quality test and Node capture
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Queries may carry an intent; both sides pass it to query. The Node capture also keeps one pass's rankings so the quality test can report how far pyqmd's top 10 overlaps Node's.

- **parity**: Pin Node's ConditionalQA and embedding-input snapshots
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Node's structural, CLI flow, MCP and 30-pass quality snapshots for the ConditionalQA profile, captured at the same pinned qmd commit as SciFact's, plus Node's exact embedding inputs for the fixture documents. Local repo paths in the reference-only raw outputs are replaced with <pyqmd-repo>, as in SciFact's.

- **tests**: Find the Node qmd checkout from git worktrees
  ([#15](https://github.com/after2400/pyqmd/pull/15),
  [`24e658f`](https://github.com/after2400/pyqmd/commit/24e658f91ab82f556bfc8aa85fe07d102928226c))

The qmd_repo_root fixture looked for qmd as a sibling of this checkout's own root (parents[3]), which from a worktree under .claude/worktrees/ is .claude/worktrees/qmd, so all five Node CLI tests skipped on every worktree run. Resolve the main checkout via `git rev-parse
--git-common-dir` and look for qmd next to that instead; the main checkout and CI (no qmd checkout) behave as before.

### 📖 Documentation

- Record embedding input parity in COMMAND_STATUS
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

Also corrects CLAUDE.md's query -n default, which is 20, like search and vsearch.

- **parity**: Credit SciFact's sources and licences
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))

The committed SciFact snapshots quote abstracts and claims, as the ConditionalQA ones quote gov.uk pages, but only ConditionalQA was credited. Also note both built-in profiles' snapshots are committed.

- **specs**: Publish the embedding input parity spec
  ([#16](https://github.com/after2400/pyqmd/pull/16),
  [`6d9e622`](https://github.com/after2400/pyqmd/commit/6d9e62259c7db4ecccb2bdd50d0e6646a01cdf34))


## v0.7.0 (2026-09-29)

### ✨ Features

- **config**: Publish releases to PyPI as pyqmd-mlx
  ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

Every release PSR cuts is now built, checked and uploaded to PyPI via trusted publishing (no stored token). A reusable build workflow builds on Apple Silicon, runs twine check and smoke-tests the wheel with the latest dependencies; CI runs it on every PR. docs commits no longer release.

### 🐞 Bug Fixes

- **config**: Render multi-commit squash merges cleanly in the changelog
  ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

PSR splits a squash body into one entry per branch commit, but also parses the squash title, so the title's commit was listed twice; and GitHub's "---------" separator before the co-author trailers leaked into the last entry. Keep one entry per summary line (preferring the one with a body) and drop the separator paragraph.

- **llm**: Hide the Hugging Face progress bar for cached models
  ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

Model loads printed Hugging Face's "Fetching N files" bar on every run, even when every file was already cached. It's now hidden when the model is a local directory or its config.json is in the HF cache; a real first download still shows progress.

### 📖 Documentation

- Install from PyPI and state measured query timings
  ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

The README installs pyqmd-mlx from PyPI instead of a pinned git tag that went stale every release, and the performance note gives measured per-stage timings instead of "several seconds cold".

- **specs**: Publish the PyPI publishing design ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

### 🏗 Chores

- **config**: Add a review commit type for PR review follow-ups
  ([#14](https://github.com/after2400/pyqmd/pull/14),
  [`1251e00`](https://github.com/after2400/pyqmd/commit/1251e008a00936051c4b27e7fc53d0cb79b44e25))

Changes a reviewer asks for on an open PR get their own type, review, which like doh never bumps the version or reaches the changelog: a squash merge keeps each branch commit as a separate changelog entry. doh stays for fixing your own silly mistakes.


## v0.6.6 (2026-09-29)

### ⚡️ Performance Improvements

- **llm**: Score rerank candidates from the last position only
  ([#13](https://github.com/after2400/pyqmd/pull/13),
  [`8b53138`](https://github.com/after2400/pyqmd/commit/8b53138a1e9aadbf5e718ec6ee81ae2b751e68d0))

The reranker called the full model, which applies the output layer (~152k vocab) to every prompt position, then read two logits at the last one. Run the transformer body and apply the output layer to the last hidden state only: identical scores, about half the rerank time.


## v0.6.5 (2026-09-28)

### 📖 Documentation

- **specs**: Publish the multi-agent skills design
  ([#12](https://github.com/after2400/pyqmd/pull/12),
  [`b754705`](https://github.com/after2400/pyqmd/commit/b754705ccd44946746c3b34282497be576ac2b16))

The design behind the pyqmd-librarian, pyqmd-researcher and pyqmd-bench skills and the bundled pyqmd skill fix, plus the shared HTTP server pattern its mcp-setup reference documents.


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
