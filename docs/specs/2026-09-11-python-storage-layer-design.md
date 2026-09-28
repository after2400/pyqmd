# Sub-project #2: Python storage layer (design)

**Status:** Implemented (2026-09-11)

## Purpose

Port qmd's storage/search engine (`src/store.ts`, ~6256 lines) to Python, as
sub-project #2 of the Node→Python/MLX rewrite (see
`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`). This is a
mechanical port of a proven design, not a redesign — the schema and search
algorithm are already validated in production; the job here is translating
them faithfully into Python, onto SQLite FTS5 + `sqlite-vec`, built on top of
sub-project #1's already-validated `qmd_llm` (embed/rerank/query-expansion via
MLX).

Sub-project #1 proved the harder, higher-uncertainty question: can MLX match
the current Node/GGUF stack on embedding/reranking/expansion quality? Yes
(see that spec's validation results). This sub-project's risk is different —
not model quality, but whether SQLite/FTS5/`sqlite-vec` work correctly from
Python and whether the hybrid-query _algorithm_ survives translation intact.

## Scope: a deliberate MVP, not full parity

`store.ts` does far more than search: schema/migrations, CJK-normalization
FTS rebuilds, legacy-document path migration, lazy schema-column repair,
metadata filtering, cache/cleanup/vacuum utilities, comma-list glob
resolution, docid lookup, snippet extraction with intent-weighted scoring,
and more. Porting all of it in one pass would make this sub-project as large
and uncertain as the whole rewrite. Scope here is deliberately narrower: the
core storage + hybrid search pipeline, proven end-to-end, with secondary
features explicitly deferred (see "Deferred" below) rather than silently
dropped.

## Foundational decision: clean-slate database

The Python storage layer does **not** need to read or write the same SQLite
file the Node version uses. This is a clean-slate index format — no
Node/Python interop, no shared `~/.cache/qmd/index.sqlite`. This was an
explicit choice (not a default), made because the rewrite itself is already a
full replacement, not a bridge, and clean-slate gives real freedom on schema
details without needing byte-for-byte migration compatibility. Two important
things this decision does **not** mean, corrected during brainstorming after
being initially over-claimed:

- It does not mean "no schema versioning is ever needed." The Python store's
  _own_ schema will evolve after it has real user data, independent of
  Node's history entirely — that problem is not solved by clean-slate, and
  isn't deferred here either (see "Schema versioning" below).
- It does not mean every piece of `store.ts`'s legacy-handling code is
  automatically obsolete. Each such function was checked individually against
  whether it addresses a Node-specific historical event (moot for Python) or
  a generic future-schema-evolution concern (not moot) — see the per-item
  notes below rather than a blanket claim.

## Package structure

`python/` currently holds `qmd_llm` (sub-project #1) as a standalone package.
As part of this sub-project, restructure to a single `qmd` package:

- `qmd.llm` — sub-project #1's code, renamed in place (`qmd_llm` →
  `qmd.llm`), no behavior change. Existing tests/scripts' imports updated.
- `qmd.store` — this sub-project's new code.
- One `pyproject.toml`, one dependency set, one test suite root.

Rejected alternative: keeping `qmd_llm` and a new `qmd_store` as separate
packages. `qmd_llm`'s original isolation was about validating a
high-uncertainty technical bet independently before committing further — that
uncertainty is resolved now, and `qmd.store` will always depend on `qmd.llm`
(hybrid query calls `embed`/`rerank`/`expand_query` directly), so there is no
real scenario where they'd be used or versioned independently. Merging avoids
maintaining two dependency graphs for one product.

## Architecture: `Store` class, logic in methods

`store.ts` actually uses a hybrid pattern: ~200 plain exported functions
(each taking `db` explicitly) do the real work, and `createStore(dbPath)` is
a factory returning a `Store` object whose fields are those same functions
pre-bound to a specific `db` via closures — a thin facade over a functional
core. That split exists in `store.ts` for a specific reason: some consumers
(the Node SDK, one-off scripts, CLI tooling) call the plain functions
directly without constructing a full `Store`.

For the Python port, that reason doesn't apply — there is no anticipated
consumer that needs storage operations without a `Store` in hand. Combined
with SQLite's cheap connection/object construction and Python's dynamic
typing (swapping a method for a test fake is trivial, no mocking-framework
tax), the facade-plus-duplicate-function split isn't worth its indirection
cost here. **Decision: `Store` is a plain class whose methods contain the
actual logic** (`store.search_fts(...)`, `store.insert_document(...)`, etc.),
constructed via `Store(db_path)`, which opens the connection and initializes
the schema.

Exception: logic that never touched `db` in `store.ts` — reciprocal rank
fusion scoring, chunking/break-point/code-fence detection — stays as
standalone module-level functions in Python too. This isn't a re-litigation
of the class-vs-functions decision; it's the same "don't wrap pure logic in
unnecessary I/O-shaped plumbing" principle either style should follow.

## Schema

Tables, ported from `store.ts`'s proven design with migration/legacy-compat
code dropped per the per-item review below:

- **`content`** (`hash` PK, `doc`, `created_at`) — content-addressable
  storage; the untouched source of truth for file bodies.
- **`documents`** (`id`, `collection`, `path`, `title`, `hash`→`content`,
  `created_at`, `modified_at`, `active`, `UNIQUE(collection, path)`) — maps
  collection-relative paths to content, soft-deleted via `active`.
- **`collections`** (`name` PK, `path`, `pattern`, `ignore_patterns`,
  `include_by_default`, `update_command`, `context`) — the single source of
  truth for collection config (see "Collections: DB-only" below).
- **`content_vectors`** (`hash`, `seq`, `pos`, `model`, `embed_fingerprint`,
  `total_chunks`, `embedded_at`) — bookkeeping for which content/chunks have
  been embedded with which model, so a model change is detectable. Actual
  vectors live in a `sqlite-vec` virtual table keyed to match.
- **`llm_cache`** (`hash` PK, `result`, `created_at`) — caches
  `expand_query`/`rerank` calls by input hash.
- **`documents_fts`** — FTS5 virtual table over path/title/content.

### Collections: DB-only, no YAML

`store.ts` has an unresolved split: a comment says collections are "now
managed in `~/.config/qmd/index.yml`," legacy DB tables get dropped as a
result — but then a `store_collections` table gets created anyway "to make
the DB self-contained," synced from the YAML via `syncConfigToDb`. Two
sources of truth for the same data, apparently from an incomplete migration.
Clean-slate lets us skip that entirely: `Store` owns collections directly in
its own table as the _only_ source of truth, with no file-based sync. Any
future file-based project config (e.g. a `qmd init`-style `.qmd/index.yml`)
is a CLI-layer decision for sub-project #3, layered on top of `Store`'s
DB-backed collection methods (`add_collection()`, `list_collections()`,
etc.) — not something storage itself needs to know about.

### Schema versioning (included, not deferred)

`content_vectors` gaining columns after real Node databases already existed
without them (`runContentVectorColumnRepairs`, triggered lazily on a
missing-column query error) is not a Node-specific concern — it's a generic
"this codebase's own schema evolved after real data existed" problem, and it
will recur for Python the first time a future release needs to add a column
to any table. Unlike the CJK case below, there's no free capability lying
around to cover this later — a fresh schema simply has no missing columns
_yet_.

**Decision:** include a `PRAGMA user_version`-based schema-version check at
`Store` construction, with a `migrate()` step structured as "if version <
current, run migration steps in order." Today that's a single no-op step
(v0 → v1 is just "create the current schema"), but the scaffolding exists
before any real migration is needed — retrofitting version-awareness onto a
database that has already accumulated real, un-versioned rows is much
harder than including this now, while it's nearly free.

### FTS rebuild: manual capability, not automatic triggering

`store.ts` automatically detects when CJK-normalization rules changed since
data was indexed and transparently rebuilds the FTS table on next use
(`rebuildFTSForCjkNormalization`). This is a real future concern for Python
too (not resolved by clean-slate — the _next_ normalization bug fix will
have the same problem), but the automatic-detection-and-trigger machinery
is UX polish, not a correctness requirement: since `documents_fts` is
entirely derived from the untouched `content` table, a full rebuild
("drop the FTS table, recreate it, re-index from `content` with current
normalization rules") is always possible with zero extra bookkeeping, no
matter when it's invoked.

**Decision:** `Store` exposes an explicit `rebuild_fts()` method (cheap,
useful for other maintenance scenarios too — e.g. recovering from
corruption). Automatic version-change detection is deferred; if/when
normalization rules change, `rebuild_fts()` is called explicitly (manually,
or by a future CLI command), not silently on startup.

### Explicitly not ported: legacy path migration

`findOrMigrateLegacyDocument` is not a generic "handle renamed paths"
mechanism — it bridges one specific historical event in Node's timeline: an
early version of the code slugged paths via `handelize()` (e.g. `"Budget &
Revenue (Q4) [2024].md"` → `"Budget-Revenue-Q4-2024.md"`), a later version
switched to literal paths, and this function repairs leftover rows from the
old convention. Python starts with literal paths from day one — there is no
`handelize()` phase in its history to bridge. If Python's own path-handling
changes in the future, that will need its own targeted migration written at
the time, informed by whatever actually changed — not a defensive mechanism
built now for a transition that hasn't happened.

## Chunking

Ported from `store.ts`'s regex/heading-aware chunking: `scan_break_points`
(markdown heading and paragraph boundaries via `BREAK_PATTERNS`),
`find_code_fences` / `is_inside_code_fence` (never cut inside a fenced code
block), `find_best_cutoff`, and `chunk_document_with_break_points` tying them
together with the existing sizing constants (900 tokens/chunk ≈ 3600 chars,
15% overlap). All pure functions, no DB.

**Out of scope:** AST-aware chunking (tree-sitter) — that's sub-project #5
per the roadmap, a separate strategy layered on top later, not a gap in this
port.

**Deferred, not included:** `chunkDocumentByTokensWithLlm`'s
token-verification safety net — a second pass that checks each char-based
chunk against the model's _real_ tokenizer and re-splits any chunk that
exceeds the token budget after all (the char/token ratio estimate can be
wrong for dense code or CJK-heavy text). This is a genuine correctness
safeguard, not cosmetic — but porting it means adding a `tokenize()`
function to `qmd.llm`'s public surface, which doesn't exist today. Deferred
because the risk is narrow (unusual content only) and a targeted fix,
informed by what actually triggers it, is cheaper than porting the full
recursive TS fallback chain speculatively now.

## Search & query pipeline

Ported faithfully from `hybridQuery`'s own docstring, which lays out a clean
algorithm worth preserving as-is rather than simplifying, since it _is_ what
"search quality" means for this tool:

1. **BM25 probe first** — if plain keyword search already gets a strong,
   unambiguous hit (score ≥ 0.85, ≥0.15 gap to the next result), skip query
   expansion entirely (saves an LLM call).
2. Otherwise, `expand_query()` (via `qmd.llm`) produces typed variants:
   `lex` (keyword rephrasings), `vec` (semantic rephrasings), `hyde`
   (hypothetical answer passage).
3. **Type-routed retrieval**: original query → vector search; `lex` → FTS;
   `vec`/`hyde` → vector search.
4. **Reciprocal Rank Fusion** merges all ranked lists (original-query lists
   get 2x weight over expansion-derived ones, plus a small top-rank bonus)
   and slices to a candidate limit (default 40).
5. Each candidate gets chunked and its single best-matching chunk selected
   by keyword overlap.
6. `rerank()` (via `qmd.llm`) scores each candidate's **best chunk**, not
   the full document body — reranking full bodies is an O(tokens) trap, and
   this is also exactly the gap flagged as a deferred finding in
   sub-project #1's validation harness (which reranked full bodies for
   simplicity). This port closes that gap for real usage, though the
   harness itself isn't retroactively fixed.
7. Final score blends RRF rank position with the reranker score.
8. Dedup by file, filter by `min_score`, slice to the requested limit.

`reciprocal_rank_fusion()` and `get_hybrid_rrf_weights()` are pure functions
(no DB, no I/O) — standalone module functions, not `Store` methods, per the
architecture decision above.

`HybridQueryResult` carries a `metadata` field in `store.ts`. Since metadata
filtering is deferred (below), the field stays on the Python result shape
for forward compatibility but is always `{}` in this sub-project — no
frontmatter extraction or filter logic yet.

## Testing & validation strategy

Sub-project #1 already validated MLX embed/rerank/expand _quality_ against
Node. The new risk here is different: SQLite/FTS5/`sqlite-vec` correctness
and whether the ported pipeline _logic_ survived translation — not model
quality again. Four layers:

1. **Fast unit tests** for pure functions (RRF scoring, chunking/break-point/
   code-fence detection) — no DB, no I/O, no MLX.
2. **Fast `Store` tests** against an in-memory SQLite DB (`:memory:`), with
   `qmd.llm`'s `embed`/`rerank`/`expand_query` passed in as fake/injected
   callables — covers schema, CRUD, FTS, vector search wiring, and hybrid
   query orchestration, all without loading real MLX models.
3. **One slow/real-model integration test** (same `slow` pytest marker
   convention as `qmd.llm`) running the whole pipeline end-to-end with real
   models on a small fixture, proving the pieces fit together for real.
4. **A validation run against the existing BEIR SciFact fixture** (already
   in `python/data/scifact/` from sub-project #1) — run the real
   `Store.query()` over it and sanity-check IR metrics (MRR/F1/Recall) land
   in a reasonable range. Not a strict "must match Node exactly" comparison
   like sub-project #1's — the individual pieces are already proven; this
   confirms the _assembled_ pipeline behaves sensibly.

## Deferred (explicitly out of scope for this sub-project)

Recorded so they aren't lost, not because they block anything:

- Metadata filtering (frontmatter extraction + `MetadataFilter` query
  support) — `metadata` field stays on result shape, always `{}` for now.
- CJK auto-rebuild-on-version-change trigger — manual `rebuild_fts()`
  capability is in scope; automatic detection is not.
- Cache/cleanup/vacuum utilities (`deleteInactiveDocuments`,
  `cleanupOrphanedContent`/`Vectors`, `vacuumDatabase`, etc.).
- Comma-list glob resolution (`resolveCommaListName` and friends).
- Chunk token-verification safety net (real-tokenizer re-split pass).
- AST-aware chunking (sub-project #5, tree-sitter Python bindings).
- Any file-based project config (`.qmd/index.yml`-style) — sub-project #3's
  call, layered on top of `Store`'s DB-backed collection methods.
