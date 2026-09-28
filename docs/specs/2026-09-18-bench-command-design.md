# `pyqmd bench` command — design

**Status:** Implemented (2026-09-19)

Roadmap sub-project #12 (user-facing search-quality benchmarking),
added 2026-09-17. Trigger: once other people are indexing their own
corpora via a PyPI-published pyqmd, "did this config/chunking/rerank
change help or hurt _my_ search quality" becomes a real end-user
question, not only a maintainer/parity-suite one. Node's `qmd` already
solves this with `qmd bench <fixture.json> [--json] [-c collection]`
(`src/cli/qmd.ts`'s `case "bench"`, `src/bench/bench.ts`,
`src/bench/score.ts`, fixture format in
`src/bench/fixtures/example.json`).

## What `bench` is actually for

Worth stating explicitly, since it shaped every decision below: `bench`
is a personal regression suite for an _installed user's own corpus_,
not a Node-vs-pyqmd comparison tool. Its audience is someone who has
already indexed their own notes/docs and wants to answer "does search
still find what I expect?" after changing chunking, swapping embedding
models, toggling reranking, reorganizing their corpus, or upgrading
pyqmd. They write a small fixture by hand — a handful of queries they
personally care about, each paired with which file(s) should come back
and how high. Running all 4 backends per query gives a diagnosis, not
just a score: if bm25 finds a doc but vector doesn't, that says
something about the embedding model for that kind of query; if
reranking promotes a doc from position 5 to position 1, that's evidence
reranking earns its cost.

This is a different job from the two existing dev-only tools in
`parity/`:

- `parity/benchmark.py` — Node-vs-pyqmd **performance** timing on a
  fixed reference corpus, run by maintainers to validate the port.
- `parity/test_quality.py` — Node-vs-pyqmd **quality-regression**
  testing against the frozen scifact corpus, also maintainer-only.

Both of those exist to build confidence in the rewrite itself and stop
mattering once pyqmd is trusted standalone. `bench` ships permanently
inside pyqmd, for real users' real corpora, with no Node involved at
all. A tempting side-use — hand-running the same fixture through both
tools' `bench`/`bench` commands to eyeball Node-vs-pyqmd fidelity — was
considered and explicitly rejected as a design driver: it's an
improvised use of a fixture-format coincidence, not `bench`'s job, and
the tools that do that job already exist and do it more rigorously
(above). Optimizing `bench`'s metric set to preserve that side-use
would compromise its real, permanent audience for a temporary one.

## Scope decisions (from brainstorming)

- **Fixture format: keep Node's JSON shape, drop structured queries.**
  Same field names as Node's `fixture.json` (`id`, `query`, `type`,
  `description`, `expected_files`, `expected_in_top_k`) — anyone with
  an existing Node fixture can reuse it. Node's multi-line
  `lex:`/`vec:`/`hyde:`/`intent:` structured-query syntax is **not**
  ported: it's parsed only inside `Store.query()` today
  (`_parse_expanded_lines`, `store.py:1419`), not in `search_fts`/
  `search_vec`, and adding per-backend parsing would be new code
  serving only this one feature. Plain query strings only.

- **Metrics: reuse pyqmd's own already-tested IR metrics, not Node's
  bespoke set.** Node's `score.ts` computes `precision_at_k`,
  `recall_at_1/3/5`, overall `recall`, `mrr`, `f1` — none of which
  exist in pyqmd today. pyqmd already has `reciprocal_rank`/
  `mean_reciprocal_rank`, `recall_at_k`, and `ndcg_at_k` in
  `parity/_ir_metrics.py`, tested and used by the parity suite. `bench`
  reuses these unchanged rather than porting Node's metric math. This
  means `bench`'s output isn't numerically comparable to Node's own
  `bench` output field-for-field (only MRR lines up exactly) — an
  accepted tradeoff given `bench`'s real audience per above.
  **Side effect:** `parity/_ir_metrics.py` is dev-only and excluded
  from the wheel (only `src/qmd/` is packaged per
  `pyproject.toml`'s `packages = ["src/qmd"]`), so these pure functions
  **move** to `src/qmd/bench/_metrics.py` — not duplicated — and
  `parity/`'s own tests/callers switch their import to the new
  location.

- **Backends: all 4, matching Node.** `bm25` (`Store.search_fts`),
  `vector` (`Store.search_vec`), `hybrid` (`Store.query(...,
skip_rerank=True)`), `full` (`Store.query(...)`, reranked). Thin
  wrappers over methods that already exist, and exactly the
  before/after diagnostic the roadmap item exists for.

- **Path matching: port Node's exact-or-suffix semantics.** Node's
  `normalizePath` already strips both the `qmd://` scheme _and_ the
  collection segment before comparing (`qmd://collection/docs/x.md` →
  `docs/x.md`), so fixture authors write bare relative-within-collection
  paths either way — no difference there. The real gap is Node's
  `pathsMatch` endswith-either-direction fallback on top of the exact
  check, letting a fixture author write `expected_files: ["readme.md"]`
  and match an actually-indexed `docs/subdir/readme.md` without needing
  to check `pyqmd ls` first. Cheap to port and a real onboarding win
  (especially for someone reusing an existing Node fixture that already
  relies on it) — ported as-is.

  **Amended 2026-09-26 — intentional divergence from Node.** "As-is"
  turned out to include a bug Node shares: its bare `endsWith` has no
  path-segment boundary, so `projects-<id>-areas-alpha-notes-md.md`
  (a different file) also matched `areas-alpha-notes-md.md`.
  Canonicalization then mapped both ranked results to the same expected
  id, and `ndcg_at_k` credited it twice — a real run on 2026-09-25
  reported bm25 nDCG@k = 1.63. Node's own metrics count hits per
  expected file, so they never go above 1, but the same false match can
  still give MRR/recall credit to the wrong file. pyqmd now differs from
  Node in three ways: (1) a suffix match must start at a `/` boundary
  (`docs/subdir/readme.md` still matches `readme.md`, `docs/myreadme.md`
  no longer does); (2) `canonicalize_ranked_ids` lets each expected entry
  be claimed by at most one ranked path — the highest-ranked match,
  preferring an exact match over a suffix match — so a bare filename
  matching several indexed files is credited once; (3) `ndcg_at_k` gives
  a repeated relevant id gain only at its first occurrence, so the score
  can never exceed 1. A Node fixture that relied on non-boundary suffix
  matching (a partial filename) now needs the full filename.

- **Migrated-fixture handling: honor the fixture-level `collection`
  field, reject structured queries loudly instead of silently
  mis-scoring.** Prompted by a direct question: would a Node user, after
  re-indexing the same corpus in pyqmd, be able to reuse their existing
  fixture file unmodified? Two gaps found and closed:
  - Node's fixture JSON has an optional top-level `collection` field
    (`runBenchmark`'s `options.collection ?? fixture.collection`), used
    so a fixture is self-contained about which collection it targets.
    `BenchFixture` carries this field, and the CLI resolves the target
    collection as `-c`/`--collection` if given, else `fixture.collection`,
    else all collections — same precedence as Node.
  - A migrated fixture whose `query` field uses the (unsupported, see
    above) `lex:`/`vec:`/`hyde:`/`intent:` structured syntax would
    otherwise be silently treated as one literal multi-line query
    string, search for that garbled text, and score near-zero with no
    indication why. `_fixture.py` instead detects a query whose text
    starts with one of those prefixes and rejects the fixture at load
    time with an error naming the offending query id — a loud failure
    instead of a silent, confusing one.

## Design

### Architecture & file layout

New `src/qmd/bench/` package, mirroring the existing split-by-
responsibility pattern in `store/` and `cli/`:

- **`src/qmd/bench/_metrics.py`** — the pure IR-metric functions,
  moved from `parity/_ir_metrics.py` (`reciprocal_rank`,
  `mean_reciprocal_rank`, `recall_at_k`, `ndcg_at_k`). No behavior
  change; their existing tests move with them. `parity/`'s own
  consumers import from the new location.
- **`src/qmd/bench/_pathmatch.py`** — ports Node's `normalizePath`/
  `pathsMatch`: normalizes a `qmd://collection/path` (or bare path)
  down to a lowercased, collection-stripped path, then exposes
  `canonicalize_ranked_ids(ranked_paths, expected_files)`, which
  rewrites each ranked result to the exact `expected_files` string it
  matches (exact-or-suffix, either direction) so `_metrics.py`'s
  functions can do plain exact-set-membership scoring afterward — no
  fuzzy-matching logic leaks into the metrics themselves.
- **`src/qmd/bench/_fixture.py`** — loads and validates the fixture
  JSON into simple dataclasses (`BenchQuery`, `BenchFixture` — the
  latter carrying `description`, `version`, `collection: str | None`,
  and `queries`), raising a clear, actionable error on malformed input
  (missing `queries` array, missing required fields, invalid JSON, or a
  query using the unsupported structured `lex:`/`vec:`/`hyde:`/
  `intent:` syntax).
- **`src/qmd/bench/runner.py`** — orchestrates: for each query × each
  of the 4 backends, calls the matching `Store` method, dedupes/
  canonicalizes result paths, scores with `_metrics.py`, collects
  per-query and summary (averaged) results.
- **`src/qmd/cli/commands/bench.py`** — the Typer command itself, a
  plain single-command function registered flat via
  `app.command("bench")(bench.bench)` (matching `status.py`/`pull.py`'s
  precedent), not a sub-app — there's only one verb, so there's no
  reason to introduce the `add_typer`/single-command-collapse quirk
  that only affects namespaced multi-command groups like `skill`/
  `collection`.

### Data flow

```
pyqmd bench <fixture.json> [--json] [-c/--collection <name>]
  |
  load + validate fixture JSON -> BenchFixture (rejects structured
  lex:/vec:/hyde:/intent: queries at this point, naming the query id)
  |
  resolve target collection: -c/--collection if given, else
  fixture.collection, else all collections (Node's own precedence)
  |
  readiness check: does the target collection (or, if none given, does
  any collection) have active indexed documents? If not, fail fast
  with a clear message ("Run 'pyqmd collection add'/'pyqmd update'
  first") -- mirrors Node's assertBenchCollectionReady, avoids
  grinding through every query just to print all-zero scores.
  |
  for each query, for each of the 4 backends:
    - call the matching Store method with
      limit = max(query.expected_in_top_k, 10)
    - dedupe result paths (a doc could otherwise appear twice via
      different chunks), preserving rank order
    - canonicalize each result path against expected_files
      (exact-or-suffix match) via _pathmatch
    - score with _metrics.py: reciprocal_rank,
      recall_at_k(k=expected_in_top_k), ndcg_at_k(k=expected_in_top_k)
    - record latency_ms
  |
  aggregate: per-backend averages across all queries
  |
  output: human-readable table + summary (default), or full JSON dump
  (--json)
```

### Error handling

Each backend call is wrapped individually — if one backend errors
(e.g. embeddings not generated yet), that backend scores 0 for that
query rather than aborting the whole run, matching Node's resilience
(Node's own `search_vec` equivalent already returns `[]` rather than
raising when the vector table doesn't exist, but the wrap covers any
other failure mode too, e.g. a model load failure).

If every backend's aggregate score is 0 across the board, print a
warning suggesting the collection may be unindexed or the fixture's
`expected_files` don't match anything real (porting Node's
`allZeroBenchWarning` — a real "why is everything zero" trap for a
first-time user).

### CLI surface & output format

```
pyqmd bench <fixture.json> [--json] [-c/--collection <name>]
```

Exactly Node's flag set — no `-n`, `--full-path`, etc.; this command's
shape is fixed by its own fixture (per-query `expected_in_top_k`
drives the limit, not a global `-n`).

Default output: a per-query/per-backend table (`recall_at_k`, `mrr`,
`ndcg_at_k`, `latency_ms`) followed by a per-backend summary of
averages. `--json` dumps the same data as structured JSON (fixture
path, timestamp, per-query results, summary) — no separate schema,
just the same numbers in a machine-readable shape.

### Testing plan

- `_metrics.py`: existing `parity/_ir_metrics.py` tests move with the
  code, unchanged.
- `_pathmatch.py` (new logic): exact match, suffix match in either
  direction, `qmd://collection/` stripping, case-insensitivity, and a
  no-match case. (Added 2026-09-26: suffix matching only at a `/`
  boundary, each expected file credited once, exact beats suffix.)
- `_fixture.py`: valid fixture loads correctly (including a
  fixture-level `collection` field); missing `queries` array, malformed
  JSON, missing required fields, and a query using structured
  `lex:`/`vec:`/`hyde:`/`intent:` syntax each raise a clear error
  naming the problem (and, for the structured-query case, the offending
  query id).
- `runner.py`: a small seeded in-memory `Store` (matching existing
  store test patterns) run through all 4 backends with hand-computed
  expected recall/mrr/ndcg; one test monkeypatches a backend to raise
  and confirms the run continues with a 0 score instead of aborting.
- CLI (`bench.py`): `CliRunner`-based — table output for a healthy
  fixture, `--json` produces valid parseable JSON, the readiness-check
  error fires for a missing/empty collection, and `-c` overrides a
  fixture's own `collection` field when both are present.

### Documentation updates

Part of this work, per the doc-sync convention established for the
`skill` commands: `CLAUDE.md`'s Commands section and
`src/qmd/skills/qmd/SKILL.md` both get the new `pyqmd bench` entry, a
new `COMMAND_STATUS.md` row, and the roadmap doc's #12 entry marked
done once shipped.
