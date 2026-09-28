# pyqmd ↔ Node qmd feature-parity test suite — design

**Status:** Implemented (2026-09-12)
Date: 2026-09-12

## Purpose

A durable, reusable methodology for validating that the Python/MLX rewrite
(`pyqmd`, standalone repo at `python/`) behaves equivalently to the
reference Node implementation (`qmd`, this repo), covering the _entire_
Node feature surface — not just what's already built in `pyqmd` today. This
is a single design pass: the methodology and full feature inventory are
locked in now so that as each remaining rewrite sub-project (#5 AST
chunking, #8 `update`, and whatever else surfaces) gets its own
brainstorm, that brainstorm can write concrete test cases against an
already-agreed framework instead of re-deriving one.

**Explicit non-goal:** this is not a request to write detailed test cases
for features that don't exist in `pyqmd` yet. Those features haven't been
through their own design brainstorm (what will `pyqmd doctor` actually
check? what does `update` do once #8 is designed?), so detailed test cases
against an undesigned target would be speculative and likely rewritten
anyway. This suite covers already-built features with real, running tests,
and gives every other feature a lightweight inventory entry (what Node
does, which test category it'll fall into, what data it'll need) so a
future sub-project's brainstorm can move straight to writing test cases
using this suite's established harness.

**Second purpose, added during design review:** the entire harness is
built dataset-agnostic from the start, not SciFact-specific with SciFact
hardcoded. A **dataset profile** (a config file naming a corpus directory,
a queries file, and an optional relevance-judgments file) determines what
the suite runs against. SciFact ships as the built-in default profile for
this project's own internal validation, but the same harness lets anyone
considering replacing their live Node `qmd` with `pyqmd` build a profile
pointing at _their own_ documents and queries, capture a Node baseline
from their own corpus once (while they still have Node installed), and
then verify `pyqmd` stays consistent with that baseline as they migrate or
as `pyqmd` evolves. This is not a separate feature bolted on afterward —
it's the same architecture the SciFact validation already needs, just
without hardcoding the one dataset.

## Scope

**In scope:**

- A verified inventory of every Node qmd feature (CLI commands, flags, MCP
  tools), each classified as `done` / `not-yet-built` / `N/A` in `pyqmd`,
  with a proposed test category (`structural` / `quality` / `N/A`).
- A **dataset-profile abstraction**: a config file naming a corpus
  directory, a queries file, and an optional relevance-judgments (qrels)
  file, so the entire suite (structural and quality) runs against whatever
  profile is active rather than a hardcoded corpus.
- A golden-snapshot harness comparing `pyqmd`'s CLI/MCP output against
  frozen Node qmd output, for every `done` feature, parameterized by the
  active dataset profile.
- An IR-quality harness comparing `pyqmd`'s search quality against Node's,
  in two modes depending on whether the active profile has qrels: absolute
  metrics (MRR/nDCG/Recall) with a numeric tolerance margin when qrels
  exist (SciFact today), or relative agreement (rank correlation / top-K
  overlap between the two systems' own result lists) when they don't (the
  expected case for a user's own corpus) — extending sub-project #1/#2's
  existing scripts rather than replacing them.
- A manual, on-demand capture script, parameterized by dataset profile,
  that generates the golden snapshots and the Node quality baseline from a
  pinned Node qmd commit.
- The built-in SciFact profile, using the existing sub-project #1/#2 fixture.
- Real, running pytest coverage for every feature already in `pyqmd`,
  against the SciFact profile.

**Deferred / explicitly out of scope:**

- Detailed test cases for features not yet built in `pyqmd` (`init`,
  `context`, `doctor`, `update`, `trust`, `bench`, `cleanup`, AST chunking,
  `--full-path`/`--explain`/`--chunk-strategy`/`--index`/`-C`/`--timeout`/
  `--all`). Each gets a one-line inventory entry only.
- Live dual-invocation testing against a running Node process (rejected in
  favor of golden snapshots — see "Architecture").
- Performance/latency parity (sub-project #1 already has a latency bar for
  the reranker specifically; this suite is about correctness/quality, not
  speed).
- Actually building `pyqmd bench` (Node's own quality-benchmark command) —
  though this suite's quality harness is a natural prototype for what that
  command could eventually wrap; noted as a future-sub-project hook, not
  built here.

## Architecture: golden snapshots against a pinned Node commit

Live dual-invocation (running both `bun` and `pyqmd` per test) was
considered and rejected: this qmd fork is not tracking upstream and is
being actively replaced by the rewrite, so its reference behavior is
effectively frozen already. Under that premise, golden snapshots are
strictly better — no `bun`/Node runtime dependency for `pyqmd`'s day-to-day
test suite, faster (no second process per test), simpler CI.

The one risk golden snapshots normally carry — silent staleness when the
reference changes — is handled by recording the exact Node commit SHA the
snapshots were captured from (`parity/node_ref/<profile-name>/COMMIT.txt`) and never
re-capturing automatically. If someone deliberately changes Node qmd's
behavior before the rewrite fully replaces it, re-running
`capture_node_snapshots.py` is a manual, deliberate step — matching this
project's standing rule that indexing/query commands are never run
automatically.

## Dataset profiles

A dataset profile is a small YAML file naming what the suite runs against:

```yaml
# parity/datasets/scifact.yaml (built-in default)
name: scifact
corpus_dir: ../data/scifact/corpus # paths resolve relative to the profile file
queries_file: ../data/scifact/rerank-fixture.json # {query_id, query, ...}[] -- reuses the existing fixture
qrels_file: ../data/scifact/qrels-test.tsv # optional; presence of this key selects qrels-mode
```

```yaml
# parity/datasets/local/mine.yaml (a user's own profile, gitignored)
name: mine
corpus_dir: /Users/me/notes # their real collection
queries_file: ./mine-queries.yaml # a flat list of query strings they actually type
# no qrels_file -- they don't have curated relevance judgments, so this
# profile runs in agreement-mode automatically
```

`queries_file` accepts either shape: the existing SciFact fixture's
`{query_id, query, ...}` JSON (for continuity with sub-project #1/#2), or a
plain YAML/JSON list of query strings (auto-assigned stable IDs by index,
for a user who just wants to list queries they care about). `dataset_profile.py`
loads a profile, resolves both relative to the profile file's own
location, and exposes one flag the rest of the suite branches on:
`has_qrels: bool` — this is what selects **qrels-mode** (absolute metrics)
vs. **agreement-mode** (relative comparison) in `test_quality.py`, with no
separate configuration needed for that choice; it falls out of whether a
`qrels_file` key is present.

**Selecting the active profile:** a `--dataset-config <path>` pytest
option (or a `PARITY_DATASET_CONFIG` env var, for convenience), defaulting
to `parity/datasets/scifact.yaml` when neither is set. Running
`uv run pytest parity/` with no arguments always validates against
SciFact; `uv run pytest parity/ --dataset-config parity/datasets/local/mine.yaml`
(or a config anywhere else on disk — a user's profile need not live inside
this repo at all) runs the exact same suite against a user's own corpus.

**Privacy:** `parity/datasets/local/` is gitignored specifically so a
user's own documents, queries, and captured Node baselines never
accidentally get committed to this repo. A profile's `corpus_dir` can also
point anywhere on disk (absolute path), so a user validating their own
migration never needs to copy their real data into this repository at all
— only the profile YAML itself needs to exist somewhere reachable, and it
can live outside the repo too.

**Structural scenarios and dataset profiles:** a scenario like "seed the
corpus, search for a known query, assert the expected document comes
back" needs _some_ query/expected-result pair from the active profile to
run at all. For the built-in SciFact profile, scenarios reference the
fixture's own queries. For a user's own profile, `capture_node_snapshots.py`
generates the "expected" side by running each of the user's own queries
against their own frozen Node qmd first — the user's own captured Node
output _is_ the expected result, exactly like the SciFact case; there's no
new mechanism here, just no pre-existing fixture to draw from.

```
python/
  parity/
    README.md                       # methodology, capture/run instructions, how to build your own profile
    datasets/
      scifact.yaml                  # built-in default profile (checked in)
      local/                        # gitignored -- a user's own profile + private data live here
        .gitkeep
    node_ref/
      <profile-name>/
        COMMIT.txt                  # pinned Node qmd commit SHA + capture date, per profile
        snapshots/
          cli/                      # one JSON file per structural CLI scenario
          mcp/                      # one JSON file per structural MCP scenario
        quality/
          node_query_baseline.json  # present only for qrels-mode profiles (e.g. scifact)
          node_query_results.json   # present only for no-qrels profiles -- Node's raw result lists, for agreement comparison
    scenarios/
      cli_scenarios.py              # scenario definitions: name, command, args, expected-shape
      mcp_scenarios.py
    dataset_profile.py              # loads/validates a profile config, resolves paths, detects qrels-mode vs. agreement-mode
    test_structural.py              # pytest: pyqmd output vs. golden snapshot, per scenario, for the active profile
    test_quality.py                 # pytest: mode-dependent quality comparison for the active profile
    capture_node_snapshots.py       # manual, on-demand -- drives frozen Node qmd via bun, for the active profile
  data/scifact/                     # already exists (sub-project #1), reused as-is; referenced BY datasets/scifact.yaml:
                                     # rerank-fixture.json (30 queries + embedded qrels),
                                     # qrels-test.tsv, corpus/, queries.jsonl
  scripts/
    validate_store_query.py         # already exists (sub-project #2) -- test_quality.py's qrels-mode extends its pattern
```

`scenarios/` is the single source of truth for what gets tested: both
`capture_node_snapshots.py` (writes the golden files) and
`test_structural.py` (compares live `pyqmd` output against them) import
from the same scenario definitions, so the two can never silently drift
apart on what's actually being compared. The active profile determines
which corpus/queries those scenarios run against, but not what's being
asserted structurally — a scenario like "get by docid returns the right
document" means the same thing regardless of which corpus is loaded.

`node_ref/` is namespaced by profile name so multiple profiles' captured
baselines coexist without clobbering each other, and so the built-in
`scifact` profile's checked-in snapshots are never confused with a user's
own private captures.

## Feature inventory

Verified against the actual current code in both repos on 2026-09-12 (not
assumed from memory) — `src/cli/qmd.ts`'s help text and command switch for
Node, and a direct read of every `qmd/cli/commands/*.py` file plus
`qmd/mcp/server.py` for `pyqmd`.

### CLI commands

| Feature                                                          | pyqmd status                | Test category                                                       | Notes                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `query`                                                          | done                        | structural + quality                                                | Core hybrid search; `--filter`/`--intent`/`--no-rerank` all present                                                                                                                                                                                                                                                            |
| `search`                                                         | done                        | structural + quality                                                | `--filter` present (sub-project #7)                                                                                                                                                                                                                                                                                            |
| `vsearch`                                                        | done                        | structural + quality                                                | `--filter` present (sub-project #7)                                                                                                                                                                                                                                                                                            |
| `get`                                                            | done                        | structural                                                          | `--from`/`-l`/`--line-numbers`/`--no-line-numbers` all present; `Folder Context:` line added 2026-09-15                                                                                                                                                                                                                        |
| `multi-get`                                                      | done (comma-separated only) | structural                                                          | **Gap within a done feature**: Node's `multi-get` (and its own docstring) supports glob patterns; `qmd/cli/_multiget.py:26` only ever comma-splits — no glob matching. Structural tests must cover this gap explicitly (assert glob patterns are NOT supported yet, so a future fix doesn't silently change untested behavior) |
| `ls`                                                             | done                        | structural                                                          | No flags on either side                                                                                                                                                                                                                                                                                                        |
| `collection add`                                                 | done                        | structural                                                          | `--name`/`--mask`/`--exclude`/`--update-cmd` (sub-project #8); Node also supports `include`/`exclude` toggling (default-search inclusion) — **not yet built** in pyqmd                                                                                                                                                         |
| `collection list`/`show`/`remove`/`rename`                       | done                        | structural                                                          | `remove` has `--yes`/`-y`; matches Node's destructive-by-default cascade (verified in sub-project #3)                                                                                                                                                                                                                          |
| `collection include`/`exclude` (toggle default-search inclusion) | not-yet-built               | structural                                                          | Not implemented in pyqmd's `collection.py` at all                                                                                                                                                                                                                                                                              |
| `collection set-update`/`update-cmd` (pre-update hook)           | done                        | structural                                                          | Sub-project #8; `pyqmd collection update-cmd <name> [command...]` (variadic, no `set-update` alias)                                                                                                                                                                                                                            |
| `embed`                                                          | done                        | structural (correctness already covered by #1/#2's IR-quality work) | `-c`/`--collection`, `--force`; Node's `--timeout` (session cap) and `--chunk-strategy` are **not yet built**                                                                                                                                                                                                                  |
| `status`                                                         | done                        | structural                                                          | `Contexts: N` + truncated previews per collection added 2026-09-15 (replacing a flat, always-empty `Context:` line)                                                                                                                                                                                                            |
| `mcp` (stdio)                                                    | done                        | structural                                                          |                                                                                                                                                                                                                                                                                                                                |
| `mcp --http [--port]`                                            | done                        | structural                                                          | `--daemon`/`mcp stop` are **not yet built** (sub-project #4 explicitly deferred daemon mode)                                                                                                                                                                                                                                   |
| `init`                                                           | not-yet-built               | structural                                                          | Project-local `.qmd` index                                                                                                                                                                                                                                                                                                     |
| `context add`/`list`/`remove`                                    | done                        | structural                                                          | Human-written summaries attached to paths (2026-09-15); no global (`/`-wide) context, no `rm` alias (matches `collection`'s own `remove`, not `rm`)                                                                                                                                                                            |
| `doctor`                                                         | not-yet-built               | structural                                                          | Config/index/model/device diagnostics                                                                                                                                                                                                                                                                                          |
| `update [--pull]`                                                | done                        | structural                                                          | Sub-project #8; `--pull` is a real (pyqmd-only) implementation — Node's own `--pull` was dead code                                                                                                                                                                                                                             |
| `trust [list\|revoke]`                                           | not-yet-built               | structural                                                          | Approves checked-in `.qmd` config hooks/paths/models — meaningless until pyqmd has a checked-in-config-with-hooks story at all (see `collection set-update`, above)                                                                                                                                                            |
| `pull`                                                           | **N/A (confirmed)**         | N/A                                                                 | Node's `pull` explicitly pre-fetches GGUF models before use. `qmd.llm` (sub-project #1) already lazy-downloads MLX models from Hugging Face on first use — no architectural need for a separate pre-fetch command                                                                                                              |
| `bench`                                                          | not-yet-built               | N/A-for-this-suite, quality-shaped                                  | Node's own quality-benchmark command. This suite's `test_quality.py` harness is a natural prototype for what `pyqmd bench` could wrap later — noted, not built                                                                                                                                                                 |
| `skills list/get/path`, `skill show/install`                     | not-yet-built               | structural                                                          | Distributes bundled agent-skill instruction files. Initially proposed N/A (pyqmd's MCP `_instructions.py` covers _dynamic runtime_ instructions already) but the user confirmed this is a real future feature, not architecturally moot — pyqmd's own equivalent still needs its own design brainstorm when picked up          |
| `cleanup [--dry-run]`                                            | not-yet-built               | structural                                                          | Drop inactive docs/orphans, compact FTS, vacuum                                                                                                                                                                                                                                                                                |

### Search/embed option surface (cross-cutting, applies to `search`/`vsearch`/`query` unless noted)

| Option                               | pyqmd status                            | Test category        | Notes                                                                                                                                                     |
| ------------------------------------ | --------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-c`/`--collection` (repeatable)     | done                                    | structural           |                                                                                                                                                           |
| `--filter <json>`                    | done                                    | structural + quality | Sub-project #7                                                                                                                                            |
| `-n` (limit)                         | done                                    | structural           |                                                                                                                                                           |
| `--min-score`                        | done                                    | structural           | Sub-project #7's `skip_rerank` fix (`1/rank` scale) is a real, already-fixed regression worth a dedicated structural test in this suite                   |
| `--all`                              | not-yet-built                           | structural           |                                                                                                                                                           |
| `--full`                             | done                                    | structural           |                                                                                                                                                           |
| `-C`/`--candidate-limit`             | not-yet-built (CLI flag)                | structural           | `Store.query()` already accepts `candidate_limit`; not exposed as a CLI option yet                                                                        |
| `--no-rerank` (`query` only)         | done                                    | structural + quality |                                                                                                                                                           |
| `--no-gpu`                           | **N/A (confirmed)**                     | N/A                  | No GPU/CPU backend-selection story exists in MLX's unified-memory model — this switch has no MLX equivalent by design                                     |
| `--line-numbers`/`--no-line-numbers` | done                                    | structural           |                                                                                                                                                           |
| `--full-path`                        | not-yet-built                           | structural           | Sub-project #3 explicitly deferred this                                                                                                                   |
| `--explain`                          | not-yet-built                           | structural           | Retrieval score traces                                                                                                                                    |
| `--format <kind>`                    | done (all 6: cli/json/csv/md/xml/files) | structural           |                                                                                                                                                           |
| `--index <name>` (named multi-index) | not-yet-built                           | structural           | pyqmd currently has exactly one global DB path (`~/.cache/pyqmd/index.sqlite`, overridable only via `PYQMD_DB` env var, not a per-invocation named index) |
| `--intent` (`query` only)            | done                                    | structural + quality | Sub-project #3                                                                                                                                            |
| `--chunk-strategy <auto\|regex>`     | not-yet-built                           | structural           | Roadmap #5 (AST chunking)                                                                                                                                 |
| `--timeout` (embed session cap)      | not-yet-built                           | structural           |                                                                                                                                                           |

### MCP surface

| Feature                   | pyqmd status  | Test category        | Notes                                                                               |
| ------------------------- | ------------- | -------------------- | ----------------------------------------------------------------------------------- |
| `query` tool              | done          | structural + quality | Now includes `filter` (sub-project #7)                                              |
| `get` tool                | done          | structural           |                                                                                     |
| `multi_get` tool          | done          | structural           | Same comma-only gap as the CLI's `multi-get`                                        |
| `status` tool             | done          | structural           |                                                                                     |
| `qmd://{path}` resource   | done          | structural           |                                                                                     |
| stdio transport           | done          | structural           |                                                                                     |
| Streamable HTTP transport | done          | structural           | Origin/host guard already covered by sub-project #4's own tests; not re-tested here |
| HTTP `--daemon` mode      | not-yet-built | structural           |                                                                                     |

Node's MCP tool surface is exactly these 4 tools + 1 resource (verified
directly against `src/mcp/server.ts`'s `registerTool` calls) — this side of
the inventory is already 1:1 with `pyqmd`, making it the easiest category
to get full structural coverage on immediately.

## Structural test methodology

Each scenario in `scenarios/cli_scenarios.py` / `scenarios/mcp_scenarios.py`
is a plain data structure: a name, the command/tool + arguments to run, and
a description of what in the output must match (e.g. "the set of returned
document paths," "the error message shape," "the JSON structure's keys").
Comparisons are on _model-independent_ content — which documents come
back, error behavior, output structure — never on floating-point
scores/rankings, since those differ between MLX and GGUF by design.

`capture_node_snapshots.py` runs every scenario against a frozen Node qmd
checkout (via `bun`) over a collection freshly seeded from the active
profile's `corpus_dir`, and writes each result as a JSON file under
`parity/node_ref/<profile-name>/snapshots/`. `test_structural.py` runs the
same scenario against live `pyqmd` using the same active profile, loads
the matching snapshot, and asserts the model-independent parts match.

## Quality test methodology

Branches on the active profile's `has_qrels` flag (see "Dataset profiles"):

**Qrels-mode** (SciFact today, or any profile that supplies curated
relevance judgments): extends `scripts/validate_store_query.py`'s existing
pattern — real `Store.query()` over the profile's queries, scored against
its qrels. `test_quality.py` computes MRR/nDCG/Recall for `pyqmd` and
asserts each metric is `>= node_baseline - margin`, where `margin` starts
at **0.05 absolute** — a starting point matching the order of magnitude of
sub-project #1's own bar (Spearman correlation, ~2x latency ceiling), to be
tuned once real side-by-side numbers exist for the full hybrid pipeline
(sub-project #1 only measured Node's reranker in isolation; Node's
full-pipeline `query` baseline doesn't exist yet and must be captured
fresh by `capture_node_snapshots.py`). `node_query_baseline.json` holds
Node's own measured MRR/nDCG/Recall.

**Agreement-mode** (any profile with no `qrels_file` — the expected case
for a user's own corpus): there is no ground truth to score either system
against, so the comparison is system-to-system instead. For each query,
`test_quality.py` runs `pyqmd query` live and loads Node's captured result
list from `node_query_results.json` (the raw ranked document list Node
returned for that same query, captured once), then computes: top-K
overlap (what fraction of Node's top-K documents also appear in pyqmd's
top-K) and Spearman rank correlation over the documents both systems
returned — the same style of system-agreement metric sub-project #1 used
when it wasn't asserting absolute correctness, just comparing two systems.
Reports both per-query and averaged; a tolerance threshold here (e.g.
overlap `>= 0.7`) is a starting point to tune once real numbers exist on
an actual second dataset, not a value to over-specify against a dataset
that doesn't exist yet.

## Capture script

`capture_node_snapshots.py` is a manual, on-demand script, parameterized
by `--dataset-config` (same flag/env-var convention as the test suite,
defaulting to the SciFact profile) — never run automatically by CI or by
an agent unprompted, matching this project's standing rule against running
indexing/query commands automatically. It:

1. Records the current Node qmd commit SHA + capture date into
   `parity/node_ref/<profile-name>/COMMIT.txt`.
2. Seeds a fresh collection from the active profile's `corpus_dir` and runs
   every structural scenario against frozen Node qmd via `bun`, writing
   results to `parity/node_ref/<profile-name>/snapshots/`.
3. Runs Node's real `qmd query` end-to-end over the active profile's
   queries. If the profile has qrels, computes and writes
   MRR/nDCG/Recall to `node_query_baseline.json` (qrels-mode). If not,
   writes Node's raw per-query result lists to `node_query_results.json`
   (agreement-mode) for `test_quality.py` to compare `pyqmd`'s own live
   results against later.

Re-run only when deliberately updating the pinned reference (for the
built-in SciFact profile) or when a user wants to refresh their own
profile's baseline — the README documents both cases explicitly.

## Deferred / explicitly out of scope (restated)

- Detailed test cases for `init`/`context`/`doctor`/`update`/`trust`/
  `pull`/`bench`/`skills`/`skill`/`cleanup` and the not-yet-built option
  surface — each gets only the inventory entry above until its own
  sub-project brainstorm designs it.
- `pull` and `--no-gpu` are confirmed N/A (user decision, 2026-09-12).
  `skills`/`skill` was initially proposed N/A but the user confirmed it's a
  real future feature — reclassified as `not-yet-built`, structural
  category, pending its own design brainstorm.
- Live dual-invocation testing.
- Actually implementing `pyqmd bench`.
- A UI/wizard for building a custom dataset profile — a user hand-writes
  the YAML file per the documented schema; no scaffolding tool is built
  for this now.
- Tuning the agreement-mode overlap/correlation thresholds against real
  numbers — there's no second real dataset to tune against yet; the
  starting values are placeholders explicitly flagged as such.
