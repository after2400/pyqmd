# Parity-suite coverage for mutating CLI commands — design

**Status:** Implemented (2026-09-17)

Roadmap: `2026-09-10-python-mlx-rewrite-roadmap.md`, "Next step" section,
pickup-order item 1. Not a numbered sub-project — a validation-coverage
task against the existing parity suite (`parity/README.md`), same
cross-cutting category as that suite itself.

## Problem

`COMMAND_STATUS.md` marks eight commands 🟡 **unverified** against Node:
`collection add`/`remove`/`rename`/`update-cmd`, `context add`/`list`/
`remove`, `update`, and `embed`. Not suspected broken — just never
structurally cross-checked against the real Node CLI the way `search`/
`get`/`ls`/etc. already are.

They don't fit the existing parity-suite mechanism as-is. Every current
`CliScenario` (`parity/scenarios/cli_scenarios.py`) is a single,
independent, **read-only** CLI call, run against one **session-scoped,
shared** already-built index (`indexed_pyqmd_store` in `parity/
conftest.py`, embedding the full scifact corpus once because that's
expensive). All eight commands here are mutating, and several are
inherently sequential — `collection rename` needs a collection that
already exists, `context remove` needs a prior `add`, `update` needs an
actual file change on disk to react to. They cannot share the existing
session store without corrupting other tests' fixed expectations, and
forcing each into an "independent" scenario would mean silently
duplicating setup (via direct `Store` calls, not CLI) inside every single
scenario — hiding exactly the kind of interaction (does `add` really
leave the state `list` expects?) this work exists to check.

## Reference implementation

Node's `src/cli/qmd.ts`: `case "collection"` (`addCollection`/
`listCollections`/`showCollection`/`removeCollection`/`renameCollection`/
`setUpdateCommand`), `case "context"`, `case "update"`
(`updateCollections`), `case "embed"` (`vectorIndex`).

## Findings from investigating this (avoid re-deriving them)

- **`update --pull` is a dead flag in Node.** `cli.values.pull` is parsed
  (`pull: { type: "boolean" }`, `qmd.ts:3137`) and mentioned in help text,
  but `updateCollections()` — the function `case "update"` actually calls
  — never reads it anywhere (`grep -n "values.pull"` across the whole
  file returns nothing). There is no real Node behavior to compare
  pyqmd's own working `--pull` against, so it's excluded from this
  work's scope entirely — not xfailed, just not attempted.
- **`COMMAND_STATUS.md`'s existing `pull` (standalone) row is wrong** and
  gets corrected as part of this work. It currently reads "❌ (folded into
  `update --pull`)" — but Node's standalone `qmd pull` downloads
  embedding/generation/rerank **GGUF model files**; it has nothing to do
  with git or with `update`'s dead `--pull` flag. The right framing is
  that pyqmd's MLX models load straight from the Hugging Face Hub cache
  on first use, so no separate "pull the model first" step exists or is
  needed — not that a Node feature was folded elsewhere.
- **A real, small correctness gap, found while reading pyqmd's `embed`
  for this spec**: Node's embed path validates `-c/--collection` against
  configured collections up front, so a typo errors clearly (`Collection
not found: X`, comment at `qmd.ts` above `resolveCollectionFilter`
  call). pyqmd's `Store.get_indexable_content(collection)`
  (`src/qmd/store/store.py:872`) just runs a `WHERE d.collection = ?`
  query with no existence check — `pyqmd embed -c typo-name` silently
  prints `Nothing to embed.` and exits 0. This is a real bug, not a
  documented scope gap, and gets fixed alongside adding its scenario
  (see Decision 6).
- **`collection remove` prompts for confirmation** unless `--yes`/`-y`
  is passed (`src/qmd/cli/commands/collection.py`'s `remove`). Every
  flow step that calls it must pass `--yes`, or `CliRunner.invoke` will
  hang waiting on stdin / fail the confirm read. Node's `collection
remove` never prompts at all — no `--yes`/confirmation concept exists
  there. This means, unlike every other step so far, the two sides need
  _different_ argv for this one step (Decision 2 amended below).
- **Node's `update` has no `-c`/`--collection` filtering at all** —
  `resolveCollectionFilter` (the validation Node's `embed`/`search`/
  `vsearch`/`query` all call) is never called from `case "update"`. A
  stray `-c`/`--collection` on Node's `update` parses fine (the flag is
  declared globally for the whole CLI) and is then silently ignored —
  Node always re-scans every collection regardless. pyqmd's `update -c
<name>` (including its existing, correct "No such collection" error
  for a bad name) is a real added capability with no Node equivalent to
  compare against — not a bug, and not something an error-path scenario
  can meaningfully check both sides against. Excluded from
  `update_lifecycle`'s error step for the same reason as `update --pull`
  above: no real Node behavior on the other side of the comparison.

## Decisions

1. **Fixture corpus**: a new, small, synthetic corpus — 3-5 tiny
   `.md` files, checked into `parity/fixtures/mutating/` — distinct from
   the shared scifact profile. Copied fresh into a `tmp_path` per flow
   test. Keeps `embed`/`update` runs fast and deterministic; re-embedding
   the full scifact corpus per mutating scenario (as the existing
   `indexed_pyqmd_store` fixture already notes is expensive) would make
   this suite's runtime balloon for no benefit — none of these scenarios
   need real corpus scale, they need controlled, exactly-known content.
2. **New `CliFlowScenario` type**, additive to `parity/scenarios/
cli_scenarios.py` (or a new sibling module,
   `parity/scenarios/cli_flow_scenarios.py`, kept separate since the
   fixture/isolation mechanics genuinely differ from the read-only
   scenarios' — see Decision 3). Shape:

   ```python
   @dataclass
   class CliFlowStep:
       name: str                                  # unique within the flow
       args: list[str]                             # used for pyqmd, and for Node unless node_args is set
       extract: Callable[[str, int], object]
       expect_failure: bool = False               # documents an error-path step
       node_args: list[str] | None = None          # override when Node's argv genuinely differs

   @dataclass
   class CliFlowScenario:
       name: str
       steps: list[CliFlowStep]
   ```

   `node_args` exists for the one case found so far where the two CLIs
   can't share one argv list at all: `collection remove` needs
   `--yes`/`-y` on the pyqmd side to skip its confirmation prompt, but
   Node's `collection remove` has no such flag or prompt (passing
   `--yes` there would just be an unrecognized extra token). Every other
   step in every flow uses identical argv on both sides, matching the
   rest of the suite's existing scenarios.

   Each flow runs sequentially against one fresh store/corpus; every
   step's `(args, extracted, exit_code)` becomes one entry in that flow's
   snapshot (a dict keyed by step name, not one file per step — keeps a
   flow's snapshot reviewable as a single unit reflecting its narrative).

3. **Isolation mechanism**: mirrors `capture_node_snapshots.py`'s
   existing `_fresh_isolated_index` (own index.sqlite + own derived
   `QMD_CONFIG_DIR`) for the Node side. For live pyqmd
   (`test_structural.py`), a new function-scoped fixture
   `fresh_pyqmd_store(tmp_path)` (a bare `Store` over a fresh
   `tmp_path/index.sqlite`, no pre-registered collection) replaces
   `indexed_pyqmd_store` for flow tests only — the existing session-scoped
   fixture is untouched, still used by every existing read-only
   `CliScenario`.
4. **Extraction signal**: reuse existing shape functions where a step's
   output already matches one (`_collection_list_shape`,
   `_collection_show_shape` for the `show` step). New extractors, all
   following the suite's established preference for structural signals
   over literal/colored text:
   - `_ok_shape(stdout, exit_code) -> {"exit_code": ...}` for pure
     confirmation commands (`add`, `rename`, `remove`,
     `update-cmd`, `context add`/`remove`) — a flow's very next step is
     always a `list`/`show` that verifies the mutation actually landed,
     so the confirmation step itself only needs to assert success/failure,
     not parse wording that #10 already deliberately left uncolored/
     unnormalized for some of these commands.
   - `_context_list_shape` — group-by-collection structural shape (which
     collections have contexts, how many entries each), same
     never-compare-raw-text philosophy as `_collection_list_shape`.
   - `_update_counts_shape` — parses pyqmd's `Indexed: N  Updated: N
Unchanged: N  Removed: N` line (and Node's equivalent — its own
     wording gets confirmed, not assumed, during capture) into a dict of
     ints. Exact counts are a legitimate structural check here (unlike
     the existing `_status_shape`'s deliberate "presence only" comment for
     the _scifact_ corpus) because this fixture's file set is fully
     controlled by the flow itself.
   - `_embed_counts_shape` — parses `Embedded N chunk(s) across M
document(s).` / `Nothing to embed.` / `Cleared N existing
embedding(s).` into a dict, `None` for absent lines.
5. **Error-path steps**, at least one per flow where a genuine shared
   Node/pyqmd failure mode exists, each with `expect_failure=True`,
   asserting matching exit-code-nonzero-ness (not matching error text,
   which #10 never attempted to make byte-identical): duplicate
   `collection add`, `collection rename` onto an existing name,
   `collection remove` on an already-removed collection, `context add`
   on a path outside any indexed collection, `embed -c <nonexistent>`.
   `update_lifecycle` has none — see Decision/finding above.
6. **Fix the `embed -c <nonexistent>` gap** found above as part of this
   work, not just document it: add the same up-front existence check
   Node has (`Store.get_collection(collection) is None` →
   `ValueError(f"Collection not found: {collection}")`, routed through
   the existing `run_or_exit` pattern already used by every other
   mutating command in `collection.py`/`context.py`). This is a one-line
   fix directly adjacent to work already touching `embed.py`'s tests, not
   a scope creep — leaving a newly-found, trivially-fixable correctness
   bug undocumented-and-unfixed in the same file this task is already
   adding coverage for would be worse than fixing it now.
7. **`update --pull` and the `pull` (standalone) command**: excluded from
   scope (Finding above). `COMMAND_STATUS.md`'s `pull` row corrected to
   reflect the real Node feature (GGUF model download) instead of the
   "folded into `update --pull`" framing, which investigation showed was
   never accurate. A stub `pyqmd pull` (exits 0, explains MLX models
   auto-download so no pull step exists) is a plausible follow-up UX
   nicety, but it isn't a parity gap — Node's real download behavior has
   no pyqmd analog to check against — so it's tracked as a roadmap
   backlog item instead of built here.

## The four flows

Each gets its own fresh `fresh_pyqmd_store`/temp fixture-corpus copy (or,
for `context_lifecycle`/`update_lifecycle`/`embed_lifecycle`, one fresh
store with the fixture collection already added+scanned as flow setup —
setup steps are still real CLI invocations captured like any other step,
not hidden `Store` calls, so the flow reads as a complete, honest
transcript).

1. **`collection_lifecycle`**, in order:
   1. `add` (fixture corpus, name `flow-a`)
   2. `add` again, same name (error — duplicate)
   3. `add` (fixture corpus, name `flow-dummy`) — a second collection,
      needed only so step 6 has an existing name to collide with
   4. `list`
   5. `update-cmd flow-a -- echo hi`
   6. `rename flow-a flow-dummy` (error — name already taken)
   7. `rename flow-a flow-b` (real rename, succeeds)
   8. `show flow-b`
   9. `remove flow-b --yes`
   10. `remove flow-b --yes` again (error — already gone)
   11. `list` (shows only `flow-dummy` left)
2. **`context_lifecycle`** (setup: `collection add` the fixture corpus,
   no embed needed — context doesn't require vectors): `context add
qmd://flow-ctx/ "root context"` → `context add
qmd://flow-ctx/sub.md "file context"` → `context list` → `context add
./not-a-real-path "x"` (error) → `context remove qmd://flow-ctx/` →
   `context remove qmd://flow-ctx/` again (error, already gone) →
   `context list`.
3. **`update_lifecycle`** (setup: `collection add` the fixture corpus):
   mutate the temp corpus copy on disk (edit one file's content, add a
   new file, delete one file) → `update` (expect `Indexed: 1  Updated:
1  Unchanged: N  Removed: 1`, exact N known from the fixture's size).
   No error-path step — see the "Node's `update` has no `-c` filtering"
   finding above; there is no Node behavior left to exercise a failure
   against once `--pull` and `-c` are both out of scope for this flow.
4. **`embed_lifecycle`** (setup: `collection add` the fixture corpus,
   unembedded): `embed` (initial `Embedded N chunk(s) across M
document(s).`) → `embed` again (`Nothing to embed.`) → `embed
--force` (`Cleared N existing embedding(s).` + re-embed line) →
   `embed -c does-not-exist` (error, once Decision 6 lands — currently
   would be `Nothing to embed.`/exit 0, which is exactly the gap being
   fixed).

## Integration points

- `parity/fixtures/mutating/*.md` — new fixture files.
- `parity/scenarios/cli_flow_scenarios.py` — new module: `CliFlowStep`,
  `CliFlowScenario`, `build_cli_flow_scenarios()` (parallels
  `build_cli_scenarios`, still profile-independent-shaped even though
  today it only ever uses the one fixture corpus, for consistency with
  the rest of the suite's "never hardcode a dataset" convention).
- `parity/capture_node_snapshots.py` — new
  `capture_cli_flow_snapshots()`, called from `main()` alongside the
  existing three capture phases. One fresh `_fresh_isolated_index` per
  flow (four total), running each step's real Node CLI invocation via
  the existing `run_node_cli`, writing one JSON file per flow under a new
  `node_ref/<profile>/cli_flow/<flow_name>.json` (a dict of step name →
  extracted result — see Decision 2's snapshot-per-flow-not-per-step
  choice).
- `parity/test_structural.py` — new `fresh_pyqmd_store` fixture, new
  `test_cli_flow_scenario_matches_snapshot`, parametrized the same way
  `pytest_generate_tests` already parametrizes `test_cli_scenario_
matches_snapshot` (profile resolved from `--dataset-config` at
  collection time).
- `src/qmd/cli/commands/embed.py` — the one real code fix (Decision 6).
- `COMMAND_STATUS.md` — flip all eight rows' "Matching parity" column
  from 🟡 unverified to ✅ verified (or 🟡 with a specific noted gap, if
  anything surfaces during implementation that isn't worth fixing
  immediately); correct the `pull` (standalone) row's explanation.
- `docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md` — mark the item
  done once merged.

## Testing / verification plan

1. Write `cli_flow_scenarios.py` + the fixture corpus + the
   `fresh_pyqmd_store` test path first, and get every flow passing
   against **live pyqmd only**, self-consistently (a flow's own
   `list`/`show` steps corroborating its `add`/`remove` steps) — this
   needs no Node/`bun` involvement and is the normal TDD loop.
2. Fix the `embed -c` gap (Decision 6), confirm its new error-path step
   passes.
3. Only once (2) is green: **explicitly ask before running**
   `capture_node_snapshots.py` against the pinned Node commit to produce
   the real `cli_flow` golden snapshots — per this repo's standing rule
   that script is manual/on-demand only. This is the one step in this
   plan that isn't "just write and run pytest."
4. Run `just test-fast` and `just test-parity` (parity suite, which will
   now include the four new flow tests) to confirm nothing regressed.
5. `just lint`.
6. Update `COMMAND_STATUS.md` and the roadmap doc's "Next step" section.

## Out of scope

- `update --pull` / standalone `pull` (Decision 7).
- `collection include`/`exclude` — pickup-order item 2, a distinct,
  not-yet-built feature, not a parity-verification task.
- Any other newly-noticed-but-unrelated gap: if implementation surfaces
  one, it gets documented (a memory note or a `COMMAND_STATUS.md`
  caveat), not silently fixed inline, unless it's as small and directly
  adjacent as Decision 6.
