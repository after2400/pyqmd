# Output-text parity for the mutating CLI commands — design

**Status:** Implemented (2026-09-24)

Roadmap: `2026-09-10-python-mlx-rewrite-roadmap.md`, "Next step" section,
2026-09-24 pickup order, first bullet. Not a numbered sub-project — a
parity-coverage task in the same category as
`2026-09-17-parity-suite-mutating-commands-design.md`, which this builds
directly on.

## Problem

`COMMAND_STATUS.md`'s "Output format" column is 🟡 **unverified** for
`collection add`/`show`/`remove`/`rename`/`update-cmd`/`include`/
`exclude`, `context add`/`list`/`remove`, `update`, and `embed`. Their
_results_ are verified by the parity suite's `CliFlowScenario` flows
(`parity/scenarios/cli_flow_scenarios.py`), but each step's `extract`
function only parses a structured shape out of stdout (exit code, counts,
name sets) — wording, line layout, and stderr are never compared, and the
Node side of each flow was captured as extracted shapes only, with no raw
text kept (unlike the read-only scenarios, which have `cli_raw/`).

Known differences already, from a first read of both sides:

- `collection add`: pyqmd prints `Indexed: 3  Updated: 0  Unchanged: 0
Removed: 0`; Node prints `Creating collection 'x'...`, then
  `Indexed: 3 new, 0 updated, 0 unchanged, 0 removed`, then
  `✓ Collection 'x' created successfully`.
- `embed`: pyqmd prints `Embedded N chunk(s) across N document(s).` /
  `Nothing to embed.`; Node prints `✓ Done! Embedded N chunks from N
documents in <duration>` / `✓ All content hashes already have
embeddings.`.
- Error paths: Node prints a yellow first line to stderr plus a separate
  hint line (e.g. duplicate `collection add`); pyqmd prints one combined
  sentence.

## Reference implementation

Node `src/cli/qmd.ts` at the pinned commit in
`parity/node_ref/scifact/COMMIT.txt` (`8262698`): `collectionAdd`
(~l.1830), `collectionRemove`/`Rename` (~l.1870-1915), `case "collection"`
subcommands incl. `set-update`/`include`/`exclude` (~l.4600-4630),
`context` handlers (~l.1060-1215), `updateCollections` (~l.990-1013),
`vectorIndex`/embed (~l.2210-2290), `cleanup` (~l.5010-5027).

## Findings (avoid re-deriving them)

- **Color is off on both sides whenever output is captured.** Node:
  `useColor = !process.env.NO_COLOR && process.stdout.isTTY` (l.248), so
  every capture is plain text. pyqmd: `_theme.py` wraps `typer.style`,
  which click strips when stdout isn't a TTY (and `CliRunner` isn't
  one). So an automated text comparison can only check plain text;
  color has to be audited from source.
- **stdout and stderr are separable on both sides.** Node's capture
  already records `result.stderr`; pyqmd's `CliRunner` (click 8.5)
  exposes `result.stderr` separately from `result.stdout`.
- **Corpus paths differ between capture and test runs.** Each flow's
  fixture corpus is a fresh `tempfile.mkdtemp` created at
  `build_cli_flow_scenarios()` time, on both the capture side and the
  test side, so any path echoed in output differs run to run. The
  builders currently don't expose which temp dirs they made.
- **`update`/`embed`/`cleanup` are uncolored only because of scope.**
  `_theme.py`'s docstring says roadmap #10 left them "out of this
  roadmap item's scope" — not a deliberate divergence from Node, which
  colors all three (green `✓` lines, bold counts).
- **The capture script has no phase selection.** `main()` always runs
  structural + flow + MCP + quality phases; the quality baseline alone
  takes hours. Re-capturing just the flows needs a new option.

## Decisions

1. **Approach: exact match after normalization, with declared
   per-step exceptions** (chosen over line-subset matching, which is
   subjective and lets drift slip through, and over freezing pyqmd's own
   output as golden text, which guards against pyqmd changing but not
   against pyqmd differing from Node).
2. **Node's wording and layout are the reference.** pyqmd's output text
   changes to match unless a difference falls in one of the three
   declared categories below. Only printed text changes, never behavior:
   if matching Node's text would require a behavior change, the
   difference is declared as an exception and recorded in
   `COMMAND_STATUS.md` rather than "fixed".
3. **Three allowed exception categories**, each declared on the step
   with a reason:
   1. _Node-only machinery_ — lines about things pyqmd doesn't have:
      the GGUF `Model: …` line, the embed-lock busy message, trust
      messages, Node's 100% progress-bar line.
   2. _Deliberate pyqmd supersets_ — extra information pyqmd prints on
      purpose (`collection show`'s `Documents: N` line, pyqmd's
      skipped-files listing on `add`/`update`). Kept; stripped before
      comparison.
   3. _Unavoidable output differences_ — things normalization can't
      reasonably hide (e.g. chunk counts, if pyqmd's chunker splits the
      tiny fixtures differently from Node's).
4. **Command name in hint text is always normalized** (`qmd …` and
   `pyqmd …` both become `<CMD>`), not declared per step — pyqmd is
   deliberately a different command.
5. **Add Node's colors to `update`, `embed`, and `cleanup`** (user
   decision, 2026-09-24), including `cleanup` even though no flow
   exercises it, so the whole CLI's coloring is consistent. `_theme.py`'s
   docstring is updated to drop the "deliberately untouched" note.
6. **Capture before fixing.** The capture-side changes land first
   (without changing any pyqmd output), the user runs a flow-only
   capture, and the fixes are then driven by the real diffs the new test
   reports, not by reading `qmd.ts` alone.

## Components

### 1. Capture-side changes (`parity/capture_node_snapshots.py`)

- `capture_cli_flow_snapshots` additionally writes
  `node_ref/<profile>/cli_flow_raw/<flow>.json`:

  ```json
  {
    "placeholders": { "/var/folders/.../pyqmd-parity-flow-abc": "<CORPUS_1>", "...": "<INDEX_DIR>" },
    "steps": {
      "add_flow_a": { "args": [...], "stdout": "...", "stderr": "...", "exit_code": 0 }
    }
  }
  ```

  Stale files are wiped first, matching the existing `cli_raw/` pattern.
  `placeholders` maps every run-specific path the flow used (each corpus
  dir from the scenario's `corpus_dirs`, plus the isolated index's
  directory) to a stable placeholder.

- New `--phase {all,structural,cli-flow,mcp,quality}` option, default
  `all` (current behavior unchanged). A single-phase run first compares
  the Node checkout's `HEAD` to the commit recorded in the existing
  `COMMIT.txt` and refuses to run on a mismatch — otherwise one profile
  directory could hold snapshots from two different Node commits. A
  single-phase run does not rewrite `COMMIT.txt` (the commit is
  unchanged by construction); `all` still does.

### 2. Scenario-side changes (`parity/scenarios/cli_flow_scenarios.py`)

- `CliFlowScenario` gains `corpus_dirs: list[Path]` — each builder
  records the temp dirs `_new_fixture_corpus()` gave it, in creation
  order, so both sides can map them to `<CORPUS_1>`, `<CORPUS_2>`, ….
- `CliFlowStep` gains two optional fields:
  - `text_subs: list[TextSub]` — a `TextSub(pattern, replacement,
reason)` dataclass (regex, `re.MULTILINE`); applied to both sides'
    already-normalized text, in order. `reason` is required and
    non-empty (asserted at build time).
  - `text_skip_reason: str | None` — when set, the text comparison for
    this step is skipped (pytest `skip` with this reason, so it's
    visible in test output).
- `run_pyqmd_flow` returns raw output too: `{step_name: FlowStepResult}`,
  where `FlowStepResult` is a dataclass `(extracted, stdout, stderr,
exit_code)`. The existing structural test compares
  `{name: r.extracted for name, r in results.items()}` against the
  snapshot, so its assertion is unchanged.
- Once pyqmd prints Node's `Indexed: N new, …` format, the
  pyqmd-specific `_PYQMD_COUNTS_RE` fallback in `_update_counts_shape`
  (and any equivalent in `_embed_counts_shape`) is removed; the
  extractors keep only the Node patterns.

### 3. Normalizer (new `parity/_text_normalize.py`)

One pure function, `normalize(text: str, placeholders: dict[str, str])
-> str`, applied identically to Node and pyqmd text, in this order:

1. Strip ANSI escape sequences, and OSC sequences (`\x1b]…\x07`) —
   defensive; neither side should emit any when captured.
2. Replace every placeholder path, longest first, including its
   `/private`-prefixed and `/private`-stripped forms (macOS's
   `/var` ↔ `/private/var` symlink).
3. Durations (`1.2s`, `350ms`, `2m 3s`, and Node's `formatETA` output
   forms) → `<DURATION>`.
4. Relative times (`… ago`) → `<AGO>`.
5. Docids (`#` + 6 hex) → `#<DOCID>`.
6. Command name in hint text: a `qmd` or `pyqmd` word immediately
   followed by a space and a known subcommand name → `<CMD>` (so
   `qmd://` URIs are untouched).
7. Strip trailing whitespace on every line and trailing blank lines at
   end of output. Leading blank lines, interior blank lines, and line
   order are preserved — they're part of the format.

Carriage-return progress output (`\r…`) is not special-cased here: if
Node's captured stdout contains it, it's handled by a category-1
`text_subs` entry on that step, where the reason is visible.

### 4. Test (`parity/test_structural.py`)

`test_cli_flow_step_text_matches_node`, parametrized per (flow, step).
A module-scoped fixture, `pyqmd_flow_text_runs`, runs each flow at most
once via `run_pyqmd_flow` (with its own `pytest.MonkeyPatch.context()`
and a fresh `Store` in a `tmp_path_factory` dir, since function-scoped
`monkeypatch`/`tmp_path` aren't available at module scope) and caches
the results by flow name; each per-step test reads from that cache. The
text test's parametrization calls `build_cli_flow_scenarios()` itself
and never shares scenario objects with the structural test — each call
creates fresh corpora, and `update_lifecycle` mutates its corpus on disk
via `before`, so running one scenario object's steps twice would give
wrong results. The pyqmd side's placeholder map is that scenario's
`corpus_dirs` plus the fixture's store directory as `<INDEX_DIR>`. For
each step: load
`cli_flow_raw/<flow>.json`, normalize both sides with their own
placeholder maps, apply the step's `text_subs`, and assert stdout and
stderr separately, failing with a `difflib.unified_diff`. Exit codes are
already compared by the existing structural test and are not duplicated
here.

- `cli_flow_raw/` absent → `pytest.skip("no raw flow capture; run
capture_node_snapshots.py --phase cli-flow")`, so the suite stays green
  between this landing and the user's capture run.
- Step absent from the raw file (new step added since capture) → skip
  with a re-capture message, same as above.
- `text_skip_reason` set → `pytest.skip(reason)`.

## Order of work

1. **Setup** (no pyqmd output changes): capture-side `--phase` and
   `cli_flow_raw/`, `corpus_dirs`, `TextSub`/`text_skip_reason`, the
   normalizer with its own unit tests, and the new test (which skips).
2. **User runs the capture** — `uv run python
parity/capture_node_snapshots.py --qmd-repo-root ../qmd --phase
cli-flow` (exact command given at that point). This is never run
   automatically, per `CLAUDE.md`.
3. **Fixes, driven by the new test's diffs**: per step, change pyqmd's
   text to match Node or declare an exception. Separately, the color
   pass on `update`/`embed`/`cleanup`, plus a source-level color audit
   of messages no flow exercises (e.g. `cleanup`'s lines, error paths).

## Testing / verification

- Normalizer: unit tests per rule (ANSI/OSC, placeholders incl.
  `/private`, durations incl. `formatETA` forms, `<AGO>`, docids,
  `<CMD>` not touching `qmd://`, whitespace).
- Existing `tests/cli/` assertions on today's pyqmd wording are updated
  to Node's wording _before_ the code changes (TDD).
- Color: new unit tests invoking the CLI with color forced on
  (`CliRunner().invoke(..., color=True)`) assert the ANSI codes on newly
  colored lines, for `update`, `embed`, and `cleanup`.
- Grep the bundled `src/pyqmd_mlx/skills/*/SKILL.md` files, `README.md`, and
  `scripts/` for quoted output text that goes stale, and update it.
- `just test-fast`, `just lint`, and `just test-parity` green, with the
  new text test passing (not skipping) for every step without a declared
  `text_skip_reason`.

## Docs

- `COMMAND_STATUS.md`: each covered row's "Output format" cell → ✅, or
  🟡 with its declared exceptions listed. Any result-side difference the
  audit turns up is recorded, not hidden.
- `parity/README.md`: document `--phase`, `cli_flow_raw/`, the
  normalizer, and how to declare a `TextSub`/`text_skip_reason`.
- Roadmap "Next step": mark this bullet done.
- Doc/skill edits in this work touch quoted output text only. The import
  package is `pyqmd_mlx` (renamed from `qmd` in PR #3, 2026-09-24), so
  any module or file path added uses `pyqmd_mlx.*` / `src/pyqmd_mlx/…`.

## Out of scope

- Output text of the read-only commands (`search`, `vsearch`, `query`,
  `get`, `multi-get`) — their 🟡 cells stay as they are.
- `mcp --http` parity, `skills` (plural), qrels-margin calibration — each
  is its own spec/plan cycle.
- An automated text check for `cleanup` — no flow exercises it; it gets
  the color pass and a source-level text audit only.
- Re-pinning the Node reference commit.
