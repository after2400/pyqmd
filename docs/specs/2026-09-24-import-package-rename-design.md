# Import package and distribution rename (`qmd` → `pyqmd_mlx`, `pyqmd` → `pyqmd-mlx`)

**Date:** 2026-09-24
**Status:** Implemented (2026-09-24)
**Related:** roadmap Backlog entry "PyPI publish prerequisites (other half
of #6)", item 1 (`docs/specs/2026-09-10-python-mlx-rewrite-roadmap.md`,
added by PR #2). The PyPI upload itself stays deferred; this is prep work.

## Decision

Rename the internal import package `qmd` to `pyqmd_mlx` and the
distribution `pyqmd` to `pyqmd-mlx`, in one change, **with no `qmd`
compatibility shim**. The installed command stays `pyqmd`.

### Why

- PyPI `qmd` 0.1.2 (chengzhag/qmd-py) ships a top-level `qmd/` that
  overlaps ours at `qmd/__init__.py` and `qmd/cli/__init__.py`, plus a
  `qmd` console script and an auto-loaded `pytest11` plugin
  (`qmd.testing.contract`). In any shared environment pip silently
  overwrites overlapping files, so which package works depends on
  install/uninstall order. (Per-tool `uv tool install` venvs are
  unaffected, but library/dev environments are not.)
- `pyqmd` is not usable either: PyPI `pyqmd` 0.1.0 (jeffrichley/pyqmd)
  ships a top-level `pyqmd/`.
- Nothing is published and nothing imports this package as a library
  (it is a CLI + MCP server), so the rename is as cheap now as it will
  ever be.

### Rejected alternatives

- **Keep `qmd`** — zero cost now, but ships a generic top-level name
  another PyPI project owns, with order-dependent breakage in shared envs.
- **Rename + `qmd` shim** — self-defeating: the shim re-ships a top-level
  `qmd/`, which is the collision being removed, and there are no external
  importers to protect.

## Naming after the change

| Surface                                                             | Before           | After                        |
| ------------------------------------------------------------------- | ---------------- | ---------------------------- |
| Distribution (`[project].name`, uv tool name, `importlib.metadata`) | `pyqmd`          | `pyqmd-mlx`                  |
| Import package / source dir                                         | `qmd`, `src/qmd` | `pyqmd_mlx`, `src/pyqmd_mlx` |
| Console command (`[project.scripts]` key)                           | `pyqmd`          | `pyqmd` (unchanged)          |

Explicitly **unchanged** (user- or protocol-facing, not import paths):
MCP server name `"qmd"` (Node parity, visible to MCP clients), `qmd://`
URIs, the `qmd:` frontmatter namespace (`_metadata.py`), model ids such
as `qmd-query-expansion-1.7b-mlx`, the index location
(`~/.cache/pyqmd/`), `PYQMD_DB`, commitlint scope names, and prose that
refers to the Node `qmd` tool.

## Mechanics

1. `git mv src/qmd src/pyqmd_mlx` (preserves per-file history).
2. Scripted replacement over `src/`, `tests/`, `parity/`, `scripts/`
   touching only module references:
   - `from qmd.` / `from qmd import` / `import qmd` statements (~233
     lines across 107 `.py` files);
   - quoted dotted module paths `"qmd.<submodule>..."` / `'qmd.<...>'`
     (~201 occurrences: `mock.patch` targets, parity command→module
     tables in `parity/test_structural.py` and
     `parity/scenarios/cli_flow_scenarios.py`);
   - `src/qmd/` and `qmd/<submodule>/` path strings in comments/docs.
     The replacement must not match `qmd://`, `"qmd"` alone,
     `qmd_namespace`, `qmd-query-expansion`, `~/.cache/qmd`, or `pyqmd`.
3. **Guard grep** after the replace: no `\bqmd\.(cli|store|llm|mcp|bench|skills|version)\b`
   and no `src/qmd` may remain anywhere outside the dated specs/plans
   and `CHANGELOG.md`. Dated specs/plans and the changelog are historical
   records and are left as written.
4. Path-depth logic is unaffected: `src/qmd/llm/_constants.py`
   (`parent.parent.parent.parent`) and `src/qmd/cli/commands/skill.py`
   (`parents[2]`) keep the same directory depth after the rename, so only
   their comments change. This avoids a real conflict with the separate
   `DEFAULT_EXPAND_MODEL` fix, whichever lands first.

## Non-Python files

- `pyproject.toml`: `name = "pyqmd-mlx"`; `pyqmd = "pyqmd_mlx.cli.app:main"`;
  `[tool.hatch.version] path = "src/pyqmd_mlx/version.py"`;
  `[tool.hatch.build.targets.wheel] packages = ["src/pyqmd_mlx"]`;
  `[tool.semantic_release] version_variables = ["src/pyqmd_mlx/version.py:__version__"]`.
  Also update the `description` if it names the package.
- `uv.lock`: regenerate (`uv lock`).
- `.Justfile`: `uninstall` recipe → `uv tool uninstall pyqmd-mlx`.
- `tests/cli/test_version_flag.py`: `version("pyqmd-mlx")`.
- `.commitlintrc.mjs`: `src/qmd/...` path comments → `src/pyqmd_mlx/...`
  (scope names unchanged).
- `CLAUDE.md`: path references (`src/qmd/...`, `qmd.store.Store`,
  `qmd/cli/_db.py`), plus one line stating the three names
  (distribution `pyqmd-mlx`, import `pyqmd_mlx`, command `pyqmd`).
- `COMMAND_STATUS.md`, `parity/README.md`: path references.
- `README.md`: one-time migration note for existing editable installs —
  `uv tool uninstall pyqmd && just install` (uv otherwise refuses because
  the old `pyqmd` tool already owns the `pyqmd` executable).
- `src/pyqmd_mlx/skills/pyqmd/references/mcp-setup.md`: the install line
  `pip install pyqmd  # or: uv tool install pyqmd` currently installs
  the unrelated jeffrichley `pyqmd` package. Replace it with an
  install-from-repo instruction (`uv tool install git+<repo URL>`) until
  the PyPI upload happens.

## Coordination

- No open branch edits `src/qmd` (PR #1 / `docs/bundled-skill-fix` is
  already squash-merged as `4fc8fe1`). `claude/next-steps-ee4dd5` is
  docs-only.
- Roadmap Backlog item 1 lives on PR #2's branch
  (`claude/vibrant-rhodes-055d0d`). If PR #2 has merged by the time this
  lands, mark item 1 "decided: renamed" in the same change; otherwise
  report it as a follow-up.

## Commits

1. `refactor: rename import package qmd to pyqmd_mlx` — `git mv`,
   imports, string module paths, pyproject path fields, comments.
2. `build(config): rename distribution to pyqmd-mlx` — `[project].name`,
   `uv.lock`, `.Justfile`, `test_version_flag.py`.
3. `docs(docs): update docs for pyqmd_mlx / pyqmd-mlx rename` —
   CLAUDE.md, COMMAND_STATUS.md, parity/README.md, README.md,
   mcp-setup.md, commitlint comments.

None of these types bumps the version under the current semantic-release
config.

## Verification

- `just test-fast`, `just lint`, `just test-parity` all pass.
- Guard grep (above) is clean.
- `uv build`, then inspect the wheel: it contains `pyqmd_mlx/` and **no**
  top-level `qmd/`; `entry_points.txt` has `pyqmd = pyqmd_mlx.cli.app:main`;
  the dist-info directory is `pyqmd_mlx-<version>.dist-info`; bundled
  `skills/*/SKILL.md` files are present.
- `uv tool uninstall pyqmd && just install`, then `pyqmd --version`,
  `pyqmd skill list`, and a start-and-exit of `pyqmd mcp` (stdio).
  Nothing that writes to the index is run.
