# GitHub CI + automatic releases — design

**Status:** Implemented (2026-09-20)

Covers the roadmap's "CI/CD tooling once a GitHub
remote exists" backlog item, minus PyPI publishing (explicitly out of
scope — no PyPI access yet). CI (lint + fast tests) and release
automation (`python-semantic-release`) land together as one "GitHub
automation" piece: the release job depends on CI passing, so splitting
them would only create process overhead.

## Goals

- Every PR and every push to `main` gets automated `lint` + `test-fast`
  verification on Ubuntu and macOS.
- Every push to `main` with releasable commits automatically produces a
  version bump, CHANGELOG entry, git tag, and GitHub Release — no manual
  release step, no PyPI upload.
- `pyqmd --version` reports the current release version from a single
  source of truth.

## Non-goals

- PyPI publishing (no access; excluded until that changes).
- `test-slow` (real MLX model downloads) and `test-parity` (needs a
  sibling Node `qmd` checkout) in CI — per the roadmap these stay
  manual/scheduled, not PR-blocking.
- Branch protection rules (require checks before merge) — a repo-settings
  step done on GitHub after push, noted as a follow-up, not something CI
  configures itself.
- Commitlint in CI — the local pre-commit `commit-msg` hook already
  enforces it; no need to duplicate.

## CI workflow (`.github/workflows/ci.yml`)

Triggers: `pull_request` (any branch) and `push` to `main`. No schedules,
no manual dispatch, no path filters in the first version.

Two jobs (Approach B from brainstorming):

1. **`lint`** — `ubuntu-latest` only, Python 3.12. Checkout, install `uv`
   via `astral-sh/setup-uv`, `uv sync --dev`, then the exact local
   commands: `uv run ruff check .` and `uv run ruff format --check .`.
   Lint is platform-independent, so running it once avoids wasting a
   macOS runner.
2. **`test`** — matrix `os: [ubuntu-latest, macos-latest]`, Python 3.12
   via `actions/setup-python`. Same checkout + `setup-uv` + `uv sync
--dev` setup, then `uv run pytest -m "not slow"` (exactly `just
test-fast`). `needs: [lint]` so a formatting failure doesn't burn two
   full test runs.

Visible PR checks: `lint`, `test (ubuntu-latest)`, `test
(macos-latest)`. macOS runners are ARM by default, exercising the Apple
Silicon path (MLX/tree-sitter native imports) that Ubuntu can't cover.
`test-fast` excludes `slow`, so no Hugging Face weights are fetched in
CI. Enable uv's built-in cache to keep runs fast. No existing files
change — CI calls the same commands developers run locally.

## Version source of truth

- New file `src/qmd/version.py` containing `__version__ = "0.1.0"` —
  the single version string, updated by semantic-release on each
  release.
- `pyproject.toml`: replace `version = "0.1.0"` with `dynamic =
["version"]` plus `[tool.hatch.version] path =
"src/qmd/version.py"` so builds resolve the version from that file.
- New `--version` flag on the root Typer app printing `__version__`
  (reads from `qmd.version`, not `importlib.metadata`, which can be
  stale or missing in editable installs).
- MCP server's `_package_version()` switches from
  `importlib.metadata.version("qmd")` to `qmd.version.__version__` —
  the package (including MCP) is one version for now.

## Release automation (`python-semantic-release`)

Config lives in `pyproject.toml` (a `[tool.semantic_release]` block;
the tool itself is `uv tool install`ed pinned in the release workflow,
not a dev dependency):

- Version variable: `src/qmd/version.py:__version__`.
- `doh` commit type: non-releasing **and** excluded from the CHANGELOG
  (matches its "never bumps version, never appears in a changelog"
  contract in `.commitlintrc.mjs`).
- Standard 0.x semantics otherwise: `feat` → minor, `fix` → patch.
- Tag format `vX.Y.Z`.

Release workflow (`.github/workflows/release.yml`, separate file so CI
stays readable and each job has its own permissions):

- Triggers on push to `main`, gated on the CI workflow passing.
- Permissions: `contents: write` only (push version-bump commit + tag,
  create the GitHub Release). Nothing else.
- Each release produces all four: version bump in `src/qmd/version.py`,
  CHANGELOG.md entry at repo root, `vX.Y.Z` git tag, GitHub Release
  with the changelog excerpt as notes. No PyPI upload step.
- Changelog rendering uses vendored templates copied verbatim from the
  release-reference-repo precedent (`.semrel/`, verified project-agnostic),
  enabled via `mode = "init"` + `template_dir = ".semrel"` and pinned
  to the same PSR version — re-verify rendering if the pin ever moves.
- No push-back loop: the release commit message carries `[skip ci]`, so
  it triggers neither CI nor another release run.

## Verification

- CI validated by opening a PR (or pushing a scratch branch) and
  confirming `lint`, `test (ubuntu-latest)`, `test (macos-latest)` run
  green.
- Release path validated dry first: semantic-release's no-push/no-commit
  mode shows what version bump and changelog entry the pending commits
  would produce, without cutting a real release. Real end-to-end (tag +
  GitHub Release) happens on the first actual merge to `main` after
  this lands.
- `--version` covered by a CLI test asserting `pyqmd --version` prints
  `qmd.version.__version__`.
- No existing test behavior changes — CI runs exactly the local
  `test-fast` and `lint` commands.

## Follow-ups (not this spec)

- Branch protection on GitHub requiring the three checks before merge.
- `workflow_dispatch`/scheduled `test-slow` (+ `test-parity` once the
  Node repo is reachable from CI) — revisit after basic CI beds in.
- PyPI publishing once access exists (the remaining half of #6).
