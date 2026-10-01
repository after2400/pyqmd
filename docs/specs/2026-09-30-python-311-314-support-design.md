# Python 3.11–3.14 support — design

**Status:** Implemented (2026-09-30)

**Related:** the roadmap's "Python 3.11 support" backlog entry (added
2026-09-29); `2026-09-28-pypi-publishing-design.md` (the build and smoke-test
workflow this changes).

## Problem

pyqmd requires Python 3.12 (`requires-python = ">=3.12"`), and CI tests only
3.12. Python 3.11 is supported upstream until October 2027, so a user on 3.11
can't install pyqmd at all. 3.13 and 3.14 are allowed by `requires-python`
but nothing tests them or says they work.

## Findings (spike, 2026-09-30)

Measured on a scratch copy of `main` at `87c9243`, on Apple Silicon.

**Dependencies.** `uv pip compile --only-binary :all:` for macOS 15 arm64
resolves every dependency from wheels on 3.11, 3.13, and 3.14. 3.13 and 3.14
get exactly 3.12's versions. 3.11 differs only in numpy (2.4.6 against 2.5.3)
and scipy (1.17.1 against 1.18.1), whose newest releases dropped 3.11. mlx,
mlx-lm, mlx-embeddings, sqlite-vec, and the tree-sitter grammars all ship
3.11 wheels.

**Code.** `ruff check --target-version py311` finds one 3.12-only construct:
the f-string at `src/pyqmd_mlx/cli/commands/update.py:152`, which reuses its
outer quote inside a replacement field (`f"...{coll["pattern"]}..."`, PEP 701).
On 3.11 it is a `SyntaxError` at import, so `update` (and anything importing
the CLI app) fails. A grep for 3.12-only stdlib APIs (`itertools.batched`,
`Path.walk`, `typing.override`) and PEP 695 syntax found nothing.

**Tests.** With that line rewritten:

| Python  | fast suite (`-m "not slow"`) | slow suite (real MLX models) | sqlite `enable_load_extension` |
| ------- | ---------------------------- | ---------------------------- | ------------------------------ |
| 3.11.16 | 1278 passed, 15 skipped      | 12 passed                    | yes                            |
| 3.13.13 | 1278 passed, 15 skipped      | 12 passed                    | yes                            |
| 3.14.5  | 1278 passed, 15 skipped      | 12 passed                    | yes                            |

3.11 used Homebrew's build; 3.13 and 3.14 used uv-managed builds, the kind
CI installs.

The wheel is `py3-none-any`, so one build serves every Python version.

## Design

### 1. Package metadata (`pyproject.toml`)

- `requires-python = ">=3.11"`. It stays open-ended, the usual convention: a
  future Python that breaks is caught by CI or a user report, not refused in
  advance.
- Classifiers: `Programming Language :: Python :: 3.11`, `3.12`, `3.13`, and
  `3.14`, in place of the single `3.12`.
- `[tool.ruff] target-version = "py311"`, so lint rejects any new 3.12-only
  syntax. `ruff format --check` must still pass unchanged.

### 2. The f-string (`cli/commands/update.py:152`)

Bind `coll["pattern"]` to a local before the f-string. The output is
byte-identical; the existing `update` text-parity tests cover it.

### 3. Lockfile

Run `uv lock` after the `requires-python` change. The lock gains split
resolutions for numpy and scipy (3.11 against 3.12+); nothing else moves.
Local development keeps `.python-version` at 3.12.

### 4. CI (`ci.yml`)

The `test` job's matrix becomes `os: [ubuntu-24.04, macos-latest]` ×
`python: ["3.11", "3.12", "3.13", "3.14"]` (8 jobs), with
`fail-fast: false` so one version's failure doesn't hide the others'.
`setup-uv`'s `python-version` takes `${{ matrix.python }}`; it sets
`UV_PYTHON`, which overrides `.python-version` for `uv sync`. The macOS
`only-managed` step is unchanged. `lint` stays a single 3.12 job, since
ruff's target version, not the interpreter, decides what it checks.

### 5. Build and smoke test (`build.yml`)

Today one job builds, runs `twine check`, smoke-tests the wheel on 3.12, and
uploads the `dist` artifact. A matrix over that job would upload `dist` four
times, so it splits in two:

- `build` (macOS, 3.12): `uv build`, `twine check --strict`, upload `dist`.
- `smoke` (`needs: build`, macOS, matrix `python` over the four versions,
  `fail-fast: false`): download `dist`, then the existing smoke step with
  `uv tool install --python ${{ matrix.python }} dist/*.whl`, `--help`, and
  the `--version` check against `expected-version`.

The smoke test installs the newest dependencies without `uv.lock`, so it is
what catches a future dependency release dropping a Python version. Because
`release.yml`'s `publish` job needs the whole reusable `build` workflow, a
smoke failure on any version blocks a release. That is intended: the fix is
a pin or dropping that version, decided when it happens.

`release.yml` doesn't change; its Python only runs PSR.

### 6. Docs

- Roadmap: the "Python 3.11 support" entry is marked done with this spec's
  numbers, and the 2026-09-29 pickup-order paragraph stops calling embedding
  input parity "in progress" (it merged as #16).
- No other doc states a Python version (`CLAUDE.md`, `README.md`,
  `COMMAND_STATUS.md`, and the bundled `SKILL.md` were checked), so none
  changes. PyPI shows the new `requires-python` and classifiers.

## Commits and release

Released as a minor (0.8.0), decided by the owner on 2026-09-30.

1. `feat(config): support Python 3.11 through 3.14`: `pyproject.toml`,
   `update.py`, `uv.lock`.
2. `ci: test and smoke-test the wheel on Python 3.11 through 3.14`:
   `ci.yml`, `build.yml`.
3. `docs(specs): publish the Python 3.11–3.14 support spec`.

## Testing

- Locally, on each of 3.11–3.14: the fast and slow suites, as in the spike
  but against the branch rather than a patched copy, plus `just lint`.
- CI on the PR: 8 test jobs and 4 smoke jobs, all green.
- `just test-parity` on 3.12 as usual; it doesn't run per version.

### Results (2026-09-30)

On the branch, with `uv.lock` relocked, on Apple Silicon with uv-managed
builds of every version:

| Python  | fast suite (`-m "not slow"`) | slow suite (real MLX models) |
| ------- | ---------------------------- | ---------------------------- |
| 3.11.15 | 1283 passed, 10 skipped      | 12 passed                    |
| 3.12.13 | 1283 passed, 10 skipped      | 12 passed                    |
| 3.13.13 | 1283 passed, 10 skipped      | 12 passed                    |
| 3.14.5  | 1283 passed, 10 skipped      | 12 passed                    |

The counts differ from the spike's (1278 passed, 15 skipped) only because
the branch's checkout finds a Node `qmd` checkout, which five tests need;
they're the same on every version. The relock added only numpy 2.4.6 and
scipy 1.17.1 for 3.11; no other package moved. The wheel, installed with
the newest dependencies via `uv tool install`, starts and prints its version
on 3.11 and 3.14.

## Out of scope

- Python 3.10 or older: 3.10 reaches end of life in October 2026.
- Running the slow suite in CI: GitHub's macOS runners have no Metal access,
  the same limit 3.12 has today.
- Free-threaded (3.13t/3.14t) builds.
- Changing the development default in `.python-version`.
