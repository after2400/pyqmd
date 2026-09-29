# Publish pyqmd to PyPI as `pyqmd-mlx` — design

**Date:** 2026-09-28
**Status:** Implemented (released as 0.7.0)
**Roadmap:** the other half of #6, whose three PyPI prerequisites —
import package rename, HF-hosted expansion model, classifiers/README
metadata — are done.
**Related:** `2026-09-20-github-ci-release-design.md` (the CI and PSR
release workflows this extends); the 2026-09-28 public relaunch (which
PyPI waited for).

## Goal

Every release PSR cuts is also published to PyPI as `pyqmd-mlx`, so
`uv tool install pyqmd-mlx` always installs the latest release. Publishing
uses PyPI trusted publishing: no stored token anywhere.

## Decisions

| Question                   | Decision                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| When does PyPI get a build | On every release. `docs` commits stop releasing, so every remaining release changes something that ships.                                                                                   |
| Where the upload runs      | A job in the existing `release.yml` (not a separate workflow).                                                                                                                              |
| Job layout                 | PyPA's three-way split: `release` (PSR) → `build` (no credentials) → `publish` (credentials, no repo code).                                                                                 |
| Upload tool                | `pypa/gh-action-pypi-publish` — generates PEP 740 attestations by default (`uv publish` doesn't).                                                                                           |
| Credentials                | Trusted publishing (OIDC). PyPI pending publisher, already added by the user: project `pyqmd-mlx`, owner `after2400`, repo `pyqmd`, workflow `release.yml`, environment `pypi`. No secrets. |
| Approval gate              | None for now. A required reviewer can be added to the `pypi` environment later without code changes.                                                                                        |
| Rehearsal                  | No TestPyPI. Instead, the same build-and-check job runs on every PR. TestPyPI can be added later if needed.                                                                                 |
| First version              | Continue 0.x: the implementing PR is `feat(config): …`, so PSR cuts 0.7.0, the first upload. 1.0 stays a later milestone.                                                                   |
| Name reservation           | Not done. PyPI can't reserve a name without an upload; the pending publisher doesn't reserve it. `pyqmd-mlx` was still unclaimed on 2026-09-28.                                             |

Rejected along the way: uploading only `feat`/`fix`/`perf` releases
(needs filtering logic; the conventional fix is to not release on `docs`
at all); on-demand publishing (easy to forget, PyPI drifts); per-push
pre-releases to PyPI (the `pypi` environment would have to allow every
branch); per-push uploads to TestPyPI (PSR prerelease groups plus custom
version stamping, to re-prove an upload path that only needs proving
once); commit SHAs in versions (PyPI rejects `+local` versions, and
`rc`/`dev` numbers must be integers).

## Workflows

### New reusable workflow: `.github/workflows/build.yml`

`on: workflow_call` with inputs `ref` (what to check out) and
`expected-version` (optional). One job, `build`, on **`macos-latest`**
(Apple Silicon — the only platform pyqmd supports), with the same
`UV_PYTHON_PREFERENCE=only-managed` step `ci.yml` uses on macOS:

1. Check out `ref`.
2. `uv build` → `dist/pyqmd_mlx-<v>-py3-none-any.whl` and the sdist.
3. `uvx twine check --strict dist/*` — metadata valid and the README
   renders as the PyPI project page.
4. Smoke test: install the wheel into a fresh `uv tool` environment and
   run `pyqmd --help` and `pyqmd --version`. When `expected-version` is
   set, `--version` must print exactly that version.
   - The install resolves the **latest** dependencies, not `uv.lock`,
     exactly like a user's `uv tool install pyqmd-mlx`. So this is also
     the one CI check that catches a new dependency release breaking
     pyqmd, which the locked test suite can't see.
   - GitHub's macOS runners are VMs without Metal/GPU access, so no CI
     job can run MLX models. The smoke test proves install + entry point
     - CLI startup only, never that the models work. That boundary
       already exists for the test suite (`slow` tests run locally via
       `just test-slow`).
5. Upload `dist/` as a workflow artifact (default 90-day retention).

The whole job runs on macOS: splitting build (Ubuntu) from smoke test
(macOS) would pass artifacts between jobs for no gain — the wheel is
pure Python either way.

### `ci.yml`

A new job `build` that calls `build.yml` with the PR head (or the pushed
`main` commit) and no `expected-version`. It reports as the status check
**`build / build`**, which the user adds to `main`'s required checks.

### `release.yml`: three jobs

- **`release`** — the existing PSR steps, unchanged (checkout with
  `RELEASE_PAT`, `contents: write`, no environment). One new final step
  sets outputs `released` (true/false) and `tag`, by checking whether the
  new `HEAD` carries a `v*` tag.
- **`build`** — `needs: release`, `if: released == 'true'`; calls
  `build.yml` with `ref: <tag>` and `expected-version: <tag without v>`.
  It checks out the **tag**, not `main`: `release` checks out whatever
  `main` is when it starts, and another merge may land in between.
- **`publish`** — `needs: build`, `environment: pypi`, job-level
  permissions `id-token: write` and `contents: read` only. Downloads the
  `dist/` artifact and uploads it with `pypa/gh-action-pypi-publish`
  (pinned to a release tag). No checkout, no repo code, no `RELEASE_PAT`.

Why this is safe with the environment on one job only: an environment is
per job. PyPI's trust check matches both the workflow (`release.yml`) and
the environment (`pypi`) claims, so only `publish` can obtain a PyPI
token; `release` never runs in `pypi`. The `pypi` environment is
restricted to the `main` branch (a `workflow_run` workflow runs in
`main`'s context, so the restriction holds). A reviewer added later
pauses only the upload — the tag and GitHub release already exist by
then. The environment adds a cosmetic "Deployments" panel to the repo
page.

## Release config and commit conventions

- **`pyproject.toml`:** `patch_tags = ["fix", "perf", "chore"]` (drops
  `docs`). A docs-only merge makes no tag, no GitHub release, no upload;
  its commits still appear under "📖 Documentation" in the next real
  release's changelog and notes. PSR keeps `--skip-build` (building now
  lives in `build.yml`). `chore` keeps releasing: `chore(config)` can
  change dependencies, which changes what users install.
- **`CLAUDE.md`** (`AGENTS.md` is a git-tracked symlink to it — edit
  `CLAUDE.md` only, never write `AGENTS.md`):
  - Commit conventions: `docs` commits never release — use them for
    `docs/`, `README.md`, `CLAUDE.md`, `COMMAND_STATUS.md` (a README
    change reaches the PyPI page with the next real release). Anything
    under `src/pyqmd_mlx/` ships in the wheel, including the bundled
    skills' `SKILL.md` and `references/` — those use `fix`/`feat`, never
    `docs`.
  - Status paragraph: "#6 (local install, no PyPI yet)" and "Released as
    0.x Beta on GitHub (install from a tag); PyPI …" become "published to
    PyPI as `pyqmd-mlx`".
- **The implementing PR** squash-merges with the title
  `feat(config): publish releases to PyPI as pyqmd-mlx` (existing scope:
  `config` already covers the release tooling) → 0.7.0.

### Per-commit changelog entries from squash merges

The repo keeps squash-only merging. GitHub's squash default
(`COMMIT_MESSAGES`) lists every branch commit in the squash body, and
PSR 10's conventional parser already splits it
(`parse_squash_commits = True` by default): each branch commit becomes its
own changelog entry in its own section, all linked to the PR. Verified
2026-09-28 by rendering a simulated three-commit squash with the repo's
real templates (PSR 10.6.1): version 0.7.0, one entry each under
Features, Bug Fixes and Documentation. Rejected: merge commits or rebase
merges (per-commit entries too, but the commits don't carry `(#N)`, so
entries lose their PR links; squash also keeps `main` linear and the
`pr-title` check meaningful).

Two template fixes in `.semrel/` (the same kind as PR #11's):

1. **Duplicate entry:** PSR also parses the squash title as a commit, so
   when the PR title equals a branch commit's subject (the normal case)
   the entry appears twice. Drop entries identical in type, scope and
   description within one version.
2. **Separator leak:** GitHub inserts a `---------` line before the
   collected `Co-authored-by` trailers, and it lands in the last entry's
   body. Strip `---------` paragraphs in `format_body_without_footers`
   alongside the `Co-authored-by`/`Signed-off-by` ones.

Commit discipline follows: every commit on a PR branch becomes a
changelog entry, so each must be a meaningful conventional commit (no
"wip"/"fix typo" commits — fold those in before opening the PR).
`CLAUDE.md`'s commit conventions say so. The commitlint hook already
enforces the format locally.

Follow-ups after a PR is open get their own type, `review` (added during
implementation): changes a reviewer asked for, with the usual scopes. Like
`doh` (fixing your own silly mistake), it never bumps the version and is
excluded from the changelog. It covers only code the PR itself adds; a
real bug a review turns up in released code is its own `fix`. Rejected:
using `doh` for review changes (mislabels them), and folding them in with
`--fixup` + force-push (rewrites history the reviewer already saw).

This PR itself is the first multi-commit PR, with six commits:
`feat(config)` (the pipeline, `patch_tags`, `CLAUDE.md`),
`fix(config)` (the two template fixes), `fix(llm)` (the progress bar
below), `docs` (the README), `chore(config)` (the `review` type) and
`docs(specs)` (this spec's public copy).

### Hugging Face progress bar

`pyqmd embed` (and any model load) prints Hugging Face's
"Fetching N files" progress bar even when the model is already cached;
Node prints nothing like it. Hide it when nothing needs downloading, and
keep real download progress on a first run so it doesn't look frozen (if
separating the two isn't straightforward, hiding it for cached models is
the minimum). Commit: `fix(llm)`.

## README and shipped docs

- **`README.md`** (also the PyPI project page; it has no relative links,
  which would break there):
  - Install: `uv tool install pyqmd-mlx`, alternative
    `pipx install pyqmd-mlx`; a note that the package is `pyqmd-mlx` but
    the command is `pyqmd` (PyPI's `pyqmd` is an unrelated project);
    upgrade with `uv tool upgrade pyqmd-mlx`. The pinned
    `git+…@v0.6.3` line goes (it went stale every release).
    "Build it yourself from a clone" (`just install`) stays for
    development.
  - Beta banner: drop "ahead of a PyPI release"; keep "public beta" and
    the feedback request. The "4 - Beta" classifier stays.
  - Performance note: replace "several seconds cold" with measured
    per-stage numbers. Stage timings on 2026-09-28 (Apple Silicon, warm
    disk, 3 runs each): `search` 0.4–0.5 s, `vsearch` 2.5–2.9 s,
    `query --no-rerank` 3.8–4.2 s, `query` 15.2–15.6 s — reranking alone
    is ~11.5 s, traced to `llm/rerank.py` computing logits for every
    prompt position. That got its own `perf(llm)` fix **before** this PR
    (0.6.6: output layer on the last position only, bit-identical scores;
    batching candidates was tried and rejected, no faster and it reordered
    results). The README quotes the numbers measured on 0.6.6 with the
    same per-stage method: `search` 0.4–0.5 s, `vsearch` about 2.5 s,
    `query --no-rerank` 3.5–4.5 s, `query` 9–11 s.
- **`src/pyqmd_mlx/skills/pyqmd/references/mcp-setup.md`** (ships in the
  wheel): its install line
  (`git+https://github.com/after2400/pyqmd  # not on PyPI yet`) becomes
  `uv tool install pyqmd-mlx`, keeping the unrelated-`pyqmd` note.
- **Not in the PR:** `COMMAND_STATUS.md` (tracks command parity, not
  distribution). The roadmap gets updated after 0.7.0 is live
  (#6 complete, "Still open" closed, "Next step" updated).

## Rollout

**Gate before merging (passed 2026-09-28):** the MCP server has been used in a real client
(Claude Code, registered with
`claude mcp add --scope user pyqmd -- pyqmd mcp`) against a real
index — the four tools (`query`, `get`, `multi_get`, `status`), results,
errors, and warm-query speed. Anything broken ships first as normal
`fix` releases. Until now it was only exercised programmatically (parity
suite, HTTP tests, a `slow` e2e test). Result: all four tools worked
(`status`, hybrid `query` after an approved `update` + `embed`, `get` with
docid and line range, `multi_get` with a glob and truncation, a clean
not-found error), and the running server picked up new embeddings without
a restart. Front-matter snippets and line-1 fallbacks match Node's
`extractSnippet`. Findings: the Hugging Face progress bar and the
README's understated timing (both in this PR), and the slow rerank
(its own `perf(llm)` fix first).

**Prerequisite:** the `perf(llm)` rerank fix is merged and released
before this PR merges.

1. PR from a `.claude/worktrees/` worktree. Before opening it, publish
   this spec per `CLAUDE.md`'s workflow: a cleaned copy at
   `docs/specs/2026-09-28-pypi-publishing-design.md` (no plan links, an
   accurate status line), private-info checked, committed on the branch.
   The PR's first CI run exercises `build / build`. The publish path can't
   run from a PR; its first real run is the merge.
2. The user, before merging:
   - creates the `pypi` environment (Settings → Environments), deployment
     branches: `main` only — otherwise GitHub auto-creates it on first
     use with no restriction;
   - after confirming the check name with `gh pr checks`, adds
     `build / build` to the required checks (`gh api -X POST
repos/after2400/pyqmd/branches/main/protection/required_status_checks/contexts`).
3. The user squash-merges with the `feat(config)` title.
4. Release: `release` tags v0.7.0 and creates the GitHub release; `build`
   builds and checks the tag (`--version` must print `0.7.0`); `publish`
   uploads, creating the PyPI project via the pending publisher.
5. Claude verifies and reports:
   - pypi.org/project/pyqmd-mlx shows 0.7.0, the README renders, and
     attestations are listed;
   - `uvx --from pyqmd-mlx pyqmd --version` prints `0.7.0` (a throwaway
     environment; the user's editable install is untouched);
   - `CHANGELOG.md`'s 0.7.0 section has one entry per branch commit (1
     Feature, 2 Bug Fixes, 2 Documentation, 1 Chore), with no duplicate
     entry and no `---------` line.
6. Claude updates the roadmap and removes the worktree.

## Failure handling

| Failure                                           | Result                                                         | Fix                                                                                                               |
| ------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `build` fails on a PR                             | PR blocked; nothing released                                   | Fix in the PR                                                                                                     |
| `build` fails in Release (after tagging)          | Tag and GitHub release exist; nothing on PyPI                  | Fix forward; the next release uploads. That one version is skipped on PyPI (acceptable)                           |
| `publish` fails (e.g. trusted-publisher mismatch) | PyPI refuses before anything is published                      | Fix the PyPI or environment setting, then "Re-run failed jobs" (the `dist/` artifact is kept 90 days; no rebuild) |
| A broken package passes every check               | That version is on PyPI permanently (versions can't be reused) | Yank it on PyPI, release a `fix`                                                                                  |

## Testing

These are workflow and config changes, so the proof is the CI jobs
themselves: `build / build` on the PR, then the Release run on merge.
`tests/test_packaging.py` covers the packaging metadata and entry point,
and pins `patch_tags` (no `docs`) and the `doh`/`review` types (never a
release tag, always excluded from the changelog). PSR proves the
`patch_tags` change for real on the next docs-only merge (no release
happens).

## Out of scope

TestPyPI; a manual "publish tag X" workflow (YAGNI while "Re-run failed
jobs" covers retries); 1.0; a PyPI badge; a required reviewer on `pypi`
(the user can add one any time).
