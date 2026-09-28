# `pyqmd skill` commands — design

**Status:** Implemented (2026-09-18)

Roadmap sub-project #11 (agent-discoverability skill install), added
2026-09-17. Trigger: once pyqmd is published to PyPI, a stranger's
`pip install`ed pyqmd has no equivalent to this repo's own `CLAUDE.md`
teaching an AI coding agent how to drive pyqmd. Node's `qmd` already
solves this with `qmd skill show`/`qmd skill install`.

## Scope decisions (from brainstorming)

- **Singular only.** Node's `qmd skills` (plural) is a _separate_,
  more general subsystem — it discovers and serves any bundled runtime
  skill (Node's repo currently bundles two: `qmd` and `release`) via
  `list`/`get`/`path` subcommands with `--json`. pyqmd has no
  multi-skill ecosystem yet (a future `release` skill is a real,
  already-anticipated item per the roadmap's CI/CD backlog entry, but
  doesn't exist today). Building generic N-item discovery for N=1 is
  speculative regardless of whether N becomes 2 later — there's no
  lock-in cost to adding `pyqmd skills list/get/path` as a small,
  isolated follow-up once a second skill actually exists. Out of scope
  for this spec.
- **Stub install, matching Node's anti-staleness design.** `skill
install` copies the bundled skill tree, then overwrites the copied
  `SKILL.md` with a minimal stub pointing back at `pyqmd skill show` —
  so an installed copy can never drift out of sync with whatever pyqmd
  version is actually running. `references/mcp-setup.md` is the only
  real static content that lands on disk.
- **Claude symlink included.** Node's optional `.claude/skills/qmd` →
  `.agents/skills/qmd` symlink (for Claude-specific discovery) is
  low-risk to port and directly useful, since pyqmd sessions are
  frequently Claude Code sessions.
- **Content is fresh, not ported.** Node's actual `SKILL.md` and
  `references/mcp-setup.md` describe capabilities pyqmd doesn't have
  (`doctor`/`trust`/`init`/`bench`, an MCP `query` tool with a
  `searches: [{type: lex/vec/hyde}]` array, HTTP daemon mode, a
  specific protocol version) — copying either file would produce
  actively wrong documentation. Both get written from scratch against
  pyqmd's real, verified command and MCP tool surface.

## Design

### Command structure

New `src/qmd/cli/commands/skill.py`, a namespaced Typer sub-app
mirroring the `collection`/`context` pattern (`app = typer.Typer()`,
mounted via `app.add_typer(skill.app, name="skill")` in `app.py`) —
not the single-command-app pattern used for `pull`/`cleanup`, since
`skill` has two real subcommands:

- `pyqmd skill show` — prints the bundled `SKILL.md` verbatim.
- `pyqmd skill install [--global] [--force] [--yes]` — flag semantics
  below.

### Packaging

`src/qmd/skills/qmd/SKILL.md` and
`src/qmd/skills/qmd/references/mcp-setup.md`, nested under `src/qmd/`
(not a repo-root `skills/` directory mapped in via Hatchling
`force-include` — considered, but the simpler nested-under-`src`
layout has a working precedent and needs no extra build config).
`pyproject.toml`'s existing `[tool.hatch.build.targets.wheel] packages
= ["src/qmd"]` should include these non-`.py` files automatically —
this is the **first non-`.py` file ever shipped** under `src/`, so
this gets a real verification step (`uv build` + inspect the built
wheel's contents), not an assumption.

### Install semantics

Pure filesystem path helpers (no `Store`/DB involvement) in
`skill.py`:

```python
def get_skill_install_dir(global_: bool) -> Path:
    return Path.home() / ".agents" / "skills" / "qmd" if global_ else Path.cwd() / ".agents" / "skills" / "qmd"

def get_claude_skill_link_path(global_: bool) -> Path:
    return Path.home() / ".claude" / "skills" / "qmd" if global_ else Path.cwd() / ".claude" / "skills" / "qmd"
```

`install`:

1. If the target dir exists and `--force` wasn't given: error `Skill
already exists: <dir> (use --force to replace it)`, exit nonzero,
   no partial write.
2. With `--force`: remove the existing dir/symlink first (handles
   both).
3. Copy the bundled `src/qmd/skills/qmd/` tree (`SKILL.md` +
   `references/`) into the target.
4. Overwrite the copied `SKILL.md` with a generated stub — content:
   a short note that this is a pointer file and the real, always-
   current instructions come from running `pyqmd skill show`.
5. Print `✓ Installed pyqmd skill to <installDir>`.

Claude symlink, after a successful install:

- If `.claude/skills/` (at the relevant scope) already resolves via
  realpath to the same directory as the install target: skip (would
  be a self-referential loop), print a short "already sees the skill
  via …" note.
- Else if `--yes`: create it unconditionally.
- Else if not an interactive TTY: print a one-line tip suggesting the
  symlink and skip (no prompt is possible non-interactively).
- Else: prompt exactly `Create a symlink in <path>? [y/N] `, default
  no on anything but `y`/`yes`.
- If the link path already exists as a different symlink/file and
  `--force` wasn't given: error `Claude skill path already exists:
<path> (use --force to replace it)`.
- Otherwise create a **relative** symlink pointing at the install
  directory (`dir` symlink).

### Content authoring

Prerequisite: this repo's own `CLAUDE.md` "Commands" section (the
source material) is itself stale — missing `collection
include`/`exclude`, `pull`, and `--full-path`, all added earlier this
session. Refresh it first, then derive the skill doc from the
corrected version.

`SKILL.md` covers: the search→retrieve→cite workflow; `search` (BM25)
vs. `query` (hybrid + rerank) vs. `vsearch` (vector-only), with
`query`'s real parameters (`limit`, `min_score`, `collections`,
`intent`, `rerank`, `filter`); `get`/`multi-get` conventions (`#docid`,
`qmd://`, `:from:count` line-range slicing, `--full-path`); the
`--filter` JSON AST syntax; the real MCP tool shapes (below); setup/
maintenance commands (`collection add`, `update`, `embed`, `cleanup`)
explicitly gated "only when the user asked," matching this repo's own
standing rule; a pitfalls checklist.

`references/mcp-setup.md` covers: install (`pip install`/`uv tool
install`) + `collection add`/`embed` quickstart; Claude Code/Claude
Desktop MCP client config JSON pointing at `pyqmd mcp`; a reference
table of the real tool signatures (verified directly against
`src/qmd/mcp/server.py`, not assumed from Node's doc):

- `status()` — no params.
- `get(file, from_line=None, max_lines=None, line_numbers=True)`.
- `multi_get(pattern, max_lines=None, max_bytes=10240,
line_numbers=True)`.
- `query(query, limit=10, min_score=0.0, collections=None,
intent=None, rerank=True, filter=None)`.

— and a troubleshooting section scoped to what pyqmd actually does (no
daemon-mode or protocol-version claims Node's doc makes that pyqmd
doesn't implement).

### Keeping docs in sync

Add a short instruction near `CLAUDE.md`'s "## Commands" section
(alongside the existing "Command parity status" convention, which
tracks Node-vs-pyqmd parity — a different axis): whenever a CLI
command or flag is added, changed, or removed, update `CLAUDE.md`'s
own command list _and_ `src/qmd/skills/qmd/SKILL.md` — both describe
pyqmd's real command surface for different audiences (this repo's own
agent sessions vs. a stranger's installed pyqmd) and must not drift
apart, the way `CLAUDE.md`'s list itself just did.

## Testing plan

1. Path helpers (`get_skill_install_dir`, `get_claude_skill_link_path`):
   unit tests for `--global` vs. local, using `monkeypatch`/`tmp_path`
   so no test touches a real home directory.
2. `skill show`: prints the bundled `SKILL.md` verbatim, exit 0.
3. `skill install`: fresh install creates `.agents/skills/qmd/` with
   the stub `SKILL.md` + the real `references/mcp-setup.md`; an
   existing install without `--force` errors cleanly with no partial
   write; `--force` overwrites; Claude symlink — `--yes` creates it,
   non-interactive without `--yes` skips with a tip (no hanging
   prompt), the self-loop case skips when `.claude/skills/` already
   resolves to the install target, an existing non-matching path
   without `--force` errors.
4. A structural (not prose-content) test asserting the bundled
   `SKILL.md`/`references/mcp-setup.md` exist and are non-empty, so a
   packaging regression gets caught by the fast suite, not just a
   manual build check.
5. Packaging: a real `uv build` + wheel-contents inspection as an
   explicit manual implementation step — not a pytest test, since
   pytest runs against the local editable install and can't catch a
   packaging-config mistake the way inspecting an actual built wheel
   can.
6. No parity-suite coverage — like the `pull` stub, this is a
   pyqmd-native artifact with deliberately fresh (not ported) content,
   so there's no meaningful Node-output comparison to make.
7. Docs: `COMMAND_STATUS.md` gets a new `skill` row (and a `skills`
   row noting deferral, so the gap is tracked rather than silently
   dropped); roadmap doc marks #11 done.

## Out of scope

- `pyqmd skills list/get/path` (the plural, generic multi-skill
  discovery command) — deferred until pyqmd actually has 2+ bundled
  skills (e.g. a future `release` skill).
- Porting Node's `SKILL.md`/`references/mcp-setup.md` content
  verbatim — both describe capabilities pyqmd doesn't have.
- Any change to `collection add --mask`/`--exclude` or other unrelated
  CLI surface.
