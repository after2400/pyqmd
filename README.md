# qmd (Python/MLX rewrite)

Python/MLX rewrite of [qmd](https://github.com/tobi/qmd) — hybrid search over your markdown
collections. Apple Silicon + MLX only (no GGUF, no cross-platform support).

> **Beta: feedback wanted.** pyqmd is in public beta. Please report problems or surprises in
> [Issues](https://github.com/after2400/pyqmd/issues).

## Why this exists

Reasonable people might ask "why?" I read about [qmd](https://github.com/tobi/qmd), and as a
Python engineer on Apple Silicon my first thought was: is there a Python version of this that
uses MLX models? I couldn't find one, so I built it with fidelity to qmd's behavior as the
primary goal. Where fidelity wasn't feasible, or I found a clear improvement, the difference is
deliberate and tracked in
[COMMAND_STATUS.md](https://github.com/after2400/pyqmd/blob/main/COMMAND_STATUS.md). pyqmd is an
independent port and is not affiliated with qmd.

Somewhere along the way it became as much about building quality software _agentically_ as about
the port itself, and by that I mean not vibe-coded. It's built with multiple coding agents
(primarily [Claude Code](https://claude.com/claude-code), plus [OpenCode](https://opencode.ai)
with Muse Spark models), all using the [obra/superpowers](https://github.com/obra/superpowers)
skills, under my direction. The repo's convention is that agent-written commits carry a model
co-author trailer. Non-trivial changes start as a design doc
([published here](https://github.com/after2400/pyqmd/tree/main/docs/specs)) and a plan, then go
in test-first, and that workflow includes an independent review step before merging. There are
over 9k lines of code and over 20k lines of tests and parity checks, including a suite that
compares results against a pinned copy of Node qmd. Search-quality changes are measured against
pass criteria fixed beforehand, not eyeballed, and the specs record flat or negative results as
plainly as wins. The most recent experiment had its decision rule written down before it ran.

## Install

```sh
uv tool install pyqmd-mlx
```

(or `pipx install pyqmd-mlx`). The package is `pyqmd-mlx`; the command it installs is
`pyqmd` — PyPI's `pyqmd` is an unrelated project. Upgrade with `uv tool upgrade pyqmd-mlx`.

Or build it yourself from a clone:

```sh
git clone https://github.com/after2400/pyqmd && cd pyqmd
just install   # uv tool install --editable .
```

(editable, so pulling new commits in the clone updates the installed `pyqmd` command without
reinstalling)

```sh
just uninstall
```

## Usage

```sh
pyqmd collection add ~/notes --name notes
pyqmd embed
pyqmd query "how do I configure auth"
pyqmd mcp                 # MCP server over stdio
pyqmd mcp --http          # MCP server over Streamable HTTP
pyqmd --help
```

Index lives at `~/.cache/pyqmd/index.sqlite`, distinct from the live Node `qmd`'s
`~/.cache/qmd/index.sqlite` — both can coexist during the transition.

## Optional: heading breadcrumbs

Set `PYQMD_EMBED_BREADCRUMBS=1` to embed markdown chunks together with their section-heading path (`Doc > Section > Subsection`). In a vector-search benchmark on structured documents this found the right passage a few points more often, but after full-text fusion the gain was small and not statistically clear, and we have not measured it through the reranker, so it is off by default. Turning it on (or off again) changes what gets embedded, so run `pyqmd embed` afterwards; `pyqmd status` says so until you do. The setting must be the same everywhere pyqmd runs, including an MCP client's server config, or the server will report documents as pending.

## Performance note: prefer `pyqmd mcp` for repeated queries

Every one-shot CLI call (`search`/`vsearch`/`query`) that touches embeddings pays a real,
per-process MLX model-load cost, and it doesn't get cheaper on repeat launches the way
Node's mmap-backed GGUF loading does (see `parity/README.md`'s "Performance benchmark"
section for why). Measured on an Apple Silicon Mac with a warm disk: `search` 0.4–0.5 s,
`vsearch` about 2.5 s, `query --no-rerank` 3.5–4.5 s, full `query` 9–11 s (a cold disk adds
more on the first run). If you're issuing more than one embedding-backed query, run
`pyqmd mcp` (or `pyqmd mcp --http`) instead of shelling out per query — it loads models once
and serves unlimited calls from the same process, which is the actual equivalent of Node's
cheap repeated invocations here, not a faster load path. Pure-BM25 `search` doesn't touch MLX
at all and stays cheap either way.

## Development

```sh
just develop     # install dev deps + pre-commit hooks
just test-fast   # non-slow suite
just test-slow   # real-model tests (downloads MLX weights)
just lint
```
