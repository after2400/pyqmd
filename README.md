# qmd (Python/MLX rewrite)

Python/MLX rewrite of [qmd](https://github.com/tobi/qmd) — hybrid search over your markdown
collections. Apple Silicon + MLX only (no GGUF, no cross-platform support).

> **Beta: feedback wanted.** pyqmd is in public beta ahead of a PyPI release. Please report
> problems or surprises in [Issues](https://github.com/after2400/pyqmd/issues).

## Install

```sh
uv tool install git+https://github.com/after2400/pyqmd@v0.6.3
```

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

## Performance note: prefer `pyqmd mcp` for repeated queries

Every one-shot CLI call (`search`/`vsearch`/`query`) that touches embeddings pays a real,
per-process MLX model-load cost — measured at several seconds cold, and it doesn't get
cheaper on repeat launches the way Node's mmap-backed GGUF loading does (see
`parity/README.md`'s "Performance benchmark" section for why). If you're issuing more than
one embedding-backed query, run `pyqmd mcp` (or `pyqmd mcp --http`) instead of shelling out
per query — it loads models once and serves unlimited calls from the same process, which is
the actual equivalent of Node's cheap repeated invocations here, not a faster load path.
Pure-BM25 `search` doesn't touch MLX at all and stays cheap either way.

## Development

```sh
just develop     # install dev deps + pre-commit hooks
just test-fast   # non-slow suite
just test-slow   # real-model tests (downloads MLX weights)
just lint
```
