# pyqmd MCP Server Setup

## Install

```bash
uv tool install pyqmd-mlx  # the command is `pyqmd`; PyPI's `pyqmd` is an unrelated package
pyqmd collection add ~/path/to/markdown --name myknowledge
pyqmd embed
```

## Configure MCP Client

**Claude Code** (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "pyqmd": { "command": "pyqmd", "args": ["mcp"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "pyqmd": { "command": "pyqmd", "args": ["mcp"] }
  }
}
```

## HTTP mode (shared Mac server)

```bash
pyqmd mcp --http                             # localhost:8181, this Mac only
pyqmd mcp --http --port 8181 --host 0.0.0.0  # reachable from other machines
```

Runs in the foreground (there is no `--daemon`; use launchd or a terminal
multiplexer to keep it alive). Defaults match Node `qmd`: port `8181`,
host `localhost` — which listens on both `127.0.0.1` and `::1`. Host
precedence is `--host` > the `QMD_HOST` env var > `localhost`. Versions
before this change defaulted to `127.0.0.1:8000`; pin `--port 8000` if a
script still expects that.

`POST /mcp` is the MCP endpoint. `GET /health` is a liveness check
(`{"status": "ok", "uptime": <seconds>}`). There is no REST search endpoint;
non-MCP clients speak MCP over `POST /mcp` like everyone else.

| Client           | Connection                   |
| ---------------- | ---------------------------- |
| Claude Code      | HTTP `http://<mac>:8181/mcp` |
| OpenCode CLI     | HTTP `http://<mac>:8181/mcp` |
| Scripts/other    | HTTP `POST /mcp`             |
| Claude Desktop   | stdio `pyqmd mcp`            |
| OpenCode Desktop | stdio `pyqmd mcp`            |

The `<mac>` rows need the server bound off-loopback (`--host 0.0.0.0` or
the Mac's LAN/Tailscale address); the default `localhost` bind only
accepts connections from the Mac itself.

Remote access: endpoints are unauthenticated — put auth (Tailscale
serve or equivalent) in front before exposing off-host. Every request
passes a DNS-rebinding guard that mirrors Node's:

- **Origin** is always checked when present: loopback origins
  (`localhost`, `*.localhost`, `127.0.0.0/8`, `::1`) pass, anything else
  needs an entry in `QMD_ALLOWED_ORIGINS`. Requests without an `Origin`
  (curl, scripts, MCP SDK clients) pass.
- **Host** is checked for a specific bind address (`localhost`, an IP):
  loopback hosts and the bind address itself pass, anything else needs an
  entry in `QMD_ALLOWED_HOSTS`. For a wildcard bind (`0.0.0.0`, `::`) the
  Host check only runs once `QMD_ALLOWED_HOSTS` is set — without it the
  server warns at startup that Host validation is off.
- `QMD_ALLOWED_ORIGINS=*` turns the guard off entirely (with a startup
  warning); only do this behind your own authenticating proxy.

Both env vars are comma-separated. A rejected request gets `403` with a
JSON-RPC `-32003` error body.
pyqmd never runs on Linux (MLX-only); other hosts can only proxy to the Mac.

## Tools

### query

Hybrid keyword + semantic search with reranking.

| Param            | Type      | Description                                                       |
| ---------------- | --------- | ----------------------------------------------------------------- |
| `query`          | string    | The search text.                                                  |
| `limit`          | number?   | Result limit (default 10).                                        |
| `min_score`      | number?   | Minimum score threshold (default 0.0).                            |
| `candidateLimit` | integer?  | Retrieval pool size before reranking, ≥ 1 (omit for the default). |
| `collections`    | string[]? | Restrict to specific collections; omit for all non-excluded ones. |
| `intent`         | string?   | What you're actually looking for — sharpens ranking and snippets. |
| `rerank`         | bool?     | Rerank results (default true).                                    |
| `filter`         | object?   | Metadata filter AST (see SKILL.md).                               |

### get

Retrieve a document's full content.

| Param          | Type    | Description                      |
| -------------- | ------- | -------------------------------- |
| `file`         | string  | Path or `#docid`.                |
| `from_line`    | number? | Start line (1-indexed).          |
| `max_lines`    | number? | Number of lines to return.       |
| `line_numbers` | bool?   | Add line numbers (default true). |

### multi_get

Retrieve multiple documents.

| Param          | Type    | Description                                      |
| -------------- | ------- | ------------------------------------------------ |
| `pattern`      | string  | Comma-separated paths/docids, or a glob pattern. |
| `max_lines`    | number? | Truncate each document to this many lines.       |
| `max_bytes`    | number? | Skip documents larger than this (default 65536). |
| `line_numbers` | bool?   | Add line numbers (default true).                 |

### status

Index health and collections. No parameters.

## Troubleshooting

- **Server not starting**: confirm `pyqmd` is on `PATH` (`which pyqmd`),
  then try running `pyqmd mcp` directly to see any startup error.
- **No results**: `pyqmd collection list` to confirm something's indexed;
  `pyqmd embed` if vector search returns nothing (embeddings may be missing).
- **Slow first query**: expected — MLX models load lazily on first use.
