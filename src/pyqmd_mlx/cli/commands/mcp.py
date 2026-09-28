"""`pyqmd mcp` command: start the MCP server (stdio by default; --http for
the Streamable HTTP transport). A no-op @app.callback() prevents Typer's
single-command-collapse quirk -- documented precedent: pyqmd_mlx/cli/commands/
embed.py and pyqmd_mlx/cli/commands/status.py from sub-project #3. This module
is expected to stay single-command (--http/--port/--host are options, not
a second command), so the callback is permanent, matching those two
modules. app.py registers this command directly on the root app
(app.command("mcp")(mcp.mcp)), not via add_typer, for the same reason
embed/status are -- avoiding a UserWarning on every real invocation once
mounted flat via add_typer.
"""

import os
import sys

import typer

from pyqmd_mlx.mcp.server import PortInUseError, run_http, run_stdio

app = typer.Typer(help="Start the MCP server.")


@app.callback()
def _mcp_callback() -> None:
    """Start the MCP server."""


def _parse_csv_env(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or None


@app.command("mcp")
def mcp(
    http: bool = typer.Option(
        False, "--http", help="Use Streamable HTTP transport instead of stdio."
    ),
    port: int = typer.Option(8181, "--port", help="Port for the HTTP transport."),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Host to bind the HTTP transport to (falls back to QMD_HOST).",
    ),
) -> None:
    """Start the MCP server. Uses stdio by default; pass --http for the Streamable HTTP
    transport. QMD_ALLOWED_ORIGINS/QMD_ALLOWED_HOSTS (comma-separated) extend the HTTP
    transport's loopback-only origin/host guard; QMD_ALLOWED_ORIGINS=* disables it."""
    if http:
        # Precedence (Node qmd.ts/startMcpHttpServer): flag > QMD_HOST > default.
        resolved_host = host or os.environ.get("QMD_HOST", "").strip() or "localhost"
        try:
            run_http(
                host=resolved_host,
                port=port,
                allowed_origins=_parse_csv_env("QMD_ALLOWED_ORIGINS"),
                allowed_hosts=_parse_csv_env("QMD_ALLOWED_HOSTS"),
            )
        except PortInUseError as exc:
            # Startup usage error, not a serving failure: Node's one-liner
            # on stderr + exit 1 (qmd.ts). Raised by run_http, rendered here.
            print(exc, file=sys.stderr)
            raise typer.Exit(1)
    else:
        run_stdio()
