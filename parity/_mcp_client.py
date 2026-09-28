"""MCP client helpers for capture_node_snapshots.py's MCP-capture phase.

Spawns Node's `qmd mcp` server as a subprocess and speaks the MCP protocol
to it directly, via the `mcp` package's stdio client -- the same package
pyqmd's own server is built on (already a dependency, no separate script
or client/server pair needed). Isolated the same way run_node_cli isolates
CLI invocations: INDEX_PATH/QMD_CONFIG_DIR point at a fresh, private Node
collection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from parity.scenarios.mcp_scenarios import McpScenario


def normalize_call_tool_result(scenario: McpScenario, result: Any) -> dict:
    """Reduces a CallToolResult -- from either a live in-process pyqmd
    server call or a real MCP-protocol client call to Node -- to the plain
    dict scenario.extract() expects. Shared by test_structural.py (pyqmd
    side) and capture_mcp_snapshots (Node side) so the two normalizations
    can never drift apart."""
    if result.is_error:
        return scenario.extract({"is_error": True})
    structured = result.structured_content or {}
    if scenario.tool == "get":
        structured = {"found": True}
    elif scenario.tool == "multi_get":
        structured = {"count": len(result.content)}
    return scenario.extract(structured)


async def capture_mcp_scenarios_async(
    qmd_repo_root: Path, index_path: Path, scenarios: list[McpScenario]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Runs every scenario against a real Node MCP server over stdio.
    Returns (extracted_by_name, raw_by_name) -- raw is the full
    CallToolResult dump, stored alongside the extracted summary for the
    same reason the CLI phase keeps cli_raw/ (see capture_structural_
    snapshots): a future extract-function change shouldn't force another
    full re-embed just to reprocess already-captured output."""
    config_dir = index_path.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(
        command="bun",
        args=["src/cli/qmd.ts", "mcp"],
        cwd=str(qmd_repo_root),
        env={
            **os.environ,
            "INDEX_PATH": str(index_path),
            "QMD_CONFIG_DIR": str(config_dir),
        },
    )

    extracted: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for scenario in scenarios:
                result = await session.call_tool(scenario.tool, scenario.arguments)
                extracted[scenario.name] = normalize_call_tool_result(scenario, result)
                raw[scenario.name] = result.model_dump(by_alias=True, exclude_none=True)
    return extracted, raw
