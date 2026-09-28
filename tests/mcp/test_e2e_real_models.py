"""Real-model end-to-end validation: collection add -> embed -> query, all
through the real pyqmd CLI (subprocess) and the real MCP stdio server
(subprocess), speaking the actual MCP protocol via the official mcp client
SDK. No fakes/mocks anywhere in this test."""

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _python_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_mcp_query_tool_returns_real_results_via_stdio(tmp_path):
    (tmp_path / "auth.md").write_text(
        "# Authentication\n\nHow to configure authentication for your application."
    )
    db_path = tmp_path / "index.sqlite"
    env = {**os.environ, "PYQMD_DB": str(db_path)}
    repo_root = str(_python_repo_root())

    add_result = subprocess.run(
        ["uv", "run", "pyqmd", "collection", "add", str(tmp_path), "--name", "notes"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert add_result.returncode == 0, add_result.stderr

    embed_result = subprocess.run(
        ["uv", "run", "pyqmd", "embed"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert embed_result.returncode == 0, embed_result.stderr

    async def _run_query():
        server_params = StdioServerParameters(
            command="uv", args=["run", "pyqmd", "mcp"], cwd=repo_root, env=env
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("query", {"query": "authentication", "limit": 5})

    result = asyncio.run(_run_query())

    assert result.is_error is not True
    results = result.structured_content["results"]
    assert len(results) >= 1
    assert results[0]["title"] == "Authentication"
