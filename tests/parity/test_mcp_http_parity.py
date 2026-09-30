"""HTTP-transport parity leg: the same MCP scenarios return the same extracts
over Streamable HTTP (/mcp) as in-process via server.call_tool.

Both legs run through the shared `normalize_call_tool_result` so stdio/HTTP
can never drift apart. Seed stores with :memory: + fake LLM fns only --
never real MLX models, never the real index.
"""

import asyncio
import json

import pytest
from mcp.types import CallToolResult
from starlette.testclient import TestClient

from parity._mcp_client import normalize_call_tool_result
from parity.dataset_profile import DatasetProfile
from parity.scenarios.mcp_scenarios import build_mcp_scenarios
from pyqmd_mlx.mcp.server import build_http_app, build_server
from pyqmd_mlx.store import Store

_PORT = 8131
_ORIGIN = f"http://127.0.0.1:{_PORT}"
_APP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


def _fake_rerank(query, documents, model):
    return [0.9 for _ in documents]


def _fake_expand(query, model):
    return [f"lex: {query}", f"vec: {query}"]


@pytest.fixture(scope="module")
def http_profile(tmp_path_factory):
    """A synthetic two-document dataset profile -- self-contained, no scifact
    corpus needed. Bodies carry >=6-word sentences so verbatim_sentence() and
    the top-1 scenario resolve deterministically."""
    base = tmp_path_factory.mktemp("http_parity")
    corpus = base / "corpus"
    corpus.mkdir()
    (corpus / "aaa.md").write_text(
        "# Alpha\n\nAlpha document covers search topics in great depth here.\n"
    )
    (corpus / "bbb.md").write_text(
        "# Bravo\n\nBravo document discusses entirely different subjects at length today.\n"
    )
    queries_file = base / "queries.yaml"
    queries_file.write_text("- Alpha document search topics\n")
    return DatasetProfile(
        name="httpparity", corpus_dir=corpus, queries_file=queries_file, qrels_file=None
    )


@pytest.fixture(scope="module")
def http_scenarios(http_profile):
    scenarios = build_mcp_scenarios(http_profile)
    assert len(scenarios) == 6
    return scenarios


@pytest.fixture(scope="module")
def http_seeded_store(http_profile):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank, expand_fn=_fake_expand)
    store.add_collection(http_profile.name, str(http_profile.corpus_dir))
    for name in ("aaa.md", "bbb.md"):
        body = (http_profile.corpus_dir / name).read_text()
        content_hash = store.hash_content(body)
        store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
        store.insert_document(
            http_profile.name,
            name,
            name,
            content_hash,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        )
        store.index_content(content_hash, body, model="fake-model")
    yield store
    store.close()


def _http_rpc(client, payload, *, origin=_ORIGIN):
    return client.post("/mcp", json=payload, headers={**_APP_HEADERS, "Origin": origin})


def _sse_result(response):
    """Unwrap the single JSON-RPC result message from a stateless SSE response."""
    assert response.status_code == 200
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])["result"]
    raise AssertionError(f"no SSE data message in response: {response.text!r}")


def _http_call_tool(client, name, arguments):
    response = _http_rpc(
        client,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    # The wire dict uses JSON aliases (structuredContent/isError); validating
    # into CallToolResult lets the HTTP leg reuse the exact same normalizer
    # as the in-process leg below.
    return CallToolResult.model_validate(_sse_result(response))


def test_http_handshake_is_stateless():
    """The initialize handshake (stateless: 200 SSE, no session id) -- tool
    equivalence itself, including the status tool, is pinned per-scenario in
    test_http_tools_match_in_process_for_all_scenarios below."""
    store = Store(":memory:")
    app = build_http_app(store, host="127.0.0.1", port=_PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{_PORT}") as client:
        init = _http_rpc(
            client,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "parity", "version": "0"},
                },
            },
        )
        assert init.status_code not in (403, 421)
        # Stateless handshake: 200 SSE, no session id issued or required
        # (matches stateless_http=True).
        assert init.status_code == 200
        assert init.headers.get("mcp-session-id") is None
        assert _sse_result(init)["protocolVersion"] == "2025-03-26"


def test_http_tools_match_in_process_for_all_scenarios(http_scenarios, http_seeded_store):
    app = build_http_app(http_seeded_store, host="127.0.0.1", port=_PORT)
    server = build_server(http_seeded_store)
    with TestClient(app, base_url=f"http://127.0.0.1:{_PORT}") as client:
        init = _http_rpc(
            client,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "parity", "version": "0"},
                },
            },
        )
        assert init.status_code == 200
        notified = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_APP_HEADERS,
        )
        assert notified.status_code == 202

        for scenario in http_scenarios:
            in_process = asyncio.run(server.call_tool(scenario.tool, scenario.arguments))
            over_http = _http_call_tool(client, scenario.tool, scenario.arguments)
            assert normalize_call_tool_result(scenario, over_http) == normalize_call_tool_result(
                scenario, in_process
            ), f"scenario {scenario.name} diverged over HTTP"
