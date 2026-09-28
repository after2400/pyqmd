"""MCP tool scenario definitions. Node's MCP surface is exactly 4 tools --
query/get/multi_get/status -- identical to pyqmd's (verified against
src/mcp/server.ts's registerTool calls), so every scenario here is written
once and applies to both sides unchanged.

Scenarios are built from the active DatasetProfile rather than hardcoded
(a 2026-09-13 parity-suite review finding).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from parity.dataset_profile import DatasetProfile
from parity.scenarios._profile_fixtures import known_docs, sample_query, verbatim_sentence


@dataclass
class McpScenario:
    name: str
    tool: str
    arguments: dict
    extract: Callable[[dict], object]


def _query_invariants_shape(structured_content: dict) -> dict:
    """`query` is embedding-backed (hybrid FTS+vector+rerank), so which
    documents land in the top-K is model-dependent -- same reasoning as
    cli_scenarios._embedding_search_invariants_shape. Assert only result
    count, never exact document-set membership. Deliberately does NOT check
    a qmd:// URI shape: Node's MCP `query` tool emits the bare display path
    in its `file` field (confirmed via source and a live probe against a
    real Node MCP server -- unlike the CLI's --format json output, and
    unlike this same MCP server's get/multi_get tools), so there is no
    URI-shape invariant to assert here."""
    results = structured_content.get("results", [])
    return {"count": len(results)}


def _query_top1_shape(structured_content: dict) -> dict:
    """Used with a verbatim-sentence query: any competent embedding model
    should rank a document first against its own exact text, regardless of
    backend."""
    results = structured_content.get("results", [])
    return {"top_file": results[0].get("file", "") if results else None}


def _get_shape(structured_content: dict) -> dict:
    # get's CallToolResult carries an EmbeddedResource, not structured_content
    # -- see Task 8/10 for how the caller normalizes this into the same
    # {"found": bool, "uri": ...} shape regardless of transport.
    return {"found": structured_content.get("found", False)}


def _multi_get_shape(structured_content: dict) -> dict:
    return {"count": structured_content.get("count", 0)}


def _status_shape(structured_content: dict) -> dict:
    """pyqmd's `status` tool nests document counts under a "counts" key
    ({"active_documents": N, ...} -- see pyqmd_mlx/mcp/server.py's _status_impl).
    Node's real MCP status tool uses flat top-level camelCase fields
    instead (totalDocuments, needsEmbedding, ...), confirmed against a live
    Node MCP server -- a genuine cross-implementation naming difference,
    not a bug on either side. This previously only checked pyqmd's own
    shape, so it silently returned False for Node's real (differently-
    shaped) response no matter how many documents existed -- invisible
    until a real Node MCP capture existed to expose it."""
    counts = structured_content.get("counts")
    active = (
        counts.get("active_documents", 0)
        if counts is not None
        else structured_content.get("totalDocuments", 0)
    )
    return {"has_active_documents": active > 0}


def build_mcp_scenarios(profile: DatasetProfile) -> list[McpScenario]:
    query = sample_query(profile)
    doc_a, doc_b = known_docs(profile)
    sentence = verbatim_sentence(profile, doc_a)

    return [
        McpScenario(
            "mcp_query_finds_sample_query", "query", {"query": query}, _query_invariants_shape
        ),
        McpScenario(
            "mcp_query_top1_matches_verbatim_sentence",
            "query",
            {"query": sentence},
            _query_top1_shape,
        ),
        # Node's hybrid `query` (unlike lexical-only `search`) always falls
        # back to nearest-neighbor vector matches even for a nonsense term
        # -- confirmed via a real capture, this genuinely returns a full
        # page of results with real scores on both sides, not zero. Named
        # accordingly rather than "no_results".
        McpScenario(
            "mcp_query_nonsense_term_still_returns_hybrid_fallback",
            "query",
            {"query": "zzqqxx_no_such_term_zzqqxx"},
            _query_invariants_shape,
        ),
        McpScenario("mcp_get_known_document", "get", {"file": doc_a}, _get_shape),
        McpScenario(
            "mcp_multi_get_two_documents",
            "multi_get",
            {"pattern": f"{doc_a},{doc_b}"},
            _multi_get_shape,
        ),
        McpScenario("mcp_status_reports_health", "status", {}, _status_shape),
    ]
