"""Dynamic MCP server instructions built from live index state -- gives an
LLM client immediate context about what's searchable without a tool call.
Simplified relative to the reference implementation's buildInstructions
(store.ts): no global-context paragraph, since Store.get_global_context()
doesn't exist yet (deferred since sub-project #2).
"""

from pyqmd_mlx.store import Store


def build_instructions(store: Store) -> str:
    """Snapshot the counts/collection names once, at server startup (called
    from build_server). This snapshot is never refreshed for the lifetime of
    that server process: fine for stdio (a fresh process per client session),
    but a long-lived HTTP server's `instructions` will show stale counts to
    new client sessions after a `pyqmd collection add`/`pyqmd embed` run --
    a known characteristic, not a bug; restart the server to refresh it."""
    counts = store.get_status_counts()
    collections = store.list_collections()

    lines = [
        f"qmd is your local search engine over {counts['active_documents']} markdown documents."
    ]

    if collections:
        names = ", ".join(c["name"] for c in collections)
        lines.append("")
        lines.append(f"Collections (scope with the `collections` parameter): {names}")

    if counts["embedded_vectors"] == 0:
        lines.append("")
        lines.append(
            "Note: no vector embeddings yet -- the query tool's semantic search will fall "
            "back to keyword matching only. Run `pyqmd embed` to enable it."
        )
    elif counts["pending_embed"] > 0:
        lines.append("")
        lines.append(
            f"Note: {counts['pending_embed']} documents need embedding. Run `pyqmd embed` "
            "to update."
        )

    lines.append("")
    lines.append("Tools:")
    lines.append(
        "  - `query` -- hybrid search (keyword + semantic + reranking). Always provide "
        "`intent` to sharpen ranking and snippets."
    )
    lines.append(
        "  - `get` -- retrieve a single document by path or docid (#abc123). Supports a "
        "line-range suffix: `file.md:100` or `file.md:100:40`."
    )
    lines.append("  - `multi_get` -- retrieve multiple documents by comma-separated docids/paths.")
    lines.append("  - `status` -- index status: document counts, collections, embedding coverage.")

    return "\n".join(lines)
