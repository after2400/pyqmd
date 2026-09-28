"""MCP-shaped result formatting for the `query` tool: short docid, snippet
extraction, line-numbered snippet text. Reuses pyqmd_mlx.cli._snippet.extract_snippet
and pyqmd_mlx.cli._output_search.add_line_numbers rather than duplicating them --
different output shape than the CLI's 6 formats (MCP tools return
structured_content plus a text summary, not one of those formats), but
built on the same underlying primitives.
"""

from pyqmd_mlx.cli._output_search import add_line_numbers
from pyqmd_mlx.cli._snippet import extract_snippet
from pyqmd_mlx.store._types import HybridQueryResult


def format_query_result(result: HybridQueryResult, query: str, intent: str | None) -> dict:
    snippet_info = extract_snippet(
        result.body, query, 300, result.best_chunk_pos, len(result.best_chunk), intent
    )
    formatted = {
        "docid": f"#{result.docid}",
        # Deliberately the BARE display_path, not result.file (the
        # qmd://-prefixed field): Node's MCP `query` tool source
        # (src/mcp/server.ts) sets `file: r.displayPath`, confirmed both by
        # source inspection and a live probe against a real running Node
        # MCP server -- unlike the CLI's `--format json` output (which does
        # use the qmd://-prefixed field, see pyqmd_mlx/cli/commands/search.py)
        # and unlike this same MCP server's get/multi_get tools (whose
        # `uri` field is qmd://-prefixed). Node's own MCP surface is
        # genuinely inconsistent between tools here; this matches it
        # exactly rather than assuming query behaves like the CLI. A
        # previous fix applied the CLI's qmd://-prefix fix to this field
        # too, by analogy, before any real Node MCP capture existed to
        # verify it -- that assumption was wrong for this specific field.
        "file": result.display_path,
        "title": result.title,
        "score": round(result.score * 100) / 100,
        "context": result.context,
    }
    if result.metadata:
        formatted["metadata"] = result.metadata
    formatted["line"] = snippet_info.line
    formatted["snippet"] = add_line_numbers(snippet_info.snippet, snippet_info.line)
    return formatted


def format_query_summary(results: list[dict], query: str) -> str:
    if not results:
        return f'No results found for "{query}"'
    plural = "" if len(results) == 1 else "s"
    lines = [f'Found {len(results)} result{plural} for "{query}":\n']
    for r in results:
        lines.append(f"{r['docid']} {round(r['score'] * 100)}% {r['file']} - {r['title']}")
    return "\n".join(lines)
