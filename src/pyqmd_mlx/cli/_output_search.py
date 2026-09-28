"""Search-result output formatting for the CLI: json/csv/files/md/xml/cli.
Ported from formatter.ts's searchResultsTo* functions (formatter.ts:97-227),
operating on pyqmd_mlx.cli._types.DisplayResult instead of store.ts's SearchResult
so this module doesn't need to know whether a row came from search_fts,
search_vec, or query(). The "cli" format here is a plain colorized text
renderer (score color-coded, dim/bold text) -- it does NOT include the
original tool's clickable OSC-8 terminal hyperlinks, editor-URI-template
integration, or --full-path resolution, which are deferred (see design
spec).
"""

import json as _json

from pyqmd_mlx.cli._snippet import extract_snippet
from pyqmd_mlx.cli._types import DisplayResult

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_CYAN = "\x1b[36m"
_YELLOW = "\x1b[33m"
_GREEN = "\x1b[32m"


def escape_csv(value) -> str:
    if value is None:
        return ""
    s = str(value)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def add_line_numbers(text: str, start_line: int = 1) -> str:
    lines = text.split("\n")
    return "\n".join(f"{start_line + i}: {line}" for i, line in enumerate(lines))


def _format_score(score: float) -> str:
    pct = f"{round(score * 100):>3d}%"
    if score >= 0.7:
        return f"{_GREEN}{pct}{_RESET}"
    if score >= 0.4:
        return f"{_YELLOW}{pct}{_RESET}"
    return f"{_DIM}{pct}{_RESET}"


def search_results_to_json(
    results: list[DisplayResult],
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    output = []
    for row in results:
        body_str = row.body or ""
        snippet_info = (
            extract_snippet(body_str, query, 300, row.chunk_pos, None, intent) if body_str else None
        )
        body = row.body if full else None
        snippet = snippet_info.snippet if (not full and snippet_info) else None
        if line_numbers:
            if body:
                body = add_line_numbers(body)
            if snippet:
                snippet = add_line_numbers(snippet)
        entry: dict = {
            "score": round(row.score * 100) / 100,
            "file": row.display_path,
        }
        if row.docid:
            entry["docid"] = f"#{row.docid}"
        if snippet_info:
            entry["line"] = snippet_info.line
        entry["title"] = row.title
        if row.context:
            entry["context"] = row.context
        if row.metadata:
            entry["metadata"] = row.metadata
        if body:
            entry["body"] = body
        if snippet:
            entry["snippet"] = snippet
        output.append(entry)
    return _json.dumps(output, indent=2)


def search_results_to_csv(
    results: list[DisplayResult],
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    header = "docid,score,file,title,context,line,snippet"
    rows = [header]
    for row in results:
        body_str = row.body or ""
        snippet_info = extract_snippet(body_str, query, 500, row.chunk_pos, None, intent)
        content = body_str if full else snippet_info.snippet
        if line_numbers and content:
            content = add_line_numbers(content)
        rows.append(
            ",".join(
                [
                    f"#{row.docid}" if row.docid else "",
                    f"{row.score:.4f}",
                    escape_csv(row.display_path),
                    escape_csv(row.title),
                    escape_csv(row.context or ""),
                    str(snippet_info.line),
                    escape_csv(content),
                ]
            )
        )
    return "\n".join(rows)


def search_results_to_files(results: list[DisplayResult]) -> str:
    lines = []
    for row in results:
        ctx = f',"{row.context.replace(chr(34), chr(34) * 2)}"' if row.context else ""
        docid_prefix = f"#{row.docid}," if row.docid else ""
        lines.append(f"{docid_prefix}{row.score:.2f},{row.display_path}{ctx}")
    return "\n".join(lines)


def search_results_to_markdown(
    results: list[DisplayResult],
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    blocks = []
    for row in results:
        heading = row.title or row.display_path
        body_str = row.body or ""
        content = (
            body_str
            if full
            else extract_snippet(body_str, query, 500, row.chunk_pos, None, intent).snippet
        )
        if line_numbers:
            content = add_line_numbers(content)
        file_line = f"**file:** `{row.display_path}`\n"
        docid_line = f"**docid:** `#{row.docid}`\n" if row.docid else ""
        context_line = f"**context:** {row.context}\n" if row.context else ""
        metadata_line = f"**metadata:** `{_json.dumps(row.metadata)}`\n" if row.metadata else ""
        blocks.append(
            f"---\n# {heading}\n\n{file_line}{docid_line}{context_line}{metadata_line}\n{content}\n"
        )
    return "\n".join(blocks)


def search_results_to_xml(
    results: list[DisplayResult],
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    items = []
    for row in results:
        title_attr = f' title="{escape_xml(row.title)}"' if row.title else ""
        body_str = row.body or ""
        content = (
            body_str
            if full
            else extract_snippet(body_str, query, 500, row.chunk_pos, None, intent).snippet
        )
        if line_numbers:
            content = add_line_numbers(content)
        context_attr = f' context="{escape_xml(row.context)}"' if row.context else ""
        metadata_elem = (
            f"<metadata>{escape_xml(_json.dumps(row.metadata))}</metadata>\n"
            if row.metadata
            else ""
        )
        docid_attr = f' docid="#{row.docid}"' if row.docid else ""
        items.append(
            f'<file{docid_attr} name="{escape_xml(row.display_path)}"{title_attr}{context_attr}>\n'
            f"{metadata_elem}{escape_xml(content)}\n</file>"
        )
    return "\n\n".join(items)


def search_results_to_cli(
    results: list[DisplayResult],
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    blocks = []
    for row in results:
        body_str = row.body or ""
        snippet_info = extract_snippet(body_str, query, 500, row.chunk_pos, None, intent)
        content = body_str if full else snippet_info.snippet
        if line_numbers:
            content = add_line_numbers(content)
        docid_suffix = f" {_DIM}#{row.docid}{_RESET}" if row.docid else ""
        lines = [f"{_CYAN}{row.display_path}{_DIM}:{snippet_info.line}{_RESET}{docid_suffix}"]
        if row.title:
            lines.append(f"{_BOLD}Title: {row.title}{_RESET}")
        if row.context:
            lines.append(f"{_DIM}Context: {row.context}{_RESET}")
        if row.metadata:
            lines.append(f"{_DIM}Metadata: {_json.dumps(row.metadata)}{_RESET}")
        lines.append(f"Score: {_BOLD}{_format_score(row.score)}{_RESET}")
        lines.append(content)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_search_results(
    results: list[DisplayResult],
    format: str,
    query: str = "",
    full: bool = False,
    line_numbers: bool = False,
    intent: str | None = None,
) -> str:
    if format == "json":
        return search_results_to_json(results, query, full, line_numbers, intent)
    if format == "csv":
        return search_results_to_csv(results, query, full, line_numbers, intent)
    if format == "files":
        return search_results_to_files(results)
    if format == "md":
        return search_results_to_markdown(results, query, full, line_numbers, intent)
    if format == "xml":
        return search_results_to_xml(results, query, full, line_numbers, intent)
    if format == "cli":
        return search_results_to_cli(results, query, full, line_numbers, intent)
    raise ValueError(f"Unknown format: {format}")
