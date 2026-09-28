"""Snippet extraction for CLI search-result display: finds the best-
matching line in a document (optionally focused within one chunk) and
extracts a small window of context around it. Ported from store.ts's
extractSnippet (store.ts:5354-5437). Pure function, no DB access, no
dependency on Store -- this is a CLI-only display concern, Store.query()
itself never calls it.
"""

from dataclasses import dataclass

from pyqmd_mlx.store._chunking import CHUNK_SIZE_CHARS
from pyqmd_mlx.store._intent import extract_intent_terms

INTENT_WEIGHT_SNIPPET = 0.3


@dataclass
class SnippetResult:
    line: int
    snippet: str
    lines_before: int
    lines_after: int
    snippet_lines: int


def extract_snippet(
    body: str,
    query: str,
    max_len: int = 500,
    chunk_pos: int | None = None,
    chunk_len: int | None = None,
    intent: str | None = None,
) -> SnippetResult:
    total_lines = len(body.split("\n"))
    search_body = body
    line_offset = 0

    if chunk_pos is not None and chunk_pos >= 0:
        search_len = chunk_len or CHUNK_SIZE_CHARS
        context_start = max(0, chunk_pos - 100)
        context_end = min(len(body), chunk_pos + search_len + 100)
        search_body = body[context_start:context_end]
        if context_start > 0:
            line_offset = len(body[:context_start].split("\n")) - 1

    lines = search_body.split("\n")
    query_terms = [t for t in query.lower().split() if t]
    intent_terms = extract_intent_terms(intent) if intent else []
    best_line = 0
    best_score = -1.0

    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1.0 for term in query_terms if term in line_lower)
        score += sum(INTENT_WEIGHT_SNIPPET for term in intent_terms if term in line_lower)
        if score > best_score:
            best_score = score
            best_line = i

    if chunk_pos is not None and chunk_pos >= 0 and best_score <= 0:
        if chunk_pos == 0:
            # chunkPos=0 may be the chunk selector's initialization default
            # for queries where lexical scoring found no winner -- retry
            # against the full body so the real match isn't missed.
            return extract_snippet(body, query, max_len, None, None, intent)
        context_start = max(0, chunk_pos - 100)
        if chunk_pos > context_start:
            best_line = len(search_body[: chunk_pos - context_start].split("\n")) - 1
        else:
            best_line = 0

    start = max(0, best_line - 1)
    end = min(len(lines), best_line + 3)
    snippet_lines_list = lines[start:end]
    snippet_text = "\n".join(snippet_lines_list)

    if chunk_pos and chunk_pos > 0 and not snippet_text.strip():
        return extract_snippet(body, query, max_len, None, None, intent)

    if len(snippet_text) > max_len:
        snippet_text = snippet_text[: max_len - 3] + "..."

    absolute_start = line_offset + start + 1
    snippet_line_count = len(snippet_lines_list)
    lines_before = absolute_start - 1
    lines_after = total_lines - (absolute_start + snippet_line_count - 1)

    header = (
        f"@@ -{absolute_start},{snippet_line_count} @@ ({lines_before} before, {lines_after} after)"
    )
    snippet = f"{header}\n{snippet_text}"

    return SnippetResult(
        line=line_offset + best_line + 1,
        snippet=snippet,
        lines_before=lines_before,
        lines_after=lines_after,
        snippet_lines=snippet_line_count,
    )
