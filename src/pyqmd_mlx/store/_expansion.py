"""Turn the expansion model's raw output lines into typed query parts.

Mirrors Node qmd: LlamaCpp.expandQuery (src/llm.ts) keeps only lex/vec/hyde
lines that mention a query term and falls back to a fixed hyde/lex/vec set
when none survive; store.ts:expandQuery then drops parts identical to the
original query, which hybrid search already runs as-is.
"""

import re

from ._types import ExpandedQueryPart

_PREFIXES = ("hyde:", "lex:", "vec:")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def parse_expanded_lines(lines: list[str]) -> list[ExpandedQueryPart]:
    parts = []
    for line in lines:
        stripped = line.strip()
        for prefix in _PREFIXES:
            if stripped.lower().startswith(prefix):
                parts.append(
                    ExpandedQueryPart(type=prefix[:-1], query=stripped[len(prefix) :].strip())
                )
                break
    return parts


def postprocess_expansion(query: str, lines: list[str]) -> list[ExpandedQueryPart]:
    terms = _NON_ALNUM_RE.sub(" ", query.lower()).split()

    def mentions_query(text: str) -> bool:
        lower = text.lower()
        return not terms or any(term in lower for term in terms)

    parts = [p for p in parse_expanded_lines(lines) if p.query and mentions_query(p.query)]
    if not parts:
        parts = [
            ExpandedQueryPart(type="hyde", query=f"Information about {query}"),
            ExpandedQueryPart(type="lex", query=query),
            ExpandedQueryPart(type="vec", query=query),
        ]
    return [p for p in parts if p.query != query]
