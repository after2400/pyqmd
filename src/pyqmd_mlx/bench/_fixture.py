"""Loads and validates a bench fixture JSON file into plain dataclasses.
Field names match Node's src/bench/fixtures/*.json exactly (id/query/
type/description/expected_files/expected_in_top_k per query;
description/version/collection at the top level) so an existing Node
fixture can be reused verbatim -- except for structured lex:/vec:/hyde:/
intent: multi-line queries, which aren't supported (see design spec) and
are rejected here at load time rather than silently mis-scored later."""

import json
from dataclasses import dataclass, field

_STRUCTURED_QUERY_PREFIXES = ("lex:", "vec:", "hyde:", "intent:")
_REQUIRED_QUERY_FIELDS = ("id", "query", "expected_files", "expected_in_top_k")


@dataclass
class BenchQuery:
    id: str
    query: str
    type: str
    description: str
    expected_files: list[str]
    expected_in_top_k: int


@dataclass
class BenchFixture:
    description: str
    version: int
    collection: str | None
    queries: list[BenchQuery] = field(default_factory=list)


def _is_structured_query(query_text: str) -> bool:
    return any(
        line.strip().lower().startswith(_STRUCTURED_QUERY_PREFIXES)
        for line in query_text.splitlines()
        if line.strip()
    )


def load_fixture(path: str) -> BenchFixture:
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid fixture JSON in {path}: {exc}") from exc

    if not isinstance(raw.get("queries"), list):
        raise ValueError(f"Invalid fixture: missing 'queries' array in {path}")

    queries = []
    for i, q in enumerate(raw["queries"]):
        missing = [f for f in _REQUIRED_QUERY_FIELDS if f not in q]
        if missing:
            raise ValueError(
                f"Invalid fixture: query at index {i} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        if _is_structured_query(q["query"]):
            raise ValueError(
                f"Query '{q['id']}' uses structured lex:/vec:/hyde:/intent: syntax, "
                "which pyqmd's bench doesn't support yet. Rewrite it as a single "
                "plain-text query."
            )
        queries.append(
            BenchQuery(
                id=q["id"],
                query=q["query"],
                type=q.get("type", ""),
                description=q.get("description", ""),
                expected_files=q["expected_files"],
                expected_in_top_k=q["expected_in_top_k"],
            )
        )

    return BenchFixture(
        description=raw.get("description", ""),
        version=raw.get("version", 1),
        collection=raw.get("collection"),
        queries=queries,
    )
