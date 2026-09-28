"""Shared multi-get resolution logic: comma-split a pattern into
docid/path tokens (or, if it contains any glob metacharacter, treat the
whole pattern as a glob), resolve to documents, and apply a --max-bytes
skip-with-reason cutoff. Used by both the CLI's multi-get command
(pyqmd_mlx/cli/commands/documents.py) and the MCP multi_get tool
(pyqmd_mlx/mcp/server.py) so the two surfaces can never drift apart. Returns
raw, untruncated bodies -- line truncation, line numbering, and output
formatting stay a per-caller concern.
"""

from dataclasses import dataclass

from pyqmd_mlx.store import Store


@dataclass
class MultiGetEntry:
    display_path: str
    title: str
    body: str
    skipped: bool = False
    skip_reason: str | None = None
    # Size clause alone ("File too large (12KB > 10KB)"), without the CLI's
    # pointer sentence. The MCP server builds its own skip text from this +
    # the bare path instead of string-surgery on skip_reason (which the CLI
    # owns and must keep byte-identical).
    skip_detail: str | None = None
    not_found: str | None = None  # the raw token, if nothing matched it
    docid: str | None = None


_GLOB_CHARS = ("*", "?", "{")


def _is_glob_pattern(pattern: str) -> bool:
    """Mirrors Node's isCommaSeparated exclusion check (qmd.ts's
    multiGet): a comma-separated list is only treated as such when it
    contains none of these characters; otherwise (including a bare
    pattern with no comma at all) the whole string is one glob pattern.
    A pattern with both a comma and a glob metacharacter is therefore
    treated as one glob too, which in practice matches nothing -- ported
    as-is (see the design spec's findings) rather than "improved"."""
    return any(c in pattern for c in _GLOB_CHARS)


def _entry_from_doc(doc: dict, max_bytes: int) -> MultiGetEntry:
    # qmd://-prefixed, matching every other surface's real Node output
    # (get's own header, search/vsearch/query's `file` field) -- this
    # was previously a bare "collection/path" string, the same class of
    # bug as search/vsearch/query's once-bare `file` field.
    display_path = f"qmd://{doc['collection']}/{doc['path']}"
    docid = doc["hash"][:6]
    body_length = len(doc["doc"].encode("utf-8"))
    if body_length > max_bytes:
        size_clause = f"File too large ({body_length // 1024}KB > {max_bytes // 1024}KB)"
        return MultiGetEntry(
            display_path=display_path,
            title=doc["title"],
            body="",
            skipped=True,
            skip_reason=(f"{size_clause}. Use 'pyqmd get {display_path}' to retrieve."),
            skip_detail=size_clause,
            docid=docid,
        )
    return MultiGetEntry(
        display_path=display_path, title=doc["title"], body=doc["doc"], docid=docid
    )


def resolve_multi_get(store: Store, pattern: str, max_bytes: int) -> list[MultiGetEntry]:
    if _is_glob_pattern(pattern):
        return [_entry_from_doc(doc, max_bytes) for doc in store.find_documents_by_glob(pattern)]

    names = [s.strip() for s in pattern.split(",") if s.strip()]
    entries: list[MultiGetEntry] = []
    for name in names:
        doc = store.find_document_by_identifier(name)
        if doc is None:
            entries.append(MultiGetEntry(display_path=name, title="", body="", not_found=name))
            continue
        entries.append(_entry_from_doc(doc, max_bytes))
    return entries
