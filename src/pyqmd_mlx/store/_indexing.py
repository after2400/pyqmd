"""Directory scanning and document registration for `collection add`/
`update`. A free function (not a Store method), mirroring store.ts's own
boundary: reindexCollection is a free function there too, not a method on
Store -- this keeps filesystem I/O out of Store itself. Ported from
store.ts:1635-1764 (reindexCollection) and store.ts:70-... (splitGlobMask).
Metadata sync (syncDocumentMetadata) is ported too: every scanned document's
frontmatter `qmd.metadata` is synced via store.sync_document_metadata(), on
both `collection add` and `update` -- re-extracted whenever the content
changed, and otherwise only when the stored extraction is missing or from an
older extraction version. Orphaned-content cleanup (cleanupOrphanedContent)
is ported as well -- see the call to store.cleanup_orphaned_content() at the
end of scan_and_register_collection. Titles come from pyqmd_mlx.store._title,
a port of store.ts's per-extension extractors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

from ._title import extract_title
from .store import Store

_EXCLUDE_DIRS = {"node_modules", ".git", ".cache", "vendor", "dist", "build"}


@dataclass
class ReindexResult:
    indexed: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    orphaned_cleaned: int = 0
    skipped: int = 0
    skipped_files: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def split_glob_mask(mask: str) -> list[str]:
    """Split a comma-separated glob mask into individual patterns, treating
    commas inside {...} or [...] as literal (not separators). Ported from
    store.ts's splitGlobMask."""
    parts: list[str] = []
    current = ""
    brace_depth = 0
    bracket_depth = 0
    for ch in mask:
        if ch == "{":
            brace_depth += 1
            current += ch
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1
            current += ch
        elif ch == "[":
            bracket_depth += 1
            current += ch
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
            current += ch
        elif ch == "," and brace_depth == 0 and bracket_depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def scan_and_register_collection(
    store: Store,
    collection_path: str,
    glob_pattern: str,
    collection_name: str,
    ignore_patterns: list[str] | None = None,
) -> ReindexResult:
    """Walk `collection_path`, glob-match files against `glob_pattern`,
    hash and register each as a document in `collection_name`, and
    deactivate documents whose files are gone. Skips hidden files/dirs,
    excluded directories (node_modules/.git/etc), and files resolving
    outside the collection root (symlink escapes)."""
    root = Path(collection_path).resolve()
    now = datetime.now(UTC).isoformat()
    all_ignore = list(ignore_patterns or [])

    patterns = split_glob_mask(glob_pattern)
    seen_relative: set[str] = set()
    result = ReindexResult()

    all_files: set[Path] = set()
    for pattern in patterns:
        all_files.update(p for p in root.glob(pattern) if p.is_file())

    for filepath in sorted(all_files):
        try:
            relative = filepath.relative_to(root)
        except ValueError:
            continue  # defensive: root.glob() should never produce this
        relative_posix = relative.as_posix()

        if any(part.startswith(".") for part in relative.parts):
            continue  # hidden file or inside a hidden directory

        if any(part in _EXCLUDE_DIRS for part in relative.parts[:-1]):
            continue  # inside an excluded directory (node_modules/vendor/etc), any depth

        if all_ignore and any(fnmatch(relative_posix, pat) for pat in all_ignore):
            continue

        resolved = filepath.resolve()
        if resolved != root and root not in resolved.parents:
            result.skipped += 1
            result.skipped_files.append((relative_posix, "OUTSIDE_COLLECTION"))
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as err:
            result.skipped += 1
            result.skipped_files.append((relative_posix, type(err).__name__))
            continue

        if not content.strip():
            continue

        seen_relative.add(relative_posix)
        content_hash = store.hash_content(content)
        title = extract_title(content, relative_posix)
        stat = filepath.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

        existing = store.find_active_document(collection_name, relative_posix)
        if existing:
            document_id = existing["id"]
            if existing["hash"] == content_hash:
                content_changed = False
                if existing["title"] != title:
                    # Same content, new title (a changed extraction rule):
                    # Node's reindexCollection updates it and counts it as
                    # updated, with modified_at set to the scan time.
                    store.update_document_title(existing["id"], title, now)
                    result.updated += 1
                else:
                    result.unchanged += 1
            else:
                store.insert_content(content_hash, content, now)
                store.update_document(existing["id"], title, content_hash, modified_at)
                result.updated += 1
                content_changed = True
        else:
            store.insert_content(content_hash, content, now)
            # st_birthtime (file creation time) exists on macOS/BSD but
            # not on Linux -- fall back to the indexing time, mirroring
            # Node's own `stat ? birthtime : now` fallback (store.ts).
            birthtime = getattr(stat, "st_birthtime", None)
            created_at = (
                datetime.fromtimestamp(birthtime, tz=UTC).isoformat()
                if birthtime is not None
                else now
            )
            document_id = store.insert_document(
                collection_name, relative_posix, title, content_hash, created_at, modified_at
            )
            result.indexed += 1
            content_changed = True

        store.sync_document_metadata(
            document_id, content, relative_posix, only_if_stale=not content_changed
        )

    for path in store.get_active_document_paths(collection_name):
        if path not in seen_relative:
            store.deactivate_document(collection_name, path)
            result.removed += 1

    result.orphaned_cleaned = store.cleanup_orphaned_content()
    return result
