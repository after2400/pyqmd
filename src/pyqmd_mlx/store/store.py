"""Store: the pyqmd_mlx.store storage/search engine. Methods contain the actual
logic (schema, indexing, FTS, vector search, hybrid RRF query) -- see the
design spec's architecture decision for why this is a class with logic in
methods rather than a facade over module-level functions.
"""

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, resolve_expand_model
from pyqmd_mlx.llm import embed as _default_embed
from pyqmd_mlx.llm import expand_query as _default_expand_query
from pyqmd_mlx.llm import rerank as _default_rerank

from ._schema import load_sqlite_vec, migrate

if TYPE_CHECKING:
    from ._metadata import MetadataExtractionResult
    from ._metadata_filter import MetadataFilter
    from ._types import HybridQueryResult, RankedResult, SearchResult

STRONG_SIGNAL_MIN_SCORE = 0.85
STRONG_SIGNAL_MIN_GAP = 0.15
RERANK_CANDIDATE_LIMIT = 40

# search_vec: a collection-scoped or metadata-filtered search cannot push
# its scoping into the sqlite-vec MATCH operator (no join-safe predicate
# there), so a naive global ANN scan followed by a post-filter can silently
# starve small eligible sets -- their documents never make it into the
# global top `limit*3` in the first place (see store.ts:4196-4203, issues
# #791, #803, and the equivalent metadata-filter case at store.ts:4277-4282).
# Below this many eligible vectors (from either `collection` or `filter`, or
# both combined) we exact-scan the eligible set's own vectors instead of
# relying on ANN; above it we fall back to ANN with a larger over-fetch,
# capped at sqlite-vec's max k of 4096.
FILTERED_VEC_EXACT_SCAN_MAX = 20_000
VEC_HASH_SEQ_IN_CHUNK = 400
# Generic chunk size for Python-list-driven WHERE ... IN (...) / OR clauses
# (e.g. per-hash or per-doc-id deletes) -- keeps SQLite's expression-tree
# depth well under its default limit of 1000 regardless of collection size.
SQL_IN_CHUNK_SIZE = 400

DEFAULT_BUSY_TIMEOUT_MS = 120_000
BUSY_TIMEOUT_ENV_VAR = "QMD_SQLITE_BUSY_TIMEOUT"


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _chunked(items: list, size: int) -> list[list]:
    """Split `items` into consecutive chunks of at most `size` elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_context_map(raw: str | None) -> dict[str, str]:
    """Parse the collections.context column's JSON-encoded {path_prefix:
    text} map. A JSONDecodeError -- a legacy raw string from
    add_collection's own context parameter, which predates this map format
    and is never written by any CLI command -- is treated as an empty map
    rather than propagating: malformed data in one collection's context
    column must not crash status/get/search for the whole index."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _escape_like(value: str) -> str:
    """Escape a value for safe use inside a `LIKE ? ESCAPE '\\'` clause, so
    that literal '%'/'_'/'\\' characters in `value` are matched literally
    rather than as SQL LIKE wildcards."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _resolve_busy_timeout_ms() -> int:
    """Read the busy-timeout override from QMD_SQLITE_BUSY_TIMEOUT (milliseconds;
    '0' restores SQLite's fail-fast default), falling back to
    DEFAULT_BUSY_TIMEOUT_MS when unset, empty, or not a valid non-negative
    integer."""
    raw = os.environ.get(BUSY_TIMEOUT_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_BUSY_TIMEOUT_MS
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_BUSY_TIMEOUT_MS
    return parsed if parsed >= 0 else DEFAULT_BUSY_TIMEOUT_MS


def _enable_wal(conn: sqlite3.Connection, budget_ms: int) -> None:
    """Switch `conn` to WAL journal mode, retrying on 'database is locked'
    within `budget_ms`. Migrating the journal mode itself needs a brief
    exclusive lock that does NOT honor busy_timeout, so concurrent first-time
    opens of a cold database can raise this even with busy_timeout already
    set (ported from the Node reference's src/db.ts::enableWal). Once a
    database is already in WAL mode this pragma is a cheap no-op that never
    contends. On an in-memory database this pragma silently stays 'memory'
    (never raises), so this function is always safe to call unconditionally."""
    deadline = time.monotonic() + max(budget_ms, 0) / 1000
    attempt = 0
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(min(0.005 + attempt * 0.001, 0.025))
            attempt += 1


@dataclass(frozen=True)
class CollectionRemoval:
    """What Store.remove_collection_detailed deleted: every document row
    in the collection (active or not) and the content hashes that became
    orphaned as a result -- the two numbers `collection remove` prints.
    Node's own "Deleted N documents" is bun:sqlite's `changes` for the
    DELETE, which also counts trigger-cascaded FTS/metadata rows; this is
    the real document count."""

    deleted_docs: int
    cleaned_hashes: int


class Store:
    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        embed_fn=None,
        rerank_fn=None,
        expand_fn=None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        expand_model: str | None = None,
    ):
        self.db_path = db_path
        # check_same_thread=False: every other caller (CLI, the full test
        # suite) is single-threaded, so this check never fires for them --
        # it only matters for the MCP server (pyqmd_mlx/mcp/server.py), whose
        # `_locked` mutex already guarantees only one thread touches this
        # connection at a time. That mutex provides the actual safety
        # guarantee (no concurrent access); this flag just removes sqlite3's
        # redundant, stricter *thread-identity* check, which otherwise
        # raises the moment a different thread touches the connection at
        # all -- even under external serialization. See
        # tests/store/test_concurrency.py for the regression test.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        busy_timeout_ms = _resolve_busy_timeout_ms()
        self.conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        _enable_wal(self.conn, busy_timeout_ms)
        load_sqlite_vec(self.conn)
        migrate(self.conn)

        # Real pyqmd_mlx.llm calls by default; tests inject fakes here so storage
        # logic can be exercised without loading MLX models.
        self._embed_fn = embed_fn or _default_embed
        self._rerank_fn = rerank_fn or _default_rerank
        self._expand_fn = expand_fn or _default_expand_query
        self._embed_model = embed_model or DEFAULT_EMBED_MODEL
        self._rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self._expand_model = expand_model or resolve_expand_model()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def wrapping_llm_fns(self, expand=None, rerank=None):
        """Temporarily replace the expand/rerank functions with wrappers of
        the current ones. Each argument maps the current function to its
        replacement. Both are restored on exit, including on error. For
        single-threaded callers (bench --samples); the MCP server never uses
        it."""
        saved = (self._expand_fn, self._rerank_fn)
        try:
            if expand is not None:
                self._expand_fn = expand(self._expand_fn)
            if rerank is not None:
                self._rerank_fn = rerank(self._rerank_fn)
            yield
        finally:
            self._expand_fn, self._rerank_fn = saved

    def add_collection(
        self,
        name: str,
        path: str,
        pattern: str = "**/*.md",
        ignore_patterns: str | None = None,
        include_by_default: bool = True,
        update_command: str | None = None,
        context: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO collections
                (name, path, pattern, ignore_patterns, include_by_default, update_command, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                path,
                pattern,
                ignore_patterns,
                int(include_by_default),
                update_command,
                context,
            ),
        )
        self.conn.commit()

    def get_collection(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def list_collections(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM collections ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    @property
    def embed_model(self) -> str:
        """The embedding model this Store embeds and searches with."""
        return self._embed_model

    def remove_collection_detailed(self, name: str) -> CollectionRemoval | None:
        """Remove a collection and cascade-delete its documents, matching
        store.ts's actual (destructive) behavior (store.ts:3698-3715) --
        this is NOT metadata-only. Also cleans up content/content_vectors/
        vectors_vec rows that become orphaned as a result, and the affected
        documents_fts rows (store.ts relies on DB triggers for FTS cleanup;
        Store's explicit-sync design needs this done directly)."""
        doc_rows = self.conn.execute(
            "SELECT id FROM documents WHERE collection = ?", (name,)
        ).fetchall()
        doc_ids = [row["id"] for row in doc_rows]
        cleaned = 0

        if doc_ids:
            for chunk in _chunked(doc_ids, SQL_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                self.conn.execute(
                    f"DELETE FROM documents_fts WHERE rowid IN ({placeholders})", chunk
                )
                self.conn.execute(
                    f"DELETE FROM document_metadata_values WHERE document_id IN ({placeholders})",
                    chunk,
                )
                self.conn.execute(
                    f"DELETE FROM document_metadata WHERE document_id IN ({placeholders})", chunk
                )
            self.conn.execute("DELETE FROM documents WHERE collection = ?", (name,))
            cleaned = self.cleanup_orphaned_content()

        cursor = self.conn.execute("DELETE FROM collections WHERE name = ?", (name,))
        self.conn.commit()
        if cursor.rowcount == 0:
            return None
        return CollectionRemoval(deleted_docs=len(doc_ids), cleaned_hashes=cleaned)

    def remove_collection(self, name: str) -> bool:
        """Remove a collection; True if it existed. See
        remove_collection_detailed for what gets deleted."""
        return self.remove_collection_detailed(name) is not None

    def cleanup_orphaned_content(self) -> int:
        """Delete content/content_vectors/vectors_vec rows for hashes no
        longer referenced by any active document. Global (not collection-
        scoped) -- content is addressed by hash, shared across collections.
        Returns the number of orphaned hashes cleaned. Extracted from
        remove_collection so update's re-scan path can reuse it (matching
        Node's reindexCollection, which calls the same cleanup inline after
        every scan -- see store.ts:1762)."""
        orphaned_rows = self.conn.execute(
            "SELECT hash FROM content WHERE hash NOT IN "
            "(SELECT DISTINCT hash FROM documents WHERE active = 1)"
        ).fetchall()
        orphaned_hashes = [row["hash"] for row in orphaned_rows]
        if orphaned_hashes:
            for chunk in _chunked(orphaned_hashes, SQL_IN_CHUNK_SIZE):
                oh_placeholders = ",".join("?" for _ in chunk)
                self.conn.execute(
                    f"DELETE FROM content_vectors WHERE hash IN ({oh_placeholders})",
                    chunk,
                )
            table_exists = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'"
            ).fetchone()
            if table_exists:
                for chunk in _chunked(orphaned_hashes, VEC_HASH_SEQ_IN_CHUNK):
                    vec_conditions = " OR ".join("hash_seq LIKE ? ESCAPE '\\'" for _ in chunk)
                    vec_params = [f"{_escape_like(h)}\\_%" for h in chunk]
                    self.conn.execute(f"DELETE FROM vectors_vec WHERE {vec_conditions}", vec_params)
            for chunk in _chunked(orphaned_hashes, SQL_IN_CHUNK_SIZE):
                oh_placeholders = ",".join("?" for _ in chunk)
                self.conn.execute(f"DELETE FROM content WHERE hash IN ({oh_placeholders})", chunk)
        self.conn.commit()
        return len(orphaned_hashes)

    def clear_llm_cache(self) -> int:
        """Delete all cached LLM API responses. Always returns 0 today --
        nothing in pyqmd currently writes to llm_cache (no query-expansion/
        rerank response caching is wired up yet, unlike Node) -- kept for
        parity and so nothing needs revisiting if that ever changes."""
        cursor = self.conn.execute("DELETE FROM llm_cache")
        self.conn.commit()
        return cursor.rowcount

    def count_llm_cache(self) -> int:
        return self.conn.execute("SELECT COUNT(*) as n FROM llm_cache").fetchone()["n"]

    def purge_inactive_documents(self) -> int:
        """Hard-delete every documents row with active = 0, globally (no
        collection filter) -- these are the re-scan/removal tombstones
        `update`/collection-remove leave behind. Explicit documents_fts/
        document_metadata(_values) cleanup, matching remove_collection's
        pattern (store.py:212), since PRAGMA foreign_keys isn't reliably
        re-enabled on reopening an existing on-disk DB -- cascade deletes
        cannot be relied on. Returns the count of documents purged."""
        doc_rows = self.conn.execute("SELECT id FROM documents WHERE active = 0").fetchall()
        doc_ids = [row["id"] for row in doc_rows]

        if doc_ids:
            for chunk in _chunked(doc_ids, SQL_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                self.conn.execute(
                    f"DELETE FROM documents_fts WHERE rowid IN ({placeholders})", chunk
                )
                self.conn.execute(
                    f"DELETE FROM document_metadata_values WHERE document_id IN ({placeholders})",
                    chunk,
                )
                self.conn.execute(
                    f"DELETE FROM document_metadata WHERE document_id IN ({placeholders})", chunk
                )
            self.conn.execute("DELETE FROM documents WHERE active = 0")
        self.conn.commit()
        return len(doc_ids)

    def count_inactive_documents(self) -> int:
        return self.conn.execute("SELECT COUNT(*) as n FROM documents WHERE active = 0").fetchone()[
            "n"
        ]

    def count_orphaned_content(self) -> int:
        """Read-only counterpart to cleanup_orphaned_content (store.py:244)
        -- same WHERE clause, no deletes. Backs `cleanup --dry-run`."""
        return self.conn.execute(
            "SELECT COUNT(*) as n FROM content WHERE hash NOT IN "
            "(SELECT DISTINCT hash FROM documents WHERE active = 1)"
        ).fetchone()["n"]

    def vacuum(self) -> None:
        """Rebuild the database file to reclaim space from deleted rows.
        Must be called with no pending open transaction (every mutating
        method in this file commits its own transaction before returning,
        so this is always safe as the last step of a cleanup run)."""
        self.conn.execute("VACUUM")

    def optimize_documents_fts(self) -> None:
        """Merge FTS5 b-trees so deleted rows (deactivated/purged
        documents) actually leave documents_fts_data -- VACUUM alone does
        not compact FTS5. Skips quietly if documents_fts doesn't exist,
        matching cleanup_orphaned_content's vectors_vec guard (store.py:
        264) -- defensive parity with Node's own guard; unreachable via
        this Store's normal lifecycle today since schema creation always
        creates documents_fts unconditionally."""
        table_exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'"
        ).fetchone()
        if not table_exists:
            return
        self.conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")
        self.conn.commit()

    def rename_collection(self, old_name: str, new_name: str) -> None:
        self.conn.execute("UPDATE collections SET name = ? WHERE name = ?", (new_name, old_name))
        self.conn.execute(
            "UPDATE documents SET collection = ? WHERE collection = ?", (new_name, old_name)
        )
        self.conn.commit()

    def set_collection_update_command(self, name: str, command: str | None) -> None:
        self.conn.execute(
            "UPDATE collections SET update_command = ? WHERE name = ?", (command, name)
        )
        self.conn.commit()

    def set_collection_include_by_default(self, name: str, include: bool) -> None:
        self.conn.execute(
            "UPDATE collections SET include_by_default = ? WHERE name = ?",
            (int(include), name),
        )
        self.conn.commit()

    def add_context(self, collection: str, path_prefix: str, text: str) -> bool:
        """Add or overwrite the context for one path prefix within a
        collection (path_prefix "" is the collection root). Returns False
        if the collection doesn't exist. Matches Node's addContext
        (collections.ts:398-417), but against the DB-stored JSON map
        (spec decision 1) instead of a YAML config file."""
        row = self.conn.execute(
            "SELECT context FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if row is None:
            return False
        ctx_map = _parse_context_map(row["context"])
        ctx_map[path_prefix] = text
        self.conn.execute(
            "UPDATE collections SET context = ? WHERE name = ?",
            (json.dumps(ctx_map), collection),
        )
        self.conn.commit()
        return True

    def remove_context(self, collection: str, path_prefix: str) -> bool:
        """Remove one path prefix's context. Returns False if the
        collection doesn't exist, or that exact prefix has no context set.
        Clears the column back to NULL once the map becomes empty, matching
        Node's removeStoreContext (store.ts:1425-1437)."""
        row = self.conn.execute(
            "SELECT context FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if row is None:
            return False
        ctx_map = _parse_context_map(row["context"])
        if path_prefix not in ctx_map:
            return False
        del ctx_map[path_prefix]
        new_value = json.dumps(ctx_map) if ctx_map else None
        self.conn.execute(
            "UPDATE collections SET context = ? WHERE name = ?", (new_value, collection)
        )
        self.conn.commit()
        return True

    def list_all_contexts(self) -> list[dict]:
        """[{"collection": str, "path": str, "context": str}, ...] across
        every collection that has a parseable context map, ordered by
        collection name (SQL ORDER BY) then by insertion order within each
        map (json.loads preserves key order). Matches Node's
        getStoreContexts (store.ts:1356-1375), minus the global-context row
        (out of scope, spec decision 2)."""
        rows = self.conn.execute(
            "SELECT name, context FROM collections WHERE context IS NOT NULL ORDER BY name"
        ).fetchall()
        results = []
        for row in rows:
            ctx_map = _parse_context_map(row["context"])
            for path_prefix, text in ctx_map.items():
                results.append({"collection": row["name"], "path": path_prefix, "context": text})
        return results

    def get_context_for_path(self, collection: str, path: str) -> str | None:
        """Hierarchical join: every stored path-prefix whose normalized
        form (leading '/') is a prefix of the normalized `path`, sorted
        shortest (most general) to longest (most specific), joined with
        '\\n\\n'. None if the collection doesn't exist, has no context map,
        or nothing matches. Matches store.ts's getContextForPath
        (store.ts:3521-3559) -- not collections.ts's separate, YAML-side
        findContextForPath, which returns only the single most-specific
        match (spec decision 3)."""
        row = self.conn.execute(
            "SELECT context FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        if row is None or not row["context"]:
            return None
        ctx_map = _parse_context_map(row["context"])
        normalized_path = path if path.startswith("/") else f"/{path}"
        matches = []
        for prefix, text in ctx_map.items():
            normalized_prefix = prefix if prefix.startswith("/") else f"/{prefix}"
            if normalized_path.startswith(normalized_prefix):
                matches.append((normalized_prefix, text))
        if not matches:
            return None
        matches.sort(key=lambda m: len(m[0]))
        return "\n\n".join(text for _prefix, text in matches)

    def get_context_for_file(self, filepath: str) -> str | None:
        """Accepts 'qmd://collection/path' or bare 'collection/path' (same
        strip-scheme-then-split-on-first-slash shape as _docid_for_file);
        delegates to get_context_for_path. None on any parse failure (no
        '/' in the remainder) or unknown collection."""
        without_scheme = filepath[len("qmd://") :] if filepath.startswith("qmd://") else filepath
        if "/" not in without_scheme:
            return None
        collection, path = without_scheme.split("/", 1)
        return self.get_context_for_path(collection, path)

    def detect_collection_for_path(self, fs_path: str) -> tuple[str, str] | None:
        """Realpath-normalize fs_path and every collection's stored `path`
        column, then longest-prefix match. Returns (collection_name,
        relative_path) -- relative_path is "" when fs_path IS the
        collection root -- or None if fs_path isn't under any collection's
        indexed directory. Matches Node's detectCollectionFromPath
        (src/cli/qmd.ts, referenced from contextAdd/contextRemove). Only
        this comparison normalizes to a realpath -- collection add itself
        still stores whatever raw path string the user typed (spec
        decision 5)."""
        real_target = os.path.realpath(fs_path)
        best: tuple[str, str] | None = None
        best_len = -1
        for coll in self.list_collections():
            real_coll_path = os.path.realpath(coll["path"])
            if real_target == real_coll_path:
                relative = ""
            elif real_target.startswith(real_coll_path + os.sep):
                relative = real_target[len(real_coll_path) + 1 :]
            else:
                continue
            if len(real_coll_path) > best_len:
                best = (coll["name"], relative)
                best_len = len(real_coll_path)
        return best

    def resolve_full_path(self, collection: str, path: str) -> str | None:
        """The reverse of detect_collection_for_path: given a collection
        name and a document's relative path (as stored in the
        `documents` table), return its realpath'd on-disk location, or
        None if the collection doesn't exist, the resolved path escapes
        the collection's root (mirrors Node's resolveVirtualPath's
        isPathInsideDir check), or the file no longer exists on disk
        (moved/deleted since indexing)."""
        coll = self.get_collection(collection)
        if coll is None:
            return None
        real_root = os.path.realpath(coll["path"])
        candidate = os.path.realpath(os.path.join(coll["path"], path))
        if candidate != real_root and not candidate.startswith(real_root + os.sep):
            return None
        if not os.path.exists(candidate):
            return None
        return candidate

    def hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def insert_content(self, content_hash: str, content: str, created_at: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)",
            (content_hash, content, created_at),
        )
        self.conn.commit()

    def insert_document(
        self,
        collection: str,
        path: str,
        title: str,
        content_hash: str,
        created_at: str,
        modified_at: str,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (collection, path, title, content_hash, created_at, modified_at),
        )
        self.conn.commit()
        self._sync_document_fts(cursor.lastrowid, title, path)
        return cursor.lastrowid

    def find_active_document(self, collection: str, path: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE collection = ? AND path = ? AND active = 1",
            (collection, path),
        ).fetchone()
        return dict(row) if row else None

    def update_document(
        self, document_id: int, title: str, content_hash: str, modified_at: str
    ) -> None:
        self.conn.execute(
            "UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?",
            (title, content_hash, modified_at, document_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT path FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        self._sync_document_fts(document_id, title, row["path"])

    def update_document_title(self, document_id: int, title: str, modified_at: str) -> None:
        """Set a document's title and modified_at when only its extracted
        title changed (same content), and re-sync its FTS row. Ported from
        store.ts's updateDocumentTitle; the scan counts it as updated."""
        self.conn.execute(
            "UPDATE documents SET title = ?, modified_at = ? WHERE id = ?",
            (title, modified_at, document_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT path FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        self._sync_document_fts(document_id, title, row["path"])

    def sync_document_metadata(
        self, document_id: int, content: str, path: str, *, only_if_stale: bool = False
    ) -> "MetadataExtractionResult | None":
        """Extract and persist metadata for one document, replacing any
        prior rows. With only_if_stale, extraction is skipped when the
        document already has a current-version extraction row -- the cheap
        path for unchanged documents during a re-scan. Returns the
        extraction result, or None when skipped."""
        from ._metadata import extract_document_metadata

        if only_if_stale and self._is_document_metadata_current(document_id):
            return None

        extraction = extract_document_metadata(content, path)
        self.replace_document_metadata(document_id, extraction)
        return extraction

    def _is_document_metadata_current(self, document_id: int) -> bool:
        from ._metadata import METADATA_EXTRACTION_VERSION

        row = self.conn.execute(
            "SELECT extraction_version FROM document_metadata WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row is not None and row["extraction_version"] == METADATA_EXTRACTION_VERSION

    def replace_document_metadata(
        self, document_id: int, extraction: "MetadataExtractionResult"
    ) -> None:
        """Replace a document's metadata rows atomically. A failed
        extraction persists empty metadata plus the error, so stale
        metadata never survives a bad edit."""
        import json as _json

        self.conn.execute(
            """
            INSERT INTO document_metadata (document_id, metadata_json, extraction_version, extraction_error, extracted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                metadata_json = excluded.metadata_json,
                extraction_version = excluded.extraction_version,
                extraction_error = excluded.extraction_error,
                extracted_at = excluded.extracted_at
            """,
            (
                document_id,
                _json.dumps(extraction.metadata),
                extraction.extraction_version,
                extraction.error,
                _now_iso(),
            ),
        )
        self.conn.execute(
            "DELETE FROM document_metadata_values WHERE document_id = ?", (document_id,)
        )

        for key, value in extraction.metadata.items():
            scalars = value if isinstance(value, list) else [value]
            for ordinal, scalar in enumerate(scalars):
                # bool must be checked before int/float -- bool is an int subclass.
                if isinstance(scalar, bool):
                    value_type, text_value, number_value, boolean_value = (
                        "boolean",
                        None,
                        None,
                        int(scalar),
                    )
                elif isinstance(scalar, str):
                    value_type, text_value, number_value, boolean_value = (
                        "string",
                        scalar,
                        None,
                        None,
                    )
                else:
                    value_type, text_value, number_value, boolean_value = (
                        "number",
                        None,
                        scalar,
                        None,
                    )
                self.conn.execute(
                    """
                    INSERT INTO document_metadata_values
                        (document_id, key, ordinal, value_type, text_value, number_value, boolean_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        key,
                        ordinal,
                        value_type,
                        text_value,
                        number_value,
                        boolean_value,
                    ),
                )
        self.conn.commit()

    def deactivate_document(self, collection: str, path: str) -> None:
        self.conn.execute(
            "UPDATE documents SET active = 0 WHERE collection = ? AND path = ?",
            (collection, path),
        )
        self.conn.commit()

    def get_active_document_paths(self, collection: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT path FROM documents WHERE collection = ? AND active = 1", (collection,)
        ).fetchall()
        return [row["path"] for row in rows]

    def get_active_documents_with_size(self, collection: str) -> list[dict]:
        """[{"path": str, "modified_at": str, "size": int}, ...] for a
        collection's active documents, ordered by path. `size` is the
        content's UTF-8 byte length (`LENGTH(CAST(doc AS BLOB))` --
        SQLite's plain LENGTH() on TEXT counts unicode characters, not
        bytes; casting to BLOB reinterprets the stored UTF-8 bytes so
        LENGTH() counts those instead, matching a real file's on-disk
        size). Backs `ls`'s per-file rich listing (roadmap #10)."""
        rows = self.conn.execute(
            """
            SELECT d.path, d.modified_at, LENGTH(CAST(c.doc AS BLOB)) AS size
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.collection = ? AND d.active = 1
            ORDER BY d.path
            """,
            (collection,),
        ).fetchall()
        return [dict(row) for row in rows]

    def find_document_by_identifier(self, identifier: str) -> dict | None:
        """Resolve a docid (a hash prefix, optionally '#'-prefixed), a
        'qmd://collection/path' / 'collection/path' string, or a bare
        filename to the active document's row (documents columns plus the
        content body as 'doc'). Docids never contain '/', so presence of
        '/' unambiguously means collection/path; otherwise it's tried as a
        docid prefix first, and if that finds nothing, as a bare path
        relative to ANY collection -- mirroring Node's findDocument()
        (src/store.ts), whose own per-collection relative-path fallback is
        what lets `qmd get some-file.md` work with no collection prefix at
        all (confirmed via parity-suite testing against real Node output).
        Returns None if nothing matches by any of these. If a
        docid prefix or a bare-path fallback matches multiple documents
        (across collections, or an extremely unlikely hash collision),
        returns one of them arbitrarily -- good enough for this MVP; a
        real collision would need a longer docid or a collection/path
        prefix to disambiguate."""
        ident = identifier.strip()
        if ident.startswith("qmd://"):
            ident = ident[len("qmd://") :]
        if ident.startswith("#"):
            ident = ident[1:]
        if not ident:
            return None

        if "/" in ident:
            collection, path = ident.split("/", 1)
            row = self.conn.execute(
                """
                SELECT d.*, c.doc AS doc
                FROM documents d JOIN content c ON c.hash = d.hash
                WHERE d.collection = ? AND d.path = ? AND d.active = 1
                """,
                (collection, path),
            ).fetchone()
            return dict(row) if row else None

        row = self.conn.execute(
            """
            SELECT d.*, c.doc AS doc
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.hash LIKE ? ESCAPE '\\' AND d.active = 1
            LIMIT 1
            """,
            (f"{_escape_like(ident)}%",),
        ).fetchone()
        if row:
            return dict(row)

        # Not a docid match -- fall back to treating the bare string as a
        # path relative to any collection (exact equality, not LIKE: this
        # is a real filename, not a hash-prefix pattern, so no wildcard
        # escaping is needed here).
        row = self.conn.execute(
            """
            SELECT d.*, c.doc AS doc
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.path = ? AND d.active = 1
            LIMIT 1
            """,
            (ident,),
        ).fetchone()
        return dict(row) if row else None

    def find_documents_by_glob(self, pattern: str) -> list[dict]:
        """Active documents (all collections) whose qmd://collection/path,
        bare path, or collection/path matches `pattern`. Mirrors Node's
        matchFilesByGlob (store.ts) -- same three-candidate matching, same
        brace-expansion/segment-aware-wildcard semantics (wcmatch.glob with
        the BRACE flag matches picomatch's defaults, verified directly; see
        the design spec's findings). Sorted by (collection, path) for a
        stable, deterministic order -- Node's own SQL has no ORDER BY, but
        the parity check only compares line count, not row order, so this
        is a pyqmd-side improvement, not a divergence anything depends on.
        Returns the same row shape as find_document_by_identifier (full
        `documents` columns plus `doc`), so callers can treat both sources
        uniformly."""
        from wcmatch import glob as wcglob

        rows = self.conn.execute(
            """
            SELECT d.*, c.doc AS doc
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.active = 1
            """
        ).fetchall()
        matched = []
        for row in rows:
            doc = dict(row)
            candidates = (
                f"qmd://{doc['collection']}/{doc['path']}",
                doc["path"],
                f"{doc['collection']}/{doc['path']}",
            )
            if any(wcglob.globmatch(c, pattern, flags=wcglob.BRACE) for c in candidates):
                matched.append(doc)
        matched.sort(key=lambda d: (d["collection"], d["path"]))
        return matched

    def _sync_document_fts(self, document_id: int, title: str, path: str) -> None:
        """Rebuild one document's FTS row from current documents/content data.
        Called after insert/update so documents_fts never drifts from the
        source tables (store.ts uses DB triggers for this; explicit calls
        after each write are simpler to reason about in Python and just as
        correct, since Store is the only writer)."""
        from ._fts_query import normalize_cjk_for_fts

        row = self.conn.execute(
            """
            SELECT d.path, c.doc AS body
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.id = ?
            """,
            (document_id,),
        ).fetchone()
        self.conn.execute("DELETE FROM documents_fts WHERE rowid = ?", (document_id,))
        if row:
            self.conn.execute(
                "INSERT INTO documents_fts (rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
                (
                    document_id,
                    path,
                    normalize_cjk_for_fts(title),
                    normalize_cjk_for_fts(row["body"]),
                ),
            )
        self.conn.commit()

    def ensure_vec_table(self, dimensions: int) -> None:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'"
        ).fetchone()
        if row:
            import re as _re

            match = _re.search(r"float\[(\d+)\]", row["sql"])
            existing_dims = int(match.group(1)) if match else None
            if existing_dims == dimensions:
                return
            if existing_dims is not None and existing_dims != dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: existing vectors are {existing_dims}d "
                    f"but the current model produces {dimensions}d. Re-index to switch models."
                )
            self.conn.execute("DROP TABLE IF EXISTS vectors_vec")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE vectors_vec USING "
            f"vec0(hash_seq TEXT PRIMARY KEY, embedding float[{dimensions}] distance_metric=cosine)"
        )
        self.conn.commit()

    def _representative_path(self, content_hash: str) -> str | None:
        """MIN(path) over active documents with this content: the path
        get_indexable_content pairs with each hash (Node's
        getPendingEmbeddingDocs), used when index_content's caller passes
        none. None when no active document has this content."""
        row = self.conn.execute(
            "SELECT MIN(path) AS path FROM documents WHERE hash = ? AND active = 1",
            (content_hash,),
        ).fetchone()
        return row["path"]

    def index_content(
        self,
        content_hash: str,
        content: str,
        model: str | None = None,
        filepath: str | None = None,
        chunk_strategy: str = "regex",
    ) -> int:
        """Chunk `content`, embed each chunk (unless already embedded with
        this model), and store the vectors. Returns the chunk count.
        `filepath` is get_indexable_content()'s MIN(path) for this hash; when
        None it is looked up the same way. It drives the title passed to the
        embedder and chunk_strategy="auto"'s language detection (see
        pyqmd_mlx.store._ast).

        The already-embedded check compares stored chunk *positions*
        against a fresh chunking pass, not just the chunk count: switching
        chunk_strategy (regex -> auto) can yield the same number of chunks
        at different boundaries, and a count-only check would silently
        retain the stale boundaries. When the boundaries are identical
        under both strategies there is genuinely nothing to re-embed, so
        position comparison also avoids useless re-embedding -- no schema
        change to track the producing strategy is needed.

        Raises ValueError for any chunk_strategy other than "regex"/"auto"."""
        import sqlite_vec

        from ._chunking import embedding_chunks, validate_chunk_strategy
        from ._title import extract_title

        validate_chunk_strategy(chunk_strategy)
        model = model or self._embed_model
        if filepath is None:
            filepath = self._representative_path(content_hash)
        existing_positions = [
            row["pos"]
            for row in self.conn.execute(
                "SELECT pos FROM content_vectors WHERE hash = ? AND model = ? ORDER BY seq",
                (content_hash, model),
            ).fetchall()
        ]
        chunks = embedding_chunks(content, filepath, chunk_strategy)
        if existing_positions == [pos for _text, pos in chunks]:
            return len(chunks)  # already embedded with this model and chunking

        # A model or content change: drop stale vectors for this hash first.
        self.conn.execute("DELETE FROM content_vectors WHERE hash = ?", (content_hash,))

        texts = [text for text, _pos in chunks]
        # Node embeds every chunk as "title: <extractTitle(body, path)> |
        # text: <chunk>" (store.ts:2125). Content with no document at all
        # (bare unit tests) has no path, so no title: "title: none".
        title = extract_title(content, filepath) if filepath is not None else None
        embeddings = self._embed_fn(texts, model, kind="document", title=title)
        self.ensure_vec_table(len(embeddings[0]))

        # vectors_vec only exists once ensure_vec_table has run at least
        # once, so the stale-row cleanup must happen after that call.
        self.conn.execute(
            "DELETE FROM vectors_vec WHERE hash_seq IN "
            "(SELECT hash_seq FROM vectors_vec WHERE hash_seq LIKE ? ESCAPE '\\')",
            (f"{_escape_like(content_hash)}\\_%",),
        )

        embedded_at = _now_iso()
        for seq, ((text, pos), vector) in enumerate(zip(chunks, embeddings)):
            self.conn.execute(
                """
                INSERT INTO content_vectors (hash, seq, pos, model, total_chunks, embedded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (content_hash, seq, pos, model, len(chunks), embedded_at),
            )
            self.conn.execute(
                "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
                (f"{content_hash}_{seq}", sqlite_vec.serialize_float32(vector)),
            )
        self.conn.commit()
        return len(chunks)

    def clear_embeddings(self, collection: str | None = None) -> int:
        """Delete content_vectors/vectors_vec rows for hashes belonging to
        active documents in `collection` (or all collections if None).
        Returns the number of content_vectors rows deleted. Used by the
        CLI's `embed --force` to make already-embedded hashes eligible for
        re-embedding (index_content's own no-op check would otherwise skip
        them)."""
        if collection:
            hash_rows = self.conn.execute(
                "SELECT DISTINCT hash FROM documents WHERE collection = ? AND active = 1",
                (collection,),
            ).fetchall()
        else:
            hash_rows = self.conn.execute(
                "SELECT DISTINCT hash FROM documents WHERE active = 1"
            ).fetchall()
        hashes = [row["hash"] for row in hash_rows]
        if not hashes:
            return 0

        cleared = 0
        for chunk in _chunked(hashes, SQL_IN_CHUNK_SIZE):
            placeholders = ",".join("?" for _ in chunk)
            cursor = self.conn.execute(
                f"DELETE FROM content_vectors WHERE hash IN ({placeholders})", chunk
            )
            cleared += cursor.rowcount

        table_exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'"
        ).fetchone()
        if table_exists:
            for chunk in _chunked(hashes, VEC_HASH_SEQ_IN_CHUNK):
                vec_conditions = " OR ".join("hash_seq LIKE ? ESCAPE '\\'" for _ in chunk)
                vec_params = [f"{_escape_like(h)}\\_%" for h in chunk]
                self.conn.execute(f"DELETE FROM vectors_vec WHERE {vec_conditions}", vec_params)
        self.conn.commit()
        return cleared

    def get_indexable_content(self, collection: str | None = None) -> list[dict]:
        """Distinct (hash, doc, path) rows for active documents, optionally
        scoped to one collection. `path` is MIN(d.path) across every
        document sharing that content hash -- an arbitrary but
        deterministic representative, matching Node's own
        getPendingEmbeddingDocs (store.ts:1925). It picks the title every
        chunk is embedded with and chunk_strategy="auto"'s language, so
        MIN(path) must stay what Node uses. Used by the CLI's `embed`
        command to find content to pass to index_content() -- Store owns
        all SQL, the CLI never touches `conn` directly."""
        if collection:
            rows = self.conn.execute(
                """
                SELECT c.hash, c.doc, MIN(d.path) AS path
                FROM content c JOIN documents d ON d.hash = c.hash
                WHERE d.collection = ? AND d.active = 1
                GROUP BY c.hash, c.doc
                """,
                (collection,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT c.hash, c.doc, MIN(d.path) AS path
                FROM content c JOIN documents d ON d.hash = c.hash
                WHERE d.active = 1
                GROUP BY c.hash, c.doc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def count_pending_embed(
        self,
        collection: str | None = None,
        model: str | None = None,
        chunk_strategy: str = "regex",
    ) -> int:
        """Count of distinct active-document hashes with no content_vectors
        row for `model` (or the Store's default), optionally scoped to one
        collection. Used by the CLI's `embed` command to detect a genuine
        no-op -- everything already embedded -- before doing any chunking/
        embedding work, matching Node's getHashesNeedingEmbedding check.
        Unlike get_status_counts's own "pending_embed" (always global),
        this one is collection-scopable to match embed's own -c option.

        With chunk_strategy="auto", hashes whose stored chunk boundaries
        differ from a fresh auto chunking pass also count as pending --
        otherwise `embed --chunk-strategy auto` after a regex embed would
        see a zero count and silently no-op. This requires chunking every
        candidate's content (CPU/tree-sitter only, no embedding work), so
        the SQL fast path still runs first and the scan only happens when
        it reports zero."""
        from ._chunking import embedding_chunks, validate_chunk_strategy

        validate_chunk_strategy(chunk_strategy)
        model = model or self._embed_model
        if collection:
            row = self.conn.execute(
                """
                SELECT COUNT(DISTINCT d.hash) AS n FROM documents d
                WHERE d.active = 1 AND d.collection = ?
                AND NOT EXISTS (
                    SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash AND cv.model = ?
                )
                """,
                (collection, model),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT COUNT(DISTINCT d.hash) AS n FROM documents d
                WHERE d.active = 1
                AND NOT EXISTS (
                    SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash AND cv.model = ?
                )
                """,
                (model,),
            ).fetchone()
        pending = row["n"]
        if chunk_strategy == "regex" or pending > 0:
            return pending
        # Auto strategy with nothing missing: detect stale chunk
        # boundaries left by a regex (or older auto) embed. Mirrors
        # index_content's own position comparison, which is what actually
        # decides per-hash whether re-embedding happens.
        stale = 0
        for item in self.get_indexable_content(collection):
            stored = [
                r["pos"]
                for r in self.conn.execute(
                    "SELECT pos FROM content_vectors WHERE hash = ? AND model = ? ORDER BY seq",
                    (item["hash"], model),
                ).fetchall()
            ]
            fresh = embedding_chunks(item["doc"], item["path"], "auto")
            if stored != [pos for _text, pos in fresh]:
                stale += 1
        return stale

    def get_status_counts(self, model: str | None = None) -> dict:
        """Aggregate counts for the CLI's `status` command: active document
        count, distinct embedded-vector-row count, hashes still pending
        embedding for `model` (or the Store's default embed model), and the
        most recent document modification timestamp."""
        model = model or self._embed_model
        doc_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE active = 1"
        ).fetchone()["n"]
        vector_count = self.conn.execute("SELECT COUNT(*) AS n FROM content_vectors").fetchone()[
            "n"
        ]
        pending = self.conn.execute(
            """
            SELECT COUNT(DISTINCT d.hash) AS n FROM documents d
            WHERE d.active = 1
            AND NOT EXISTS (
                SELECT 1 FROM content_vectors cv WHERE cv.hash = d.hash AND cv.model = ?
            )
            """,
            (model,),
        ).fetchone()["n"]
        most_recent = self.conn.execute(
            "SELECT MAX(modified_at) AS latest FROM documents WHERE active = 1"
        ).fetchone()["latest"]
        return {
            "active_documents": doc_count,
            "embedded_vectors": vector_count,
            "pending_embed": pending,
            "most_recent_modified_at": most_recent,
        }

    def get_collection_document_stats(self) -> dict[str, dict]:
        """Per-collection active-document count and most-recent modified_at,
        in one batched GROUP BY query (avoids an N+1 per-collection loop in
        `status`). A collection with zero active documents is simply absent
        from the returned dict -- GROUP BY has nothing to group -- so
        callers default missing keys to {"count": 0, "latest_modified":
        None} themselves."""
        rows = self.conn.execute(
            """
            SELECT collection, COUNT(*) AS n, MAX(modified_at) AS latest
            FROM documents
            WHERE active = 1
            GROUP BY collection
            """
        ).fetchall()
        return {
            row["collection"]: {"count": row["n"], "latest_modified": row["latest"]} for row in rows
        }

    def get_metadata_by_filepath(self, filepaths: list[str]) -> dict[str, dict]:
        """Batch-load canonical metadata for a set of result filepaths
        (qmd://collection/path). Chunked into SQL_IN_CHUNK_SIZE-sized IN
        clauses -- never per-result lookups."""
        import json as _json

        if not filepaths:
            return {}

        result: dict[str, dict] = {}
        for chunk in _chunked(filepaths, SQL_IN_CHUNK_SIZE):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT 'qmd://' || d.collection || '/' || d.path AS filepath, dm.metadata_json
                FROM documents d
                JOIN document_metadata dm ON dm.document_id = d.id
                WHERE d.active = 1
                AND ('qmd://' || d.collection || '/' || d.path) IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                try:
                    result[row["filepath"]] = _json.loads(row["metadata_json"])
                except (TypeError, ValueError):
                    result[row["filepath"]] = {}
        return result

    def count_documents_pending_metadata(self) -> int:
        """Count active documents without a current, error-free metadata
        extraction. These documents are excluded from filtered search until
        `collection add` (or a future `update`) runs."""
        from ._metadata import METADATA_EXTRACTION_VERSION

        row = self.conn.execute(
            """
            SELECT COUNT(*) as c FROM documents d
            WHERE d.active = 1
            AND NOT EXISTS (
                SELECT 1 FROM document_metadata dm
                WHERE dm.document_id = d.id
                AND dm.extraction_version = ?
                AND dm.extraction_error IS NULL
            )
            """,
            (METADATA_EXTRACTION_VERSION,),
        ).fetchone()
        return row["c"]

    def rebuild_fts(self) -> None:
        """Drop and repopulate documents_fts from the untouched content/
        documents tables -- always possible since normalization never
        modifies the stored content itself. Explicit/manual, not triggered
        automatically on a normalization-rule change (see design spec)."""
        from ._fts_query import normalize_cjk_for_fts

        self.conn.execute("DELETE FROM documents_fts")
        rows = self.conn.execute(
            """
            SELECT d.id, d.path, d.title, c.doc AS body
            FROM documents d JOIN content c ON c.hash = d.hash
            WHERE d.active = 1
            """
        ).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT INTO documents_fts (rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
                (
                    row["id"],
                    row["path"],
                    normalize_cjk_for_fts(row["title"]),
                    normalize_cjk_for_fts(row["body"]),
                ),
            )
        self.conn.commit()

    def _normalize_collections(self, collection: "str | list[str] | None") -> list[str] | None:
        """Normalize the `collection` filter argument accepted by search_fts/
        search_vec/query into a list, or None for "no restriction". A bare
        string means one collection; an iterable means match any of them
        (OR semantics, via SQL IN). When nothing is given, resolves to
        get_default_collection_names() instead of unconditionally returning
        None -- this is what makes include_by_default actually scope
        default (no `-c`) search results. Naming a collection explicitly is
        never filtered by include_by_default -- matches Node's
        resolveCollectionFilter override rule."""
        if collection is None:
            return self.get_default_collection_names()
        if isinstance(collection, str):
            return [collection]
        return list(collection)

    def get_default_collection_names(self) -> list[str] | None:
        """Names of collections with include_by_default=1, or None if
        every collection is included (the common case) -- callers should
        treat None identically to "no restriction" to avoid an
        unnecessary IN-clause and search_fts's fts_limit overfetch-window
        widening when nothing is actually excluded."""
        excluded = self.conn.execute(
            "SELECT 1 FROM collections WHERE include_by_default = 0 LIMIT 1"
        ).fetchone()
        if excluded is None:
            return None
        rows = self.conn.execute(
            "SELECT name FROM collections WHERE include_by_default = 1"
        ).fetchall()
        return [r["name"] for r in rows]

    def search_fts(
        self,
        query: str,
        limit: int = 20,
        collection: "str | list[str] | None" = None,
        filter: "MetadataFilter | None" = None,
    ) -> list["SearchResult"]:
        from ._fts_query import build_fts5_query
        from ._metadata import METADATA_EXTRACTION_VERSION
        from ._metadata_filter import compile_metadata_filter
        from ._types import SearchResult

        fts_query = build_fts5_query(query)
        if fts_query is None:
            return []

        params: list = [fts_query]
        sql = """
            WITH fts_matches AS (
                SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) AS bm25_score
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
            )
            SELECT
                'qmd://' || d.collection || '/' || d.path AS filepath,
                d.collection || '/' || d.path AS display_path,
                d.title,
                c.doc AS body,
                d.hash,
                d.collection,
                fm.bm25_score
            FROM fts_matches fm
            JOIN documents d ON d.id = fm.rowid
            JOIN content c ON c.hash = d.hash
            LEFT JOIN document_metadata dm ON dm.document_id = d.id
            WHERE d.active = 1
        """
        collections = self._normalize_collections(collection)
        # NOTE: unlike search_vec (which routes a selective `filter` or
        # `collection` through an eligible-set exact-scan to avoid
        # starvation -- see FILTERED_VEC_EXACT_SCAN_MAX above), search_fts
        # has no equivalent fallback: a selective filter/collection can
        # still be silently starved out of this CTE's overfetch window if
        # the matching document isn't already in the raw BM25 top-N. This
        # is exact parity with the Node reference (store.ts:4123's
        # identical `limit * 10`), not a regression -- but it's a real,
        # documented limitation, not a guarantee. A proper fix would mirror
        # search_vec's eligible-set pattern; out of scope here.
        fts_limit = limit * 10 if (collections or filter is not None) else limit
        params.append(fts_limit)

        if collections:
            placeholders = ",".join("?" for _ in collections)
            sql += f" AND d.collection IN ({placeholders})"
            params.extend(collections)

        if filter is not None:
            compiled = compile_metadata_filter(filter, "d")
            sql += (
                f" AND dm.extraction_version = {METADATA_EXTRACTION_VERSION}"
                " AND dm.extraction_error IS NULL"
                f" AND {compiled.sql}"
            )
            params.extend(compiled.params)

        sql += " ORDER BY fm.bm25_score ASC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            abs_score = abs(row["bm25_score"])
            score = abs_score / (1 + abs_score)
            results.append(
                SearchResult(
                    filepath=row["filepath"],
                    display_path=row["display_path"],
                    title=row["title"],
                    hash=row["hash"],
                    docid=row["hash"][:6],
                    collection_name=row["collection"],
                    body=row["body"],
                    score=score,
                    source="fts",
                    context=self.get_context_for_file(row["filepath"]),
                )
            )
        return results

    def _exact_vec_scan_by_hash_seq(
        self, embedding: list[float], hash_seqs: list[str], limit: int
    ) -> list[sqlite3.Row]:
        """Exact cosine-distance scan over a known set of hash_seq keys, done
        in chunks to keep the IN-list bounded. Ported from store.ts's
        exactVecScanByHashSeq (store.ts:4209-4239). Queries vectors_vec by its
        own primary key with no JOIN, so this doesn't violate the "never JOIN
        sqlite-vec with a regular table" rule."""
        import sqlite_vec

        if not hash_seqs or limit <= 0:
            return []

        query_vec = sqlite_vec.serialize_float32(embedding)
        # Over-fetch a bit so multi-chunk docs can still yield `limit` unique files.
        fetch_limit = max(limit * 3, limit)
        scored: list[sqlite3.Row] = []

        for i in range(0, len(hash_seqs), VEC_HASH_SEQ_IN_CHUNK):
            chunk = hash_seqs[i : i + VEC_HASH_SEQ_IN_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT hash_seq, vec_distance_cosine(embedding, ?) AS distance
                FROM vectors_vec
                WHERE hash_seq IN ({placeholders})
                """,
                (query_vec, *chunk),
            ).fetchall()
            scored.extend(rows)

        scored.sort(key=lambda r: r["distance"])
        return scored[:fetch_limit]

    def search_vec(
        self,
        query: str,
        limit: int = 20,
        collection: "str | list[str] | None" = None,
        model: str | None = None,
        filter: "MetadataFilter | None" = None,
    ) -> list["SearchResult"]:
        import sqlite_vec

        from ._metadata import METADATA_EXTRACTION_VERSION
        from ._metadata_filter import compile_metadata_filter
        from ._types import SearchResult

        model = model or self._embed_model
        table_exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'"
        ).fetchone()
        if not table_exists:
            return []

        embedding = self._embed_fn([query], model, kind="query")[0]
        collections = self._normalize_collections(collection)

        # sqlite-vec virtual tables must never be JOINed in the same query as
        # regular tables (hangs indefinitely), so neither `collection` nor
        # `filter` can be pushed into MATCH. Global ANN + post-filter starves
        # small eligible sets (their documents never enter the top `limit*3`
        # in the first place -- store.ts:4196-4203, issues #791/#803, and the
        # equivalent metadata-filter case at store.ts:4277-4282). So whenever
        # either is given, exact-scan the eligible vectors when the set is
        # small enough, and only then fall back to over-fetched ANN.
        if collections or filter is not None:
            eligible_sql = """
                SELECT DISTINCT cv.hash || '_' || cv.seq AS hash_seq
                FROM content_vectors cv
                JOIN documents d ON d.hash = cv.hash AND d.active = 1
            """
            eligible_conditions: list[str] = []
            eligible_params: list = []

            if collections:
                placeholders = ",".join("?" for _ in collections)
                eligible_conditions.append(f"d.collection IN ({placeholders})")
                eligible_params.extend(collections)

            if filter is not None:
                compiled = compile_metadata_filter(filter, "d")
                eligible_sql += " JOIN document_metadata dm ON dm.document_id = d.id"
                eligible_conditions.append(f"dm.extraction_version = {METADATA_EXTRACTION_VERSION}")
                eligible_conditions.append("dm.extraction_error IS NULL")
                eligible_conditions.append(compiled.sql)
                eligible_params.extend(compiled.params)

            eligible_sql += " WHERE " + " AND ".join(eligible_conditions)
            eligible_rows = self.conn.execute(eligible_sql, eligible_params).fetchall()
            eligible_hash_seqs = [row["hash_seq"] for row in eligible_rows]
            if not eligible_hash_seqs:
                return []

            if len(eligible_hash_seqs) <= FILTERED_VEC_EXACT_SCAN_MAX:
                vec_rows = self._exact_vec_scan_by_hash_seq(embedding, eligible_hash_seqs, limit)
            else:
                # Large eligible set: ANN with a bigger over-fetch instead of
                # an exact scan, hard-capped at sqlite-vec's max k of 4096.
                k = max(1, min(4096, max(limit * 30, limit * 3)))
                vec_rows = self.conn.execute(
                    "SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?",
                    (sqlite_vec.serialize_float32(embedding), k),
                ).fetchall()
        else:
            k = max(1, min(4096, limit * 3))
            vec_rows = self.conn.execute(
                "SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?",
                (sqlite_vec.serialize_float32(embedding), k),
            ).fetchall()

        if not vec_rows:
            return []

        hash_seqs = [row["hash_seq"] for row in vec_rows]
        distance_by_hash_seq = {row["hash_seq"]: row["distance"] for row in vec_rows}

        # Re-fetch document data for the matched hashes. Keep the collection/
        # filter guards here too even after the eligible-set fix above: a
        # content hash can theoretically be shared by identical content
        # living in two different documents, so this still matters for
        # picking the right document row.
        placeholders = ",".join("?" for _ in hash_seqs)
        sql = f"""
            SELECT
                cv.hash || '_' || cv.seq AS hash_seq,
                'qmd://' || d.collection || '/' || d.path AS filepath,
                d.collection || '/' || d.path AS display_path,
                d.title,
                d.hash,
                d.collection,
                c.doc AS body,
                cv.pos
            FROM content_vectors cv
            JOIN documents d ON d.hash = cv.hash AND d.active = 1
            JOIN content c ON c.hash = d.hash
            LEFT JOIN document_metadata dm ON dm.document_id = d.id
            WHERE (cv.hash || '_' || cv.seq) IN ({placeholders})
        """
        params: list = list(hash_seqs)
        if collections:
            col_placeholders = ",".join("?" for _ in collections)
            sql += f" AND d.collection IN ({col_placeholders})"
            params.extend(collections)

        if filter is not None:
            compiled = compile_metadata_filter(filter, "d")
            sql += (
                f" AND dm.extraction_version = {METADATA_EXTRACTION_VERSION}"
                " AND dm.extraction_error IS NULL"
                f" AND {compiled.sql}"
            )
            params.extend(compiled.params)

        rows = self.conn.execute(sql, params).fetchall()

        # Dedupe by filepath, keeping the chunk with the lowest (best)
        # distance, THEN compute scores/sort/slice -- matches store.ts's
        # searchVec (store.ts:4370-4396), which fixes the inflated RRF
        # contribution a multi-chunk document would otherwise get from
        # occupying multiple slots in the result list.
        best_by_filepath: dict[str, tuple[sqlite3.Row, float]] = {}
        for row in rows:
            distance = distance_by_hash_seq[row["hash_seq"]]
            existing = best_by_filepath.get(row["filepath"])
            if existing is None or distance < existing[1]:
                best_by_filepath[row["filepath"]] = (row, distance)

        results = []
        for row, distance in best_by_filepath.values():
            # cosine distance in [0, 2]; convert to a [0, 1]-ish similarity score.
            score = max(0.0, 1.0 - distance / 2.0)
            results.append(
                SearchResult(
                    filepath=row["filepath"],
                    display_path=row["display_path"],
                    title=row["title"],
                    hash=row["hash"],
                    docid=row["hash"][:6],
                    collection_name=row["collection"],
                    body=row["body"],
                    score=score,
                    source="vec",
                    chunk_pos=row["pos"],
                    context=self.get_context_for_file(row["filepath"]),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _retrieve_and_fuse(
        self,
        query: str,
        collection: "str | list[str] | None",
        candidate_limit: int,
        intent: str | None = None,
        filter: "MetadataFilter | None" = None,
    ) -> list["RankedResult"]:
        from ._expansion import postprocess_expansion
        from ._rrf import get_hybrid_rrf_weights, reciprocal_rank_fusion
        from ._types import RankedListMeta, RankedResult

        def to_ranked(results) -> list[RankedResult]:
            return [
                RankedResult(
                    file=r.filepath,
                    display_path=r.display_path,
                    title=r.title,
                    body=r.body,
                    score=r.score,
                )
                for r in results
            ]

        # Step 1: BM25 probe. Filter is passed directly into the probe too --
        # the strong-signal decision must be based only on eligible
        # documents (store.ts:5579-5582). A strong, unambiguous hit skips
        # expansion entirely -- unless intent is given: an intent-qualified
        # query means the caller has a specific angle in mind that BM25's
        # literal-term match might not reflect (store.ts:5579-5588).
        bm25_probe = self.search_fts(query, limit=2, collection=collection, filter=filter)
        if not intent and bm25_probe and bm25_probe[0].score >= STRONG_SIGNAL_MIN_SCORE:
            gap = bm25_probe[0].score - (bm25_probe[1].score if len(bm25_probe) > 1 else 0.0)
            if gap >= STRONG_SIGNAL_MIN_GAP:
                fts_results = self.search_fts(
                    query, limit=candidate_limit, collection=collection, filter=filter
                )
                vec_results = self.search_vec(
                    query, limit=candidate_limit, collection=collection, filter=filter
                )
                lists = [to_ranked(vec_results), to_ranked(fts_results)]
                meta = [
                    RankedListMeta(source="vec", query_type="original", query=query),
                    RankedListMeta(source="fts", query_type="original", query=query),
                ]
                weights = get_hybrid_rrf_weights(meta)
                return reciprocal_rank_fusion(lists, weights)[:candidate_limit]

        # Step 2: expand, then type-route each variant to FTS or vector search.
        expanded_lines = self._expand_fn(query, self._expand_model)
        expanded_parts = postprocess_expansion(query, expanded_lines)

        lists = [
            to_ranked(
                self.search_vec(query, limit=candidate_limit, collection=collection, filter=filter)
            )
        ]
        meta = [RankedListMeta(source="vec", query_type="original", query=query)]

        lists.append(
            to_ranked(
                self.search_fts(query, limit=candidate_limit, collection=collection, filter=filter)
            )
        )
        meta.append(RankedListMeta(source="fts", query_type="original", query=query))

        for part in expanded_parts:
            if part.type == "lex":
                results = self.search_fts(
                    part.query, limit=candidate_limit, collection=collection, filter=filter
                )
                source = "fts"
            else:  # "vec" or "hyde"
                results = self.search_vec(
                    part.query, limit=candidate_limit, collection=collection, filter=filter
                )
                source = "vec"
            lists.append(to_ranked(results))
            meta.append(RankedListMeta(source=source, query_type=part.type, query=part.query))

        weights = get_hybrid_rrf_weights(meta)
        return reciprocal_rank_fusion(lists, weights)[:candidate_limit]

    def query(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        candidate_limit: int = RERANK_CANDIDATE_LIMIT,
        collection: "str | list[str] | None" = None,
        skip_rerank: bool = False,
        intent: str | None = None,
        filter: "MetadataFilter | None" = None,
        chunk_strategy: str = "regex",
    ) -> list["HybridQueryResult"]:
        from ._chunking import chunk_document, validate_chunk_strategy
        from ._intent import INTENT_WEIGHT_CHUNK, extract_intent_terms
        from ._types import HybridQueryResult

        validate_chunk_strategy(chunk_strategy)
        fused = self._retrieve_and_fuse(query, collection, candidate_limit, intent, filter)
        if not fused:
            return []

        # Best-chunk selection: for each candidate, pick the chunk with the
        # most query/intent-term overlap (a cheap proxy -- reranking on the
        # full body would be an O(tokens) trap, see design spec). Matches
        # store.ts's substring-containment scoring (store.ts:5701-5710), not
        # exact-token-set intersection -- "auth" must match inside
        # "authentication". Query terms longer than 2 chars only, matching
        # store.ts's own filter.
        query_terms = {t.lower() for t in query.split() if len(t) > 2}
        intent_terms = extract_intent_terms(intent) if intent else []
        candidates = []
        for ranked in fused:
            chunks = chunk_document(
                ranked.body, filepath=ranked.file, chunk_strategy=chunk_strategy
            )
            best_text, best_pos = chunks[0]
            best_score = -1.0
            for text, pos in chunks:
                text_lower = text.lower()
                score = sum(1.0 for t in query_terms if t in text_lower)
                score += sum(INTENT_WEIGHT_CHUNK for t in intent_terms if t in text_lower)
                if score > best_score:
                    best_score = score
                    best_text, best_pos = text, pos
            candidates.append((ranked, best_text, best_pos))

        if skip_rerank:
            # Score by final rank position (1-indexed reciprocal), not the
            # raw internal RRF fusion score -- matches store.ts's skipRerank
            # branch exactly (store.ts:5736-5737). The raw RRF score's range
            # is tiny (~0.03-0.17 depending on how many ranked lists a
            # candidate appears in) and was never meant to be user-facing;
            # min_score is calibrated against the reranked branch's ~0..1
            # blended scale, so exposing raw RRF here silently filtered out
            # every result for any non-trivial min_score threshold.
            final_scores = {id(c[0]): 1.0 / (i + 1) for i, c in enumerate(candidates)}
        else:
            documents = [c[1] for c in candidates]
            # Prepend intent to the reranker's input so it scores with
            # domain context (store.ts:4624-4625) -- pyqmd_mlx.llm.rerank() has no
            # intent parameter of its own, so the composite string IS the
            # query it receives.
            rerank_query = f"{intent}\n\n{query}" if intent else query
            rerank_scores = self._rerank_fn(rerank_query, documents, self._rerank_model)
            # Blend RRF rank position with the reranker score: normalize RRF
            # score to a 0..1-ish position weight, average with reranker score.
            max_rrf = max((c[0].score for c in candidates), default=1.0) or 1.0
            final_scores = {}
            for (ranked, _text, _pos), rerank_score in zip(candidates, rerank_scores):
                position_weight = ranked.score / max_rrf
                final_scores[id(ranked)] = (position_weight + rerank_score) / 2

        seen_files: set[str] = set()
        results = []
        for ranked, best_text, best_pos in candidates:
            if ranked.file in seen_files:
                continue
            seen_files.add(ranked.file)
            score = final_scores[id(ranked)]
            if score < min_score:
                continue
            results.append(
                HybridQueryResult(
                    file=ranked.file,
                    display_path=ranked.display_path,
                    title=ranked.title,
                    body=ranked.body,
                    best_chunk=best_text,
                    best_chunk_pos=best_pos,
                    score=score,
                    context=self.get_context_for_file(ranked.file),
                    docid=self._docid_for_file(ranked.file),
                )
            )

        metadata_by_filepath = self.get_metadata_by_filepath([r.file for r in results])
        for result in results:
            result.metadata = metadata_by_filepath.get(result.file, {})

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _docid_for_file(self, filepath: str) -> str:
        # filepath is "qmd://collection/path" -- look up the document's hash.
        without_scheme = filepath[len("qmd://") :]
        collection, path = without_scheme.split("/", 1)
        doc = self.find_active_document(collection, path)
        return doc["hash"][:6] if doc else ""
