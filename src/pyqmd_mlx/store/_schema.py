"""Schema creation and PRAGMA user_version migration dispatch.

Included from the start even though today there is only one migration step
(v0 -> v1 = create the current schema): retrofitting version-awareness onto
a database that has already accumulated real, un-versioned rows is much
harder than including this scaffold now, while it costs almost nothing.
"""

import sqlite3
import sys

CURRENT_SCHEMA_VERSION = 1


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    if not hasattr(conn, "enable_load_extension"):
        raise RuntimeError(
            "This Python build lacks sqlite3 loadable-extension support "
            f"({sys.version.split()[0]} at {sys.base_prefix}) -- pyqmd's "
            "vector index cannot load. Use a full CPython build (python.org, "
            "Homebrew, or uv-managed Python)."
        )
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            doc TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            hash TEXT NOT NULL REFERENCES content(hash) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            UNIQUE(collection, path)
        );
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);

        CREATE TABLE IF NOT EXISTS collections (
            name TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            pattern TEXT NOT NULL DEFAULT '**/*.md',
            ignore_patterns TEXT,
            include_by_default INTEGER NOT NULL DEFAULT 1,
            update_command TEXT,
            context TEXT
        );

        CREATE TABLE IF NOT EXISTS content_vectors (
            hash TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0,
            pos INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL,
            embed_fingerprint TEXT NOT NULL DEFAULT '',
            total_chunks INTEGER NOT NULL DEFAULT 1,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (hash, seq)
        );
        CREATE INDEX IF NOT EXISTS idx_content_vectors_model
            ON content_vectors(model, hash);

        CREATE TABLE IF NOT EXISTS llm_cache (
            hash TEXT PRIMARY KEY,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filepath UNINDEXED,
            title,
            body,
            tokenize = 'porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS document_metadata (
            document_id INTEGER PRIMARY KEY,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            extraction_version INTEGER NOT NULL,
            extraction_error TEXT,
            extracted_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS document_metadata_values (
            document_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            value_type TEXT NOT NULL,
            text_value TEXT,
            number_value REAL,
            boolean_value INTEGER,
            PRIMARY KEY (document_id, key, ordinal),
            FOREIGN KEY (document_id) REFERENCES document_metadata(document_id) ON DELETE CASCADE,
            CHECK (value_type IN ('string', 'number', 'boolean')),
            CHECK (
                (value_type = 'string' AND text_value IS NOT NULL AND number_value IS NULL AND boolean_value IS NULL)
                OR (value_type = 'number' AND number_value IS NOT NULL AND text_value IS NULL AND boolean_value IS NULL)
                OR (value_type = 'boolean' AND boolean_value IN (0, 1) AND text_value IS NULL AND number_value IS NULL)
            )
        );

        CREATE INDEX IF NOT EXISTS idx_metadata_text_lookup
            ON document_metadata_values(key, text_value, document_id) WHERE value_type = 'string';
        CREATE INDEX IF NOT EXISTS idx_metadata_number_lookup
            ON document_metadata_values(key, number_value, document_id) WHERE value_type = 'number';
        CREATE INDEX IF NOT EXISTS idx_metadata_boolean_lookup
            ON document_metadata_values(key, boolean_value, document_id) WHERE value_type = 'boolean';
        """
    )


_MIGRATIONS: dict[int, callable] = {
    0: _create_schema_v1,
}


def migrate(conn: sqlite3.Connection) -> None:
    """Bring the database up to CURRENT_SCHEMA_VERSION, running each
    migration step in order starting from the current PRAGMA user_version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    while current < CURRENT_SCHEMA_VERSION:
        step = _MIGRATIONS[current]
        step(conn)
        current += 1
        conn.execute(f"PRAGMA user_version = {current}")
    conn.commit()
