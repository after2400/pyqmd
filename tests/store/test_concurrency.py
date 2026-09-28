import concurrent.futures
import sqlite3
import threading
import time

from pyqmd_mlx.store import Store


def test_new_store_uses_wal_mode_on_file_db(tmp_path):
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    store.close()


def test_new_store_on_memory_db_does_not_error(tmp_path):
    # PRAGMA journal_mode=WAL on :memory: is a silent no-op (stays 'memory'),
    # not an error -- this must not raise or change behavior for the many
    # existing tests that use Store(":memory:").
    store = Store(":memory:")
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "memory"
    store.close()


def test_busy_timeout_defaults_to_120_seconds(tmp_path):
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    timeout_ms = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout_ms == 120_000
    store.close()


def test_busy_timeout_env_var_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QMD_SQLITE_BUSY_TIMEOUT", "5000")
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    timeout_ms = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout_ms == 5000
    store.close()


def test_busy_timeout_env_var_zero_restores_failfast(tmp_path, monkeypatch):
    monkeypatch.setenv("QMD_SQLITE_BUSY_TIMEOUT", "0")
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    timeout_ms = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout_ms == 0
    store.close()


def test_busy_timeout_env_var_invalid_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("QMD_SQLITE_BUSY_TIMEOUT", "not-a-number")
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    timeout_ms = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout_ms == 120_000
    store.close()


def test_concurrent_writer_waits_instead_of_failing_immediately(tmp_path):
    """Two separate connections to the same on-disk database: one holds a
    write transaction open, the other's write should WAIT (honoring
    busy_timeout) rather than immediately raising SQLITE_BUSY, and succeed
    once the first commits. This is the exact scenario motivating this
    task: a CLI write racing the MCP server's long-lived connection."""
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    store.add_collection("seed", "/tmp/seed")  # forces schema onto disk
    store.close()

    conn_a = sqlite3.connect(str(db_path))
    conn_a.execute("PRAGMA busy_timeout = 120000")
    # check_same_thread=False: conn_b is used from writer_b's thread below,
    # not the thread that created it -- Python's sqlite3 forbids that by
    # default (unrelated to WAL/busy_timeout; this is Python-level, not
    # SQLite-level).
    conn_b = sqlite3.connect(str(db_path), check_same_thread=False)
    conn_b.execute("PRAGMA busy_timeout = 120000")

    conn_a.execute("BEGIN IMMEDIATE")
    conn_a.execute("INSERT INTO collections (name, path) VALUES ('a', '/a')")

    writer_started = threading.Event()
    result: dict = {}

    def writer_b():
        writer_started.set()
        try:
            conn_b.execute("INSERT INTO collections (name, path) VALUES ('b', '/b')")
            conn_b.commit()
            result["ok"] = True
        except sqlite3.OperationalError as exc:
            result["ok"] = False
            result["error"] = str(exc)

    t = threading.Thread(target=writer_b)
    t.start()
    writer_started.wait(timeout=2)
    time.sleep(0.2)  # let writer_b actually enter its blocking execute()
    conn_a.commit()
    t.join(timeout=5)

    assert result.get("ok") is True, result.get("error")
    conn_a.close()
    conn_b.close()


def test_store_connection_usable_from_a_different_thread(tmp_path):
    """Store's connection must tolerate being used from a thread other than
    the one that constructed it, as long as access is externally serialized
    (e.g. the MCP server's `_locked` mutex in pyqmd_mlx/mcp/server.py) -- this is
    exactly what anyio.to_thread.run_sync's worker-thread-pool dispatch does
    on every real MCP tool call. Without check_same_thread=False on Store's
    sqlite3.connect(), this raises sqlite3.ProgrammingError even though
    access here is already fully serialized (one call, awaited to
    completion, before the next)."""
    db_path = tmp_path / "index.sqlite"
    store = Store(str(db_path))
    store.add_collection("notes", "/notes")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        collections = executor.submit(store.list_collections).result(timeout=5)

    assert [c["name"] for c in collections] == ["notes"]
    store.close()
