from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.embed import app
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import scan_and_register_collection

runner = CliRunner()


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] for _ in texts]


def _seed_doc(store, collection, path, body):
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document(collection, path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    return h


def test_embed_embeds_pending_documents(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed"])

    assert result.exit_code == 0
    assert "Embedded 1 chunk" in result.output
    assert store.get_status_counts()["pending_embed"] == 0


def test_embed_scoped_to_collection(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "content a")
    _seed_doc(store, "b", "y.md", "content b")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed", "-c", "a"])

    assert result.exit_code == 0
    assert "1 document" in result.output


def test_embed_is_idempotent_without_force(monkeypatch):
    calls = []

    def counting_embed(texts, model, kind="query"):
        calls.append(texts)
        return _fake_embed(texts, model, kind)

    store = Store(":memory:", embed_fn=counting_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    runner.invoke(app, ["embed"])
    runner.invoke(app, ["embed"])

    assert len(calls) == 1


def test_embed_force_re_embeds(monkeypatch):
    calls = []

    def counting_embed(texts, model, kind="query"):
        calls.append(texts)
        return _fake_embed(texts, model, kind)

    store = Store(":memory:", embed_fn=counting_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    runner.invoke(app, ["embed"])
    result = runner.invoke(app, ["embed", "--force"])

    assert result.exit_code == 0
    assert len(calls) == 2
    assert "Force re-indexing: clearing all vectors..." in result.output


def test_embed_nothing_to_embed(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed"])

    assert "already have embeddings" in result.output


def test_embed_second_call_reports_nothing_to_embed(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    first = runner.invoke(app, ["embed"])
    second = runner.invoke(app, ["embed"])

    assert "Embedded 1 chunk" in first.output
    assert "already have embeddings" in second.output
    assert "Embedded" not in second.output


def test_embed_errors_on_unknown_collection(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed", "-c", "does-not-exist"])

    assert result.exit_code != 0
    assert result.stderr == "Collection not found: does-not-exist\n"


def test_embed_chunk_strategy_auto_threads_filepath(monkeypatch):
    calls = []

    class _SpyStore(Store):
        def index_content(
            self, content_hash, content, model=None, filepath=None, chunk_strategy="regex"
        ):
            calls.append((filepath, chunk_strategy))
            return super().index_content(
                content_hash, content, model=model, filepath=filepath, chunk_strategy=chunk_strategy
            )

    store = _SpyStore(":memory:", embed_fn=_fake_embed)
    store.add_collection("code", "/code")
    _seed_doc(store, "code", "sample.py", "def foo():\n    pass\n")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed", "--chunk-strategy", "auto"])

    assert result.exit_code == 0
    assert calls == [("sample.py", "auto")]


def test_embed_chunk_strategy_defaults_to_regex(monkeypatch):
    calls = []

    class _SpyStore(Store):
        def index_content(
            self, content_hash, content, model=None, filepath=None, chunk_strategy="regex"
        ):
            calls.append(chunk_strategy)
            return super().index_content(
                content_hash, content, model=model, filepath=filepath, chunk_strategy=chunk_strategy
            )

    store = _SpyStore(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "hello world")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed"])

    assert result.exit_code == 0
    assert calls == ["regex"]


def test_embed_chunk_strategy_rejects_invalid_value(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["embed", "--chunk-strategy", "bogus"])

    assert result.exit_code != 0


def test_embed_auto_after_regex_reembeds_stale_boundaries(monkeypatch):
    # Finding #1 end-to-end: a regex embed followed by
    # `embed --chunk-strategy auto` must do real re-embedding work, not
    # print "All content hashes already have embeddings."
    calls = []

    def counting_embed(texts, model, kind="query"):
        calls.append(texts)
        return _fake_embed(texts, model, kind)

    store = Store(":memory:", embed_fn=counting_embed)
    store.add_collection("code", "/code")
    body = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(400))
    _seed_doc(store, "code", "sample.py", body)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    first = runner.invoke(app, ["embed"])
    assert first.exit_code == 0
    calls_after_regex = len(calls)
    assert calls_after_regex > 0

    second = runner.invoke(app, ["embed", "--chunk-strategy", "auto"])
    assert second.exit_code == 0
    assert "already have embeddings" not in second.output
    assert len(calls) > calls_after_regex

    calls_after_second = len(calls)
    third = runner.invoke(app, ["embed", "--chunk-strategy", "auto"])
    assert "already have embeddings" in third.output
    assert len(calls) == calls_after_second
    calls_after_auto = len(calls)

    # A plain (regex) embed after an auto embed keeps the higher-quality
    # auto boundaries rather than "downgrading" them: the regex gate is
    # SQL-only by design, so auto work is never silently destroyed.
    fourth = runner.invoke(app, ["embed"])
    assert "already have embeddings" in fourth.output
    assert len(calls) == calls_after_auto


def _indexed_store(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")
    return store


def test_embed_prints_node_layout(tmp_path, monkeypatch):
    store = _indexed_store(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.monotonic", iter([10.0, 11.2]).__next__)

    result = runner.invoke(app, ["embed"])

    model = store.embed_model.rsplit("/", 1)[-1]
    assert result.exit_code == 0
    assert result.stdout == (
        f"Model: {model}\n\n"
        f"\r{'█' * 30} 100%{' ' * 36}\n"
        "\n✓ Done! Embedded 1 chunks from 1 documents in 1s\n"
    )


def test_embed_noop_prints_node_message(tmp_path, monkeypatch):
    store = _indexed_store(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    runner.invoke(app, ["embed"])

    result = runner.invoke(app, ["embed"])

    assert result.stdout == "✓ All content hashes already have embeddings.\n"


def test_embed_force_announces_clearing_first(tmp_path, monkeypatch):
    store = _indexed_store(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    runner.invoke(app, ["embed"])

    result = runner.invoke(app, ["embed", "--force"])

    assert result.stdout.startswith("Force re-indexing: clearing all vectors...\nModel: ")
    assert "✓ Done! Embedded 1 chunks from 1 documents in " in result.stdout


def test_embed_colors_match_node(tmp_path, monkeypatch):
    store = _indexed_store(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)

    first = runner.invoke(app, ["embed"], color=True)
    noop = runner.invoke(app, ["embed"], color=True)
    forced = runner.invoke(app, ["embed", "--force"], color=True)

    assert "\x1b[2mModel: " in first.stdout
    assert (
        "\x1b[32m✓ Done!\x1b[0m Embedded \x1b[1m1\x1b[0m chunks from "
        "\x1b[1m1\x1b[0m documents in \x1b[1m"
    ) in first.stdout
    assert noop.stdout == "\x1b[32m✓ All content hashes already have embeddings.\x1b[0m\n"
    assert forced.stdout.startswith("\x1b[33mForce re-indexing: clearing all vectors...\x1b[0m\n")


def test_embed_reports_progress_per_document_by_input_bytes(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("# A\nalpha")
    (tmp_path / "b.md").write_text("# B\nbravo café")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    events = []

    class RecordingProgress:
        def __init__(self, total_bytes):
            events.append(("init", total_bytes))

        def __enter__(self):
            events.append(("start",))
            return self

        def __exit__(self, *exc_info):
            events.append(("finish",))

        def update(self, bytes_processed, chunks_embedded):
            events.append(("update", bytes_processed, chunks_embedded))

    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.EmbedProgress", RecordingProgress)

    result = runner.invoke(app, ["embed"])

    assert result.exit_code == 0
    sizes = sorted(len(row["doc"].encode("utf-8")) for row in store.get_indexable_content(None))
    total = sum(sizes)
    assert events[0] == ("init", total)
    assert events[1] == ("start",)
    updates = [e for e in events if e[0] == "update"]
    assert [u[1] for u in updates][-1] == total
    assert [u[2] for u in updates] == [1, 2]
    assert events[-1] == ("finish",)


def test_embed_noop_prints_no_progress_bar(tmp_path, monkeypatch):
    store = _indexed_store(tmp_path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    runner.invoke(app, ["embed"])

    result = runner.invoke(app, ["embed"])

    assert "█" not in result.stdout
