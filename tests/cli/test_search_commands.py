import json

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.search import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _fake_rerank(query, documents, model):
    return [1.0 if query.lower() in doc.lower() else 0.1 for doc in documents]


def _seed_doc(store, path, title, body):
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document("notes", path, title, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.index_content(h, body, model="fake-model")
    return h


def test_search_command_finds_fts_match(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert len(parsed) >= 1
    assert parsed[0]["title"] == "Auth"


def test_search_command_json_file_field_is_qmd_uri_prefixed(monkeypatch):
    """Matches Node's real, current --format json behavior (toQmdPath() in
    src/cli/qmd.ts) -- a bare "collection/path" string was a real, confirmed
    parity bug."""
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["file"] == "qmd://notes/auth.md"


def test_vsearch_command_finds_vector_match(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    _seed_doc(store, "cooking.md", "Cooking", "pasta recipe instructions")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["vsearch", "login credentials for auth", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["title"] == "Auth"


def test_vsearch_command_json_file_field_is_qmd_uri_prefixed(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["vsearch", "login credentials for auth", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["file"] == "qmd://notes/auth.md"


@pytest.mark.requires_expansion_weights
def test_query_command_returns_hybrid_result(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["title"] == "Auth"


@pytest.mark.requires_expansion_weights
def test_query_command_json_file_field_is_qmd_uri_prefixed(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["file"] == "qmd://notes/auth.md"


@pytest.mark.requires_expansion_weights
def test_query_command_uses_shared_default_limit_of_20(monkeypatch):
    from pyqmd_mlx.cli.commands.search import DEFAULT_SEARCH_LIMIT

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    for i in range(DEFAULT_SEARCH_LIMIT + 5):
        _seed_doc(store, f"widget-{i}.md", f"Widget {i}", f"widget widget widget number {i}")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "widget", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert len(parsed) == DEFAULT_SEARCH_LIMIT


@pytest.mark.requires_expansion_weights
def test_query_command_no_rerank_skips_reranker(monkeypatch):
    calls = []

    def counting_rerank(query, documents, model):
        calls.append(documents)
        return _fake_rerank(query, documents, model)

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=counting_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--no-rerank", "--format", "json"])

    assert result.exit_code == 0
    assert calls == []


@pytest.mark.requires_expansion_weights
def test_query_command_accepts_intent_flag(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app, ["query", "authentication", "--intent", "deployment", "--format", "json"]
    )

    assert result.exit_code == 0


def test_search_command_respects_min_score(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app, ["search", "authentication", "--min-score", "0.99", "--format", "json"]
    )

    parsed = json.loads(result.stdout)
    assert parsed == []


def test_search_command_rejects_unknown_format_cleanly(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--format", "yaml"])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # A raw Python traceback would include the source line/module of the
    # ValueError; Typer's own invalid-choice error does not.
    assert "Traceback" not in result.output


def test_search_accepts_repeated_collection_flag(monkeypatch):
    captured = {}

    class FakeStore:
        def search_fts(self, query, limit=20, collection=None, filter=None):
            captured["collection"] = collection
            return []

    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda: FakeStore())

    from typer.testing import CliRunner

    from pyqmd_mlx.cli.commands.search import app

    runner = CliRunner()
    result = runner.invoke(app, ["search", "foo", "-c", "notes", "-c", "journals"])
    assert result.exit_code == 0
    assert captured["collection"] == ["notes", "journals"]


def test_search_with_no_collection_flag_passes_none(monkeypatch):
    captured = {}

    class FakeStore:
        def search_fts(self, query, limit=20, collection=None, filter=None):
            captured["collection"] = collection
            return []

    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda: FakeStore())

    from typer.testing import CliRunner

    from pyqmd_mlx.cli.commands.search import app

    runner = CliRunner()
    result = runner.invoke(app, ["search", "foo"])
    assert result.exit_code == 0
    assert captured["collection"] is None


def test_query_accepts_repeated_collection_flag(monkeypatch):
    captured = {}

    class FakeStore:
        def query(
            self,
            query,
            limit=10,
            min_score=0.0,
            collection=None,
            skip_rerank=False,
            intent=None,
            filter=None,
            chunk_strategy="regex",
        ):
            captured["collection"] = collection
            return []

    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda: FakeStore())

    from typer.testing import CliRunner

    from pyqmd_mlx.cli.commands.search import app

    runner = CliRunner()
    result = runner.invoke(app, ["query", "foo", "-c", "notes", "-c", "journals"])
    assert result.exit_code == 0
    assert captured["collection"] == ["notes", "journals"]


@pytest.mark.requires_expansion_weights
def test_query_command_filter_narrows_results(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    published = "---\nqmd:\n  metadata:\n    status: published\n---\nauthentication setup guide"
    draft = "---\nqmd:\n  metadata:\n    status: draft\n---\nauthentication setup guide"
    for path, title, content in (
        ("a.md", "Auth Published", published),
        ("b.md", "Auth Draft", draft),
    ):
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        doc_id = store.insert_document(
            "notes", path, title, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.sync_document_metadata(doc_id, content, path)
        store.index_content(h, content, model="fake-model")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app,
        [
            "query",
            "authentication",
            "--filter",
            '{"key":"status","operator":"eq","value":"published"}',
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert [r["title"] for r in parsed] == ["Auth Published"]


def test_search_command_filter_narrows_results(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    published = "---\nqmd:\n  metadata:\n    status: published\n---\nauthentication setup guide"
    draft = "---\nqmd:\n  metadata:\n    status: draft\n---\nauthentication setup guide"
    for path, title, content in (
        ("a.md", "Auth Published", published),
        ("b.md", "Auth Draft", draft),
    ):
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        doc_id = store.insert_document(
            "notes", path, title, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.sync_document_metadata(doc_id, content, path)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app,
        [
            "search",
            "authentication",
            "--filter",
            '{"key":"status","operator":"eq","value":"published"}',
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert [r["title"] for r in parsed] == ["Auth Published"]


def test_vsearch_command_filter_narrows_results(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    published = "---\nqmd:\n  metadata:\n    status: published\n---\nauthentication setup guide"
    draft = "---\nqmd:\n  metadata:\n    status: draft\n---\nauthentication setup guide"
    for path, title, content in (
        ("a.md", "Auth Published", published),
        ("b.md", "Auth Draft", draft),
    ):
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        doc_id = store.insert_document(
            "notes", path, title, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.sync_document_metadata(doc_id, content, path)
        store.index_content(h, content, model="fake-model")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(
        app,
        [
            "vsearch",
            "authentication",
            "--filter",
            '{"key":"status","operator":"eq","value":"published"}',
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert [r["title"] for r in parsed] == ["Auth Published"]


def test_query_command_invalid_filter_json_exits_non_zero(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--filter", "not json"])

    assert result.exit_code != 0
    assert "Invalid --filter JSON" in result.output


def test_search_command_json_includes_context_when_configured(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["context"] == "Notes root context"


def test_vsearch_command_json_includes_context_when_configured(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["vsearch", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["context"] == "Notes root context"


@pytest.mark.requires_expansion_weights
def test_query_command_json_includes_context_when_configured(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["context"] == "Notes root context"


def test_search_full_path_swaps_display_and_drops_docid(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text("authentication setup guide")
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", str(tmp_path))
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--full-path", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["file"] == str((tmp_path / "auth.md").resolve())
    assert "docid" not in parsed[0]


def test_search_full_path_falls_back_and_warns_when_file_missing(monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/nonexistent-dir")
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["search", "authentication", "--full-path", "--format", "json"])

    assert result.exit_code == 0
    assert "could not be resolved" in result.output
    # The stderr warning and stdout JSON are merged into one stream by
    # CliRunner's default mix_stderr=True, with the warning written first --
    # slice from the JSON array's opening bracket to parse just the payload.
    parsed = json.loads(result.output[result.output.index("[") :])
    assert parsed[0]["file"] == "qmd://notes/auth.md"
    assert parsed[0]["docid"].startswith("#")


@pytest.mark.requires_expansion_weights
def test_query_full_path_swaps_display_and_drops_docid(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text("authentication setup guide")
    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", str(tmp_path))
    _seed_doc(store, "auth.md", "Auth", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication", "--full-path", "--format", "json"])

    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["file"] == str((tmp_path / "auth.md").resolve())
    assert "docid" not in parsed[0]


@pytest.mark.requires_expansion_weights
def test_query_chunk_strategy_auto_threads_to_store(monkeypatch):
    calls = []

    class _SpyStore(Store):
        def query(self, query, **kwargs):
            calls.append(kwargs.get("chunk_strategy"))
            return super().query(query, **kwargs)

    store = _SpyStore(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("code", "/code")
    _seed_doc(store, "sample.py", "Sample", "def foo():\n    pass\n")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "foo", "--chunk-strategy", "auto"])

    assert result.exit_code == 0
    assert calls == ["auto"]


@pytest.mark.requires_expansion_weights
def test_query_chunk_strategy_defaults_to_regex(monkeypatch):
    calls = []

    class _SpyStore(Store):
        def query(self, query, **kwargs):
            calls.append(kwargs.get("chunk_strategy"))
            return super().query(query, **kwargs)

    store = _SpyStore(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "a.md", "A", "authentication setup guide")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication"])

    assert result.exit_code == 0
    assert calls == ["regex"]


def _failing_expand(query, model):
    from pyqmd_mlx.llm import ExpansionModelError

    raise ExpansionModelError(
        f"Could not load query-expansion model '{model}': boom. Set PYQMD_EXPAND_MODEL "
        "to a Hugging Face repo id or a local MLX model directory to override."
    )


def test_query_command_reports_expansion_model_error_cleanly(monkeypatch):
    store = Store(
        ":memory:",
        embed_fn=_fake_embed,
        rerank_fn=_fake_rerank,
        expand_fn=_failing_expand,
        expand_model="bad/model",
    )
    store.add_collection("notes", "/notes")
    _seed_doc(store, "cooking.md", "Cooking", "pasta recipe instructions")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.search.get_store", lambda db_path=None: store)

    result = runner.invoke(app, ["query", "authentication"])

    assert result.exit_code == 1
    assert "Error: Could not load query-expansion model 'bad/model'" in result.output
    assert "Traceback" not in result.output
