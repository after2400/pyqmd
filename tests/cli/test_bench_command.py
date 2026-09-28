import json
import re

from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.bench import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _fake_expand(query, model, salt=None):
    return [f"lex: {query} setup", f"vec: how to {query}"]


def _fake_rerank(query, documents, model):
    return [1.0 if query.lower() in doc.lower() else 0.1 for doc in documents]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")


def _write_fixture(tmp_path, collection=None):
    data = {
        "description": "test fixture",
        "version": 1,
        "queries": [
            {
                "id": "q1",
                "query": "authentication",
                "type": "exact",
                "description": "",
                "expected_files": ["auth.md"],
                "expected_in_top_k": 1,
            }
        ],
    }
    if collection is not None:
        data["collection"] = collection
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _seeded_store():
    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand, rerank_fn=_fake_rerank)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "pasta recipe instructions")
    return store


def test_bench_prints_table_by_default(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path])

    assert result.exit_code == 0
    assert "q1" in result.output
    assert "full" in result.output
    assert "Summary" in result.output


def test_bench_json_produces_valid_json(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path, "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["results"][0]["id"] == "q1"
    assert "full" in parsed["summary"]


def test_bench_uses_fixture_collection_when_no_flag_given(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path, collection="notes")

    result = runner.invoke(app, ["bench", fixture_path])

    assert result.exit_code == 0


def test_bench_flag_overrides_fixture_collection(tmp_path, monkeypatch):
    store = _seeded_store()
    store.add_collection("other", "/other")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path, collection="other")

    result = runner.invoke(app, ["bench", fixture_path, "-c", "notes"])

    assert result.exit_code == 0
    assert "q1" in result.output


def test_bench_fails_fast_for_unknown_collection(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path, collection="nonexistent")

    result = runner.invoke(app, ["bench", fixture_path])

    assert result.exit_code == 1
    assert "Collection not found" in result.output


def test_bench_fails_fast_for_empty_index(tmp_path, monkeypatch):
    store = Store(":memory:", embed_fn=_fake_embed)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path])

    assert result.exit_code == 1
    assert "No indexed documents" in result.output


def _failing_expand(query, model):
    from pyqmd_mlx.llm import ExpansionModelError

    raise ExpansionModelError(f"Could not load query-expansion model '{model}': boom")


def test_bench_reports_expansion_model_error_cleanly(tmp_path, monkeypatch):
    store = Store(
        ":memory:",
        embed_fn=_fake_embed,
        expand_fn=_failing_expand,
        rerank_fn=_fake_rerank,
        expand_model="bad/model",
    )
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "pasta recipe instructions")
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path])

    assert result.exit_code == 1
    assert "Error: Could not load query-expansion model 'bad/model'" in result.output
    assert "Traceback" not in result.output


def _mask_latency(text):
    return re.sub(r" *\d+ms", " Nms", text)


def _zero_volatile(d):
    d = json.loads(json.dumps(d))
    d["timestamp"] = ""
    for qr in d["results"]:
        for score in qr["backends"].values():
            score["latency_ms"] = 0
    for s in d["summary"].values():
        s["avg_latency_ms"] = 0
    return d


def test_bench_samples_one_matches_default_text(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    default = runner.invoke(app, ["bench", fixture_path])
    one = runner.invoke(app, ["bench", fixture_path, "--samples", "1"])

    assert one.exit_code == 0
    assert _mask_latency(one.stdout) == _mask_latency(default.stdout)
    assert one.stderr == ""


def test_bench_samples_one_matches_default_json(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    default = json.loads(runner.invoke(app, ["bench", fixture_path, "--json"]).stdout)
    one = json.loads(runner.invoke(app, ["bench", fixture_path, "--json", "--samples", "1"]).stdout)

    assert _zero_volatile(one) == _zero_volatile(default)
    assert "samples" not in one


def test_bench_samples_table_shows_means_and_ranges(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path, "--samples", "3"])

    assert result.exit_code == 0
    out = result.stdout
    assert out.startswith("Samples: 3 (hybrid/full: mean min–max; ms from sample 0)\n")
    cell = r"\d\.\d\d \d\.\d\d–\d\.\d\d"
    assert re.search(rf"^q1 +hybrid +{cell} +{cell} +{cell} +\d+ms$", out, re.M)
    assert re.search(rf"^q1 +full +{cell} +{cell} +{cell} +\d+ms$", out, re.M)
    assert re.search(r"^q1 +bm25 +\d\.\d\d +\d\.\d\d +\d\.\d\d +\d+ms$", out, re.M)
    rng = r"\(\d\.\d{3}–\d\.\d{3}\)"
    assert re.search(rf"^  hybrid +Recall@k=\d\.\d{{3}} {rng} MRR=\d\.\d{{3}} {rng} ", out, re.M)
    assert re.search(r"^  bm25 +Recall@k=\d\.\d{3} MRR=", out, re.M)


def test_bench_samples_columns_line_up(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    out = runner.invoke(app, ["bench", fixture_path, "--samples", "2"]).stdout

    rows = [line for line in out.splitlines() if line.startswith(("Query", "q1"))]
    assert len(rows) == 5
    assert len({len(line) for line in rows}) == 1


def test_bench_samples_json_has_every_draw(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path, "--json", "--samples", "3"])

    parsed = json.loads(result.stdout)
    assert parsed["samples"] == 3
    assert len(parsed["results"][0]["samples"]["hybrid"]) == 3
    assert "min_mrr" in parsed["summary"]["full"]


def test_bench_samples_progress_goes_to_stderr_only(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path, "--json", "--samples", "2"])

    assert result.stderr == "bench: sample 1/2…\nbench: sample 2/2…\n"
    json.loads(result.stdout)  # stdout is pure JSON


def test_bench_rejects_samples_below_one(tmp_path, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr("pyqmd_mlx.cli.commands.bench.get_store", lambda db_path=None: store)
    fixture_path = _write_fixture(tmp_path)

    result = runner.invoke(app, ["bench", fixture_path, "--samples", "0"])

    assert result.exit_code == 1
    assert "Error: --samples must be at least 1" in result.output


def test_bench_checks_samples_before_loading_the_fixture(tmp_path):
    result = runner.invoke(app, ["bench", str(tmp_path / "missing.json"), "--samples", "0"])

    assert result.exit_code == 1
    assert "Error: --samples must be at least 1" in result.output
