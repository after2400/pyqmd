import json
import sys
import types

import prepare_beir_corpus


def test_prepare_writes_the_scifact_layout(tmp_path, monkeypatch):
    source = tmp_path / "hub"
    (source / "qrels").mkdir(parents=True)
    (source / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T", "text": "body"}) + "\n", encoding="utf-8"
    )
    (source / "queries.jsonl").write_text('{"_id": "q1", "text": "a question"}\n', encoding="utf-8")
    (source / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\n")
    requested = []

    def fake_download(repo, filename, repo_type):
        requested.append((repo, filename, repo_type))
        return str(source / filename)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download)
    )
    out = tmp_path / "out"

    assert prepare_beir_corpus.prepare("nfcorpus", out) == 1

    assert (out / "corpus" / "d1.md").read_text() == "# T\n\nbody\n"
    assert (out / "queries.jsonl").read_text().startswith('{"_id": "q1"')
    assert (out / "qrels-test.tsv").read_text().splitlines()[1] == "q1\td1\t2"
    assert {r[0] for r in requested} == {"mteb/nfcorpus"}
    assert {r[2] for r in requested} == {"dataset"}
