import json

import parity._snapshot_io as snapshot_io


def test_load_flow_raw_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_io, "_NODE_REF_DIR", tmp_path)
    assert snapshot_io.load_flow_raw("scifact", "collection_lifecycle") is None


def test_load_flow_raw_reads_file(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_io, "_NODE_REF_DIR", tmp_path)
    path = tmp_path / "scifact" / "cli_flow_raw" / "collection_lifecycle.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"placeholders": {}, "steps": {}}))
    assert snapshot_io.load_flow_raw("scifact", "collection_lifecycle") == {
        "placeholders": {},
        "steps": {},
    }
