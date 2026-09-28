import json

import pytest

from pyqmd_mlx.bench._fixture import load_fixture


def _write_fixture(tmp_path, data):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _valid_fixture_data():
    return {
        "description": "Example fixture",
        "version": 1,
        "collection": "notes",
        "queries": [
            {
                "id": "exact-api",
                "query": "API versioning",
                "type": "exact",
                "description": "Direct keyword match",
                "expected_files": ["api-design.md"],
                "expected_in_top_k": 1,
            }
        ],
    }


def test_load_fixture_parses_valid_fixture(tmp_path):
    path = _write_fixture(tmp_path, _valid_fixture_data())

    fixture = load_fixture(path)

    assert fixture.description == "Example fixture"
    assert fixture.version == 1
    assert fixture.collection == "notes"
    assert len(fixture.queries) == 1
    q = fixture.queries[0]
    assert q.id == "exact-api"
    assert q.query == "API versioning"
    assert q.type == "exact"
    assert q.expected_files == ["api-design.md"]
    assert q.expected_in_top_k == 1


def test_load_fixture_defaults_missing_collection_to_none(tmp_path):
    data = _valid_fixture_data()
    del data["collection"]
    path = _write_fixture(tmp_path, data)

    fixture = load_fixture(path)

    assert fixture.collection is None


def test_load_fixture_rejects_missing_queries_array(tmp_path):
    path = _write_fixture(tmp_path, {"description": "x", "version": 1})

    with pytest.raises(ValueError, match="queries"):
        load_fixture(path)


def test_load_fixture_rejects_malformed_json(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_fixture(str(path))


def test_load_fixture_rejects_query_missing_required_field(tmp_path):
    data = _valid_fixture_data()
    del data["queries"][0]["expected_in_top_k"]
    path = _write_fixture(tmp_path, data)

    with pytest.raises(ValueError, match="expected_in_top_k"):
        load_fixture(path)


def test_load_fixture_rejects_structured_query(tmp_path):
    data = _valid_fixture_data()
    data["queries"][0]["query"] = "lex: API versioning"
    path = _write_fixture(tmp_path, data)

    with pytest.raises(ValueError, match="exact-api"):
        load_fixture(path)


def test_load_fixture_rejects_structured_query_on_any_line(tmp_path):
    data = _valid_fixture_data()
    data["queries"][0]["query"] = "API versioning\nvec: something else"
    path = _write_fixture(tmp_path, data)

    with pytest.raises(ValueError, match="structured"):
        load_fixture(path)
