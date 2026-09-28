from score_rerank_fixture import build_score_record, score_fixture


def test_build_score_record_zips_doc_ids_and_scores():
    result = build_score_record(["a", "b"], [0.9, 0.1])

    assert result == [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.1}]


def test_score_fixture_calls_rerank_per_query(monkeypatch):
    calls = []

    def fake_rerank(query, documents, model=None):
        calls.append((query, documents))
        return [1.0] * len(documents)

    monkeypatch.setattr("score_rerank_fixture.rerank", fake_rerank)

    fixture = [
        {
            "query_id": "q1",
            "query": "test query",
            "candidate_docs": [{"doc_id": "d1", "text": "text 1"}],
        }
    ]

    output = score_fixture(fixture)

    assert output["q1"]["scores"] == [{"doc_id": "d1", "score": 1.0}]
    assert output["q1"]["latency_ms"] >= 0
    assert calls == [("test query", ["text 1"])]
