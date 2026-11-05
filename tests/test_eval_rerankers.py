"""scripts/eval_rerankers.py: the median-matching fit, the metrics summary, the SciFact
loader and the BM25 candidate pool. No models are loaded."""

import json
import math

import pytest
from eval_rerankers import (
    Bm25Index,
    build_index,
    build_pool,
    evaluate_model,
    load_scifact,
    match_medians,
    read_doc,
    summarize,
)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# --- match_medians ----------------------------------------------------------


def test_match_medians_hits_the_target_medians():
    relevant = [1.0, 2.0, 3.0]  # median 2.0
    other = [0.0, 0.5, 1.0]  # median 0.5

    bias, temperature = match_medians(relevant, other, target_relevant=0.99, target_other=0.01)

    assert _sigmoid((2.0 - bias) / temperature) == pytest.approx(0.99)
    assert _sigmoid((0.5 - bias) / temperature) == pytest.approx(0.01)
    assert temperature > 0


def test_match_medians_reproduces_a_known_mapping():
    bias, temperature = 0.7, 1.5
    relevant = [3.0, 4.0, 5.0]
    other = [-2.0, -1.0, 0.0]
    target_relevant = _sigmoid((4.0 - bias) / temperature)
    target_other = _sigmoid((-1.0 - bias) / temperature)

    fitted = match_medians(relevant, other, target_relevant, target_other)

    assert fitted == pytest.approx((bias, temperature))


def test_match_medians_rejects_classes_whose_medians_are_not_ordered():
    with pytest.raises(ValueError, match="median"):
        match_medians([0.1, 0.2], [1.0, 2.0], 0.99, 0.01)


@pytest.mark.parametrize("targets", [(0.01, 0.99), (0.5, 0.5), (1.0, 0.1), (0.9, 0.0)])
def test_match_medians_rejects_unusable_targets(targets):
    with pytest.raises(ValueError, match="target"):
        match_medians([2.0, 3.0], [0.0, 1.0], *targets)


def test_match_medians_needs_both_classes():
    with pytest.raises(ValueError, match="both"):
        match_medians([], [0.0, 1.0], 0.99, 0.01)


# --- summarize --------------------------------------------------------------


def test_summarize_perfect_ranking():
    rankings = {"q1": ["a", "b", "c"], "q2": ["x", "y"]}
    qrels = {"q1": {"a"}, "q2": {"x"}}

    assert summarize(rankings, qrels) == {"mrr": 1.0, "ndcg_at_10": 1.0, "recall_at_10": 1.0}


def test_summarize_relevant_document_ranked_last_of_twenty():
    ranking = [f"d{i}" for i in range(19)] + ["gold"]

    result = summarize({"q": ranking}, {"q": {"gold"}})

    assert result["mrr"] == pytest.approx(1 / 20)
    assert result["recall_at_10"] == 0.0
    assert result["ndcg_at_10"] == 0.0


# --- SciFact loader ---------------------------------------------------------


def _write_scifact(root):
    (root / "corpus").mkdir(parents=True)
    docs = {
        "d1": "# Cats\n\nCats purr and sleep all day.",
        "d2": "# Dogs\n\nDogs bark at the mailman.",
        "d3": "# Fish\n\nFish swim in the sea.",
    }
    for doc_id, text in docs.items():
        (root / "corpus" / f"{doc_id}.md").write_text(text + "\n")
    (root / "queries.jsonl").write_text(
        "\n".join(
            json.dumps({"_id": qid, "text": text, "metadata": {}})
            for qid, text in [("1", "why do cats purr"), ("2", "unjudged query")]
        )
        + "\n"
    )
    (root / "qrels-test.tsv").write_text("query-id\tcorpus-id\tscore\n1\td1\t1\n1\td3\t0\n")
    return root


def test_load_scifact_keeps_only_judged_test_queries_and_positive_qrels(tmp_path):
    queries, qrels = load_scifact(_write_scifact(tmp_path))

    assert queries == {"1": "why do cats purr"}
    assert qrels == {"1": {"d1"}}


def test_read_doc_returns_the_markdown_file(tmp_path):
    root = _write_scifact(tmp_path)
    assert read_doc(root, "d2").startswith("# Dogs")


# --- candidate pool ---------------------------------------------------------


def test_build_pool_is_bm25_hits_then_missing_gold_without_duplicates(tmp_path):
    index = build_index(_write_scifact(tmp_path))

    pool = build_pool(index, "cats purr", gold_ids={"d1", "d3"}, size=5)

    assert pool[0] == "d1"  # the BM25 hit comes first
    assert "d3" in pool  # a gold document with no term overlap is still pooled
    assert len(pool) == len(set(pool))


def test_build_pool_matches_any_query_term_not_all():
    index = Bm25Index({"d1": "cats purr", "d2": "dogs bark", "d3": "fish swim"})

    # A long claim never contains every term of any one document.
    assert set(index.top("cats dogs fish zebras", 3)) == {"d1", "d2", "d3"}


def test_bm25_ranks_the_document_with_more_and_rarer_matches_first():
    index = Bm25Index(
        {
            "common": "the the the cat sat",
            "rare": "the quokka sat",
            "other": "the dog ran",
        }
    )

    assert index.top("quokka sat", 3)[0] == "rare"


def test_build_pool_respects_the_size_for_bm25_hits(tmp_path):
    index = build_index(_write_scifact(tmp_path))

    pool = build_pool(index, "cats dogs fish", gold_ids=set(), size=1)

    assert len(pool) == 1


def test_build_pool_with_no_matching_terms_is_just_the_gold(tmp_path):
    index = build_index(_write_scifact(tmp_path))

    assert build_pool(index, "zzzz", gold_ids={"d2"}, size=5) == ["d2"]


# --- evaluate_model ---------------------------------------------------------


def test_evaluate_model_ranks_by_score_and_prefixes_intent():
    seen = []

    def scorer(query, documents):
        seen.append(query)
        return [1.0 if "gold" in d else 0.0 for d in documents]

    rows, rankings = evaluate_model(
        scorer,
        queries={"q": "find it"},
        pools={"q": ["a", "b", "c"]},
        texts={"a": "no", "b": "gold text", "c": "no"},
        qrels={"q": {"b"}},
        intent="science",
    )

    assert rankings == {"q": ["b", "a", "c"]}
    assert seen == ["science\n\nfind it"]
    assert [(r["doc_id"], r["label"], r["score"]) for r in rows] == [
        ("a", 0, 0.0),
        ("b", 1, 1.0),
        ("c", 0, 0.0),
    ]
    assert all(r["query_id"] == "q" for r in rows)
