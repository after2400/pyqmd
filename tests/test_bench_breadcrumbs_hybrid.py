import math
import re
import zlib
from types import SimpleNamespace

import pytest
from bench_breadcrumbs_hybrid import (
    build_store,
    hybrid_ranked_pages,
    no_expansion,
    page_rank_metrics,
    verdict,
)

DIM = 1024


def bag_of_words(text):
    vec = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(token.encode()) % DIM] += 1.0
    return vec


def fake_embed(texts, model, kind="query", title=None):
    if kind == "query":
        return [bag_of_words(t) for t in texts]
    titles = title if isinstance(title, list) else [title] * len(texts)
    return [bag_of_words(f"{t} {text}") for text, t in zip(texts, titles, strict=True)]


def test_page_rank_metrics_for_a_hit_at_rank_two():
    m = page_rank_metrics(["x", "gold", "y"], "gold")
    assert m["page_mrr"] == pytest.approx(0.5)
    assert m["page_ndcg10"] == pytest.approx(1 / math.log2(3))
    assert m["page_recall10"] == 1.0


def test_page_rank_metrics_for_a_hit_at_rank_one():
    assert page_rank_metrics(["gold"], "gold") == {
        "page_mrr": 1.0,
        "page_ndcg10": 1.0,
        "page_recall10": 1.0,
    }


def test_page_rank_metrics_outside_the_top_ten_has_no_recall_or_ndcg():
    ranked = [f"p{i}" for i in range(11)] + ["gold"]
    m = page_rank_metrics(ranked, "gold")
    assert m["page_recall10"] == 0.0 and m["page_ndcg10"] == 0.0
    assert m["page_mrr"] == pytest.approx(1 / 12)


def test_page_rank_metrics_for_a_missing_page_are_all_zero():
    assert page_rank_metrics(["a", "b"], "gold") == {
        "page_mrr": 0.0,
        "page_ndcg10": 0.0,
        "page_recall10": 0.0,
    }


def test_hybrid_ranked_pages_asks_the_store_for_fusion_without_reranking():
    calls = []

    def query(q, **kwargs):
        calls.append((q, kwargs))
        return [SimpleNamespace(file="a/guide.md"), SimpleNamespace(file="notes.md")]

    store = SimpleNamespace(query=query)
    assert hybrid_ranked_pages(store, "how to") == ["guide", "notes"]
    assert calls == [("how to", {"limit": 40, "skip_rerank": True})]


def test_no_expansion_returns_no_lines():
    assert no_expansion("anything", "model") == []


def _corpus(tmp_path):
    (tmp_path / "alpha.md").write_text(
        "# Alpha\n\n## Deadline\n\nYou have ten days to reply.\n", encoding="utf-8"
    )
    (tmp_path / "beta.md").write_text(
        "# Beta\n\n## Fees\n\nThe fee is twenty pounds.\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize("breadcrumbs", [True, False])
def test_build_store_indexes_every_page_and_ranks_the_matching_one_first(tmp_path, breadcrumbs):
    store = build_store(
        _corpus(tmp_path), breadcrumbs=breadcrumbs, embed_fn=fake_embed, model="fake-model"
    )
    assert store.count_pending_embed() == 0
    assert hybrid_ranked_pages(store, "ten days to reply")[0] == "alpha"
    assert hybrid_ranked_pages(store, "fee in pounds")[0] == "beta"
    store.close()


def test_build_store_respects_the_breadcrumbs_mode(tmp_path):
    seen = []

    def recording(texts, model, kind="query", title=None):
        if kind == "document":
            seen.append(title)
        return fake_embed(texts, model, kind, title)

    on = build_store(_corpus(tmp_path), breadcrumbs=True, embed_fn=recording, model="m")
    on.close()
    assert any(isinstance(t, list) for t in seen)
    seen.clear()
    off = build_store(_corpus(tmp_path), breadcrumbs=False, embed_fn=recording, model="m")
    off.close()
    assert seen and all(isinstance(t, str) for t in seen)


def test_verdict_one_when_the_ndcg_interval_is_above_zero():
    diffs = {
        "page_mrr": (0.01, -0.002, 0.02),
        "page_ndcg10": (0.012, 0.003, 0.021),
        "page_recall10": (0.014, 0.001, 0.027),
    }
    assert verdict(diffs)[0] == 1


def test_verdict_two_when_the_interval_spans_zero_without_a_regression():
    diffs = {
        "page_mrr": (0.002, -0.01, 0.014),
        "page_ndcg10": (0.004, -0.008, 0.016),
        "page_recall10": (0.0, -0.01, 0.01),
    }
    assert verdict(diffs)[0] == 2


def test_verdict_three_when_any_metric_point_estimate_is_below_minus_one_percent():
    diffs = {
        "page_mrr": (0.01, 0.002, 0.02),
        "page_ndcg10": (0.012, 0.003, 0.021),
        "page_recall10": (-0.011, -0.03, 0.008),
    }
    # An interval above zero on the primary metric does not outrank a regression.
    assert verdict(diffs)[0] == 3


def test_verdict_exactly_minus_one_percent_is_not_a_regression():
    diffs = {
        "page_mrr": (-0.01, -0.02, 0.0),
        "page_ndcg10": (0.0, -0.01, 0.01),
        "page_recall10": (0.0, -0.01, 0.01),
    }
    assert verdict(diffs)[0] == 2
