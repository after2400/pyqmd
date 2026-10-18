import json
import math
import re
import zlib

import bench_breadcrumbs
import numpy as np
import pytest
from bench_breadcrumbs import (
    METRICS,
    PAGE_GUARD_METRICS,
    Chunk,
    Selection,
    build_chunks,
    check_exclusions,
    chunks_by_page,
    cluster_bootstrap,
    decide,
    embed_inputs,
    embed_queries,
    embed_variant,
    evidence_spans,
    format_report,
    load_dataset,
    query_metrics,
    run_experiment,
    select_questions,
    strict_evidence_spans,
    variant_title,
)


@pytest.mark.parametrize(
    ("variant", "path", "expected"),
    [
        ("base", ["A", "A1"], "Page"),
        ("full", ["A", "A1"], "Page > A > A1"),
        ("last", ["A", "A1"], "Page > A1"),
        ("full", [], "Page"),
        ("last", [], "Page"),
    ],
)
def test_variant_title(variant, path, expected):
    assert variant_title(variant, "Page", path) == expected


def test_variant_title_rejects_an_unknown_variant():
    with pytest.raises(ValueError):
        variant_title("nope", "Page", ["A"])


def fake_chunks(content, filename):
    """One chunk per body paragraph; heading-only paragraphs are skipped."""
    out, pos = [], 0
    for para in content.split("\n\n"):
        if para and not para.startswith("#"):
            out.append((para, pos))
        pos += len(para) + 2
    return out


PAGE = "# Alpha\n\n## Deadline\n\nYou have ten days.\n\n## Fees\n\nPay fees.\n"


def test_build_chunks_attaches_title_and_heading_path(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    chunks = build_chunks({"alpha": PAGE})
    assert [(c.text, c.title, c.path) for c in chunks] == [
        ("You have ten days.", "Alpha", ("Deadline",)),
        ("Pay fees.\n", "Alpha", ("Fees",)),
    ]
    assert chunks[0].pos == PAGE.index("You have")


def test_build_chunks_with_the_real_chunker_returns_one_chunk_for_a_short_page():
    chunks = build_chunks({"alpha": PAGE})
    assert len(chunks) == 1
    # One chunk spans the whole page; "## Deadline" holds the most characters.
    assert chunks[0].pos == 0 and chunks[0].text == PAGE
    assert chunks[0].path == ("Deadline",)


def test_build_chunks_sorts_by_page_so_each_pages_chunks_are_contiguous(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    chunks = build_chunks({"beta": PAGE, "alpha": PAGE})
    assert [c.page for c in chunks] == ["alpha", "alpha", "beta", "beta"]
    assert chunks_by_page(chunks) == {"alpha": [0, 1], "beta": [2, 3]}


def test_evidence_spans_find_every_occurrence_and_count_missing_items():
    content = "# T\n\nOne sentence.\n\n- A list item\n\nOne sentence.\n"
    spans, missing = evidence_spans(
        content, ["<p>One sentence.</p>", "<li>A list item</li>", "<p>Not there.</p>"]
    )
    assert sorted(spans) == sorted(
        [
            (content.index("One sentence."), content.index("One sentence.") + 13),
            (content.rindex("One sentence."), content.rindex("One sentence.") + 13),
            (content.index("A list item"), content.index("A list item") + 11),
        ]
    )
    assert missing == 1


def test_evidence_spans_normalise_entities_and_whitespace():
    # The corpus collapses whitespace; evidence is normalised the same way.
    spans, missing = evidence_spans(
        "Fish & chips cost £5.\n", ["<p>Fish &amp; chips\ncost £5.</p>"]
    )
    assert missing == 0 and spans == [(0, len("Fish & chips cost £5."))]


def test_evidence_spans_drops_unparseable_items():
    assert evidence_spans("text", ["<div>weird</div>", ""]) == ([], 2)


def dev_question(qid, url, evidences, *, not_answerable=False):
    return {
        "id": qid,
        "url": url,
        "question": f"question {qid}",
        "not_answerable": not_answerable,
        "evidences": evidences,
    }


def two_page_setup(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages = {"alpha": PAGE, "beta": PAGE}
    return pages, build_chunks(pages)


def test_select_questions_limits_gold_to_the_gold_pages_chunks(monkeypatch):
    pages, chunks = two_page_setup(monkeypatch)
    dev = [dev_question("d0", "https://www.gov.uk/beta", ["<p>You have ten days.</p>"])]
    selection = select_questions(dev, pages, chunks)
    # The sentence is on both pages; only beta's chunk (index 2) is gold.
    assert [(q.query_id, q.page, q.gold) for q in selection.questions] == [
        ("d0", "beta", frozenset({2}))
    ]


def test_select_questions_counts_every_exclusion_reason(monkeypatch):
    pages, chunks = two_page_setup(monkeypatch)
    dev = [
        dev_question("ok", "https://www.gov.uk/alpha", ["<p>Pay fees.</p>"]),
        dev_question("na", "https://www.gov.uk/alpha", ["<p>Pay fees.</p>"], not_answerable=True),
        dev_question("noev", "https://www.gov.uk/alpha", []),
        dev_question("nopage", "https://www.gov.uk/gamma", ["<p>Pay fees.</p>"]),
        dev_question("nested", "https://www.gov.uk/a/b", ["<p>Pay fees.</p>"]),
        dev_question("nomatch", "https://www.gov.uk/alpha", ["<p>Nowhere.</p>"]),
    ]
    selection = select_questions(dev, pages, chunks)
    assert [q.query_id for q in selection.questions] == ["ok"]
    assert (
        selection.unanswerable,
        selection.no_evidence,
        selection.page_missing,
        selection.unmatched,
    ) == (1, 1, 2, 1)
    assert selection.items_missing == 1
    assert selection.eligible == 4


def test_check_exclusions_fails_above_ten_percent_and_passes_below():
    q = [object()] * 9
    check_exclusions(Selection(questions=q, unmatched=1))  # 1 of 10: exactly 10%
    with pytest.raises(RuntimeError, match="excluded"):
        check_exclusions(Selection(questions=q, unmatched=2))


def test_chunk_is_a_plain_value():
    assert Chunk("p", 0, "t", "T", ()) == Chunk(page="p", pos=0, text="t", title="T", path=())


# Pages: A = chunks 0,1; B = chunk 2; C = chunks 3,4.
BOUNDS = np.array([0, 2, 3, 5])


def test_query_metrics_on_a_hand_made_ranking():
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.2])
    m = query_metrics(scores, BOUNDS, page_index=0, gold=[0])
    assert set(m) == set(METRICS)
    # Four chunks score above the gold chunk (0.1): rank 5.
    assert m["chunk_mrr"] == pytest.approx(0.2)
    assert (m["chunk_hit1"], m["chunk_hit5"], m["chunk_hit10"]) == (0.0, 1.0, 1.0)
    # Within page A the top chunk is index 1, not the gold one.
    assert m["in_page_hit1"] == 0.0
    # Page scores are [0.9, 0.5, 0.95]: page A is second.
    assert m["page_mrr"] == pytest.approx(0.5)
    assert m["page_ndcg10"] == pytest.approx(1 / math.log2(3))
    assert m["page_recall10"] == 1.0


def test_query_metrics_perfect_ranking():
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
    m = query_metrics(scores, BOUNDS, page_index=0, gold=[0])
    assert all(m[name] == 1.0 for name in METRICS)


def test_query_metrics_uses_the_best_of_several_gold_chunks():
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.2])
    m = query_metrics(scores, BOUNDS, page_index=0, gold=[0, 1])
    assert m["chunk_mrr"] == 1.0 and m["in_page_hit1"] == 1.0


def test_query_metrics_a_tie_with_the_gold_chunk_does_not_count_against_it():
    scores = np.array([0.5, 0.5, 0.1, 0.1, 0.1])
    m = query_metrics(scores, BOUNDS, page_index=0, gold=[1])
    # Ranks are 1 + the number of chunks scoring strictly higher.
    assert m["chunk_mrr"] == 1.0


def test_query_metrics_page_outside_the_top_ten_scores_zero_recall():
    bounds = np.arange(13)  # twelve one-chunk pages
    scores = np.linspace(1.0, 0.0, 12)
    m = query_metrics(scores, bounds, page_index=11, gold=[11])
    assert m["page_recall10"] == 0.0 and m["page_ndcg10"] == 0.0
    assert m["page_mrr"] == pytest.approx(1 / 12)


def test_cluster_bootstrap_constant_difference_has_a_degenerate_interval():
    values = np.ones(10)
    mean, lo, hi = cluster_bootstrap(values, [f"p{i}" for i in range(10)], resamples=500)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)


def test_cluster_bootstrap_interval_excludes_zero_for_a_consistent_gain():
    rng = np.random.default_rng(1)
    values = 0.3 + 0.05 * rng.standard_normal(60)
    _, lo, hi = cluster_bootstrap(values, [f"p{i % 30}" for i in range(60)], resamples=2000)
    assert 0 < lo < hi


def test_cluster_bootstrap_interval_includes_zero_when_the_effect_cancels():
    values = np.array([1.0, -1.0] * 15)
    mean, lo, hi = cluster_bootstrap(values, [f"p{i}" for i in range(30)], resamples=2000)
    assert mean == 0.0 and lo < 0 < hi


def test_cluster_bootstrap_resamples_whole_pages_not_questions():
    # One page holds four +1 questions, nine pages hold one -1 each. Resampling
    # questions would rarely produce a mean far from the pooled one; resampling
    # pages swings widely, so the interval must be wide.
    values = np.array([1.0] * 4 + [-1.0] * 9)
    clusters = ["big"] * 4 + [f"s{i}" for i in range(9)]
    _, lo, hi = cluster_bootstrap(values, clusters, resamples=3000)
    assert hi - lo > 0.8


def test_cluster_bootstrap_is_reproducible_for_a_seed():
    values = np.arange(20, dtype=float)
    clusters = [f"p{i % 7}" for i in range(20)]
    assert cluster_bootstrap(values, clusters, 500, seed=3) == cluster_bootstrap(
        values, clusters, 500, seed=3
    )


def zero_deltas(**override):
    deltas = dict.fromkeys(PAGE_GUARD_METRICS, 0.0)
    deltas.update(override)
    return deltas


def test_decide_yes_needs_a_positive_interval_and_no_page_regression():
    ok, reasons = decide((0.05, 0.01, 0.09), zero_deltas(), n_subset=59)
    assert ok is True and reasons


def test_decide_no_when_the_chunk_interval_includes_zero():
    ok, _ = decide((0.05, -0.01, 0.11), zero_deltas(), n_subset=59)
    assert ok is False


def test_decide_allows_a_drop_of_exactly_one_query_but_not_more():
    one = -1 / 59
    assert decide((0.05, 0.01, 0.09), zero_deltas(page_recall10=one), n_subset=59)[0] is True
    ok, reasons = decide((0.05, 0.01, 0.09), zero_deltas(page_recall10=one - 0.001), n_subset=59)
    assert ok is False and any("page_recall10" in r for r in reasons)


def test_decide_rejects_an_empty_subset():
    with pytest.raises(ValueError, match="quality-test"):
        decide((0.05, 0.01, 0.09), zero_deltas(), n_subset=0)


DIM = 1024


def bag_of_words(text):
    vec = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(token.encode()) % DIM] += 1.0
    return vec


class FakeEmbed:
    """Deterministic bag-of-words embedder: title and text both count, so a
    heading path in the title shows up in the vector."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, model, kind="query", title=None):
        self.calls.append((tuple(texts), model, kind, title))
        if kind == "query":
            return [bag_of_words(t) for t in texts]
        return [bag_of_words(f"{title} {t}") for t in texts]


def synthetic_corpus():
    pages = {
        name: f"# {name.title()}\n\n## Deadline\n\nYou have ten days.\n\n## Fees\n\nPay {name} fees.\n"
        for name in ("alpha", "beta", "gamma")
    }
    dev = [
        {
            "id": f"dev-{i}",
            "url": f"https://www.gov.uk/{name}",
            "question": f"{name} deadline",
            "not_answerable": False,
            "evidences": ["<p>You have ten days.</p>"],
        }
        for i, name in enumerate(("alpha", "beta", "gamma"))
    ]
    return pages, dev, {"dev-0", "dev-1", "dev-2"}


def test_embed_variant_embeds_each_distinct_title_in_one_call(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    chunks = build_chunks({"alpha": PAGE})
    fake = FakeEmbed()
    vectors, seconds = embed_variant(chunks, "full", "m", fake, cache_dir=None)
    assert vectors.shape == (2, DIM) and seconds >= 0
    assert sorted(c[3] for c in fake.calls) == ["Alpha > Deadline", "Alpha > Fees"]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_variant_base_uses_one_call_per_document_title(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    fake = FakeEmbed()
    embed_variant(build_chunks({"alpha": PAGE}), "base", "m", fake, cache_dir=None)
    assert [(c[2], c[3], len(c[0])) for c in fake.calls] == [("document", "Alpha", 2)]


def test_embed_variant_reuses_cached_vectors_but_not_for_a_changed_input(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    chunks = build_chunks({"alpha": PAGE})
    first = FakeEmbed()
    v1, _ = embed_variant(chunks, "full", "m", first, cache_dir=tmp_path)
    second = FakeEmbed()
    v2, _ = embed_variant(chunks, "full", "m", second, cache_dir=tmp_path)
    assert second.calls == [] and np.array_equal(v1, v2)
    # A different model id, or different text, must miss the cache.
    third = FakeEmbed()
    embed_variant(chunks, "full", "other-model", third, cache_dir=tmp_path)
    assert third.calls
    changed = build_chunks({"alpha": PAGE.replace("ten", "eleven")})
    fourth = FakeEmbed()
    embed_variant(changed, "full", "m", fourth, cache_dir=tmp_path)
    assert fourth.calls


def test_embed_queries_caches_and_normalises(tmp_path):
    fake = FakeEmbed()
    q1 = embed_queries(["alpha deadline", "beta fees"], "m", fake, tmp_path)
    again = FakeEmbed()
    q2 = embed_queries(["alpha deadline", "beta fees"], "m", again, tmp_path)
    assert again.calls == [] and np.array_equal(q1, q2)
    assert np.allclose(np.linalg.norm(q1, axis=1), 1.0)


def test_load_dataset_reads_corpus_dev_questions_and_quality_subset(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "raw").mkdir()
    for name in ("b", "a", "c"):
        (tmp_path / "corpus" / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    (tmp_path / "raw" / "dev.json").write_text(json.dumps([{"id": "dev-0"}]), encoding="utf-8")
    (tmp_path / "queries.json").write_text(
        json.dumps([{"query_id": "dev-0", "query": "q"}]), encoding="utf-8"
    )
    pages, dev, subset = load_dataset(tmp_path, max_pages=None)
    assert list(pages) == ["a", "b", "c"] and dev == [{"id": "dev-0"}] and subset == {"dev-0"}
    assert list(load_dataset(tmp_path, max_pages=2)[0]) == ["a", "b"]


def test_run_experiment_end_to_end_on_a_synthetic_corpus(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, subset = synthetic_corpus()
    result = run_experiment(
        pages, dev, subset, model="m", embed_fn=FakeEmbed(), cache_dir=None, resamples=200
    )
    # Without the heading path the fees chunk outranks the answer chunk
    # (it repeats the page name); with it, the answer chunk wins.
    assert result.means["base"]["chunk_mrr"] == pytest.approx(0.5)
    assert result.means["full"]["chunk_mrr"] == pytest.approx(1.0)
    assert result.diffs["full"]["chunk_mrr"][0] == pytest.approx(0.5)
    assert result.diffs["full"]["chunk_mrr"][1] > 0
    assert result.subset_n == 3 and result.decision is True
    assert result.n_chunks == 6
    mean_chars, max_chars, share = result.breadcrumb_stats["full"]
    # " > Deadline" is 11 characters, " > Fees" is 7.
    assert share == 1.0 and max_chars == 11 and mean_chars == pytest.approx(9.0)


def test_run_experiment_strict_mode_fails_when_evidence_cannot_be_located(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, subset = synthetic_corpus()
    for question in dev:
        question["evidences"] = ["<p>Never on the page.</p>"]
    with pytest.raises(RuntimeError, match="excluded"):
        run_experiment(pages, dev, subset, model="m", embed_fn=FakeEmbed(), cache_dir=None)


def test_run_experiment_fails_when_no_quality_subset_query_is_eligible(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    with pytest.raises(ValueError, match="quality-test"):
        run_experiment(pages, dev, set(), model="m", embed_fn=FakeEmbed(), cache_dir=None)


def test_run_experiment_smoke_mode_without_a_quality_subset_still_reports(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    result = run_experiment(
        pages,
        dev,
        set(),
        model="m",
        embed_fn=FakeEmbed(),
        cache_dir=None,
        resamples=100,
        strict=False,
    )
    assert result.subset_n == 0 and result.decision is False
    assert "smoke run" in result.reasons[0]
    assert "Decision: NO" in format_report(result)


def test_format_report_has_the_tables_and_the_decision(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, subset = synthetic_corpus()
    result = run_experiment(
        pages, dev, subset, model="m", embed_fn=FakeEmbed(), cache_dir=None, resamples=200
    )
    report = format_report(result)
    for expected in ("chunk_mrr", "in_page_hit1", "base", "full", "last", "95%", "Decision: YES"):
        assert expected in report


def test_strict_evidence_spans_keep_an_item_that_occurs_once():
    content = "# T\n\nYou must pay.\n\nThe form is online.\n"
    spans, missing, ambiguous = strict_evidence_spans(content, ["<p>The form is online.</p>"])
    start = content.index("The form")
    assert spans == [(start, start + len("The form is online."))]
    assert (missing, ambiguous) == (0, 0)


def test_strict_evidence_spans_drop_an_item_that_occurs_more_than_once():
    content = "You must:\n\n- a\n\nYou must:\n\n- b\n"
    assert strict_evidence_spans(content, ["<p>You must:</p>"]) == ([], 0, 1)


def test_strict_evidence_spans_match_on_word_boundaries():
    content = "You pay now.\n\nYou repay later.\n\nThe payment is due.\n"
    spans, missing, ambiguous = strict_evidence_spans(content, ["<p>pay</p>"])
    start = content.index("pay now")
    assert spans == [(start, start + 3)] and (missing, ambiguous) == (0, 0)


def test_strict_evidence_spans_count_missing_and_unparseable():
    assert strict_evidence_spans("text", ["<div>x</div>", "<p>nope</p>"]) == ([], 2, 0)


REPEATED = "# Alpha\n\n## A\n\nPay fees.\n\n## B\n\nPay fees.\n\nOnly here.\n"


def test_select_questions_strict_excludes_a_question_with_only_ambiguous_evidence(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages = {"alpha": REPEATED}
    chunks = build_chunks(pages)
    dev = [
        dev_question("amb", "https://www.gov.uk/alpha", ["<p>Pay fees.</p>"]),
        dev_question("mix", "https://www.gov.uk/alpha", ["<p>Pay fees.</p>", "<p>Only here.</p>"]),
    ]
    strict = select_questions(dev, pages, chunks)
    assert [q.query_id for q in strict.questions] == ["mix"]
    # Only the unique item is gold: the third chunk, "Only here."
    assert strict.questions[0].gold == frozenset({2})
    assert (strict.ambiguous, strict.items_ambiguous, strict.unmatched) == (1, 2, 0)
    assert strict.eligible == 2


def test_select_questions_all_labelling_marks_every_occurrence(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages = {"alpha": REPEATED}
    chunks = build_chunks(pages)
    dev = [dev_question("amb", "https://www.gov.uk/alpha", ["<p>Pay fees.</p>"])]
    selection = select_questions(dev, pages, chunks, labelling="all")
    assert selection.questions[0].gold == frozenset({0, 1})
    assert selection.ambiguous == 0


def test_select_questions_rejects_an_unknown_labelling(monkeypatch):
    pages, chunks = two_page_setup(monkeypatch)
    with pytest.raises(ValueError, match="labelling"):
        select_questions([], pages, chunks, labelling="nope")


def test_check_exclusions_counts_ambiguous_questions():
    q = [object()] * 8
    with pytest.raises(RuntimeError, match="excluded"):
        check_exclusions(Selection(questions=q, ambiguous=2))
    check_exclusions(Selection(questions=q, ambiguous=2), max_fraction=0.25)


def test_decide_uses_an_absolute_tolerance_when_given():
    ci = (0.05, 0.01, 0.09)
    ok, _ = decide(ci, zero_deltas(page_mrr=-0.005), n_subset=2000, tolerance=0.01)
    assert ok is True
    ok, reasons = decide(ci, zero_deltas(page_mrr=-0.02), n_subset=2000, tolerance=0.01)
    assert ok is False and any("page_mrr" in r for r in reasons)


def test_decide_names_the_guard_set_in_its_reasons():
    _, reasons = decide(
        (0.05, 0.01, 0.09), zero_deltas(), n_subset=10, tolerance=0.01, label="scored questions"
    )
    assert any("scored questions" in r for r in reasons)


def test_run_experiment_guard_tolerance_covers_all_scored_questions(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    result = run_experiment(
        pages,
        dev,
        set(),
        model="m",
        embed_fn=FakeEmbed(),
        cache_dir=None,
        resamples=200,
        guard_tolerance=0.01,
    )
    assert result.guard_all is True and result.subset_n == 3 and result.decision is True
    assert "Guard set" in format_report(result)


def test_run_experiment_defaults_to_strict_labelling_and_reports_it(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, subset = synthetic_corpus()
    result = run_experiment(
        pages, dev, subset, model="m", embed_fn=FakeEmbed(), cache_dir=None, resamples=100
    )
    assert result.labelling == "strict" and result.guard_all is False
    assert "strict" in format_report(result)


def test_run_experiment_accepts_all_labelling(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, subset = synthetic_corpus()
    result = run_experiment(
        pages,
        dev,
        subset,
        model="m",
        embed_fn=FakeEmbed(),
        cache_dir=None,
        resamples=100,
        labelling="all",
    )
    assert result.labelling == "all"


def test_load_dataset_train_split_reads_train_json_and_no_quality_subset(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "corpus" / "a.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "raw" / "train.json").write_text(json.dumps([{"id": "train-0"}]), encoding="utf-8")
    pages, dev, subset = load_dataset(tmp_path, max_pages=None, split="train")
    assert dev == [{"id": "train-0"}] and subset == set() and list(pages) == ["a"]


def test_load_dataset_names_a_missing_split_file(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError, match="train.json"):
        load_dataset(tmp_path, max_pages=None, split="train")


def test_load_dataset_rejects_an_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_dataset(tmp_path, max_pages=None, split="test")


def test_cache_misses_when_the_embedding_input_format_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    chunks = build_chunks({"alpha": PAGE})
    embed_variant(chunks, "full", "m", FakeEmbed(), cache_dir=tmp_path)
    bench_breadcrumbs.embed_queries(["q"], "m", FakeEmbed(), tmp_path)
    monkeypatch.setattr(bench_breadcrumbs, "embedding_fingerprint", lambda model: "changed")
    after_format_change = FakeEmbed()
    embed_variant(chunks, "full", "m", after_format_change, cache_dir=tmp_path)
    assert after_format_change.calls
    queries_after = FakeEmbed()
    bench_breadcrumbs.embed_queries(["q"], "m", queries_after, tmp_path)
    assert queries_after.calls


def test_embed_inputs_groups_by_title_and_returns_normalised_rows_in_input_order():
    fake = FakeEmbed()
    vectors, seconds = embed_inputs(
        ["T1", "T2", "T1"], ["a", "b", "c"], "lbl", "m", fake, cache_dir=None
    )
    assert vectors.shape[0] == 3 and seconds >= 0
    assert sorted((c[3], c[0]) for c in fake.calls) == [("T1", ("a", "c")), ("T2", ("b",))]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_inputs_caches_by_label_and_misses_when_a_text_changes(tmp_path):
    embed_inputs(["T"], ["a"], "lbl", "m", FakeEmbed(), cache_dir=tmp_path)
    again = FakeEmbed()
    embed_inputs(["T"], ["a"], "lbl", "m", again, cache_dir=tmp_path)
    assert again.calls == []
    changed = FakeEmbed()
    embed_inputs(["T"], ["a changed"], "lbl", "m", changed, cache_dir=tmp_path)
    assert changed.calls


def test_cluster_bootstrap_a_wider_level_gives_a_wider_interval():
    values = np.random.default_rng(1).random(80)
    clusters = [f"page{i // 2}" for i in range(80)]
    _m, lo95, hi95 = cluster_bootstrap(values, clusters, resamples=2000)
    _m, lo975, hi975 = cluster_bootstrap(values, clusters, resamples=2000, level=0.975)
    assert lo975 <= lo95 and hi975 >= hi95 and (lo975, hi975) != (lo95, hi95)
