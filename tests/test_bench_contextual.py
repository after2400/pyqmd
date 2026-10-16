import json
import os
import re
import zlib

import bench_breadcrumbs
import numpy as np
import pytest
from bench_breadcrumbs import Chunk, build_chunks
from bench_breadcrumbs_rerank import Logger
from bench_contextual import (
    GENERATORS,
    ContextCache,
    arm_inputs,
    build_parser,
    build_prompt,
    cached_contexts,
    chunk_prompt,
    clean_context,
    context_key,
    contextual_verdict,
    contrast,
    ctx_text,
    generate_contexts,
    regressions,
    score_run,
    window_text,
)

DIM = 1024


def bag_of_words(text):
    vec = [0.0] * DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(token.encode()) % DIM] += 1.0
    return vec


class FakeEmbed:
    """Deterministic bag-of-words embedder (title and text both count)."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, model, kind="query", title=None):
        self.calls.append((tuple(texts), model, kind, title))
        if kind == "query":
            return [bag_of_words(t) for t in texts]
        return [bag_of_words(f"{title} {t}") for t in texts]


def fake_chunks(content, filename):
    """One chunk per body paragraph; heading-only paragraphs are skipped."""
    out, pos = [], 0
    for para in content.split("\n\n"):
        if para and not para.startswith("#"):
            out.append((para, pos))
        pos += len(para) + 2
    return out


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


def test_window_text_returns_a_short_page_whole():
    assert window_text("short page", 3, 4) == "short page"


def test_window_text_is_centred_on_the_chunk_and_never_longer_than_the_limit():
    # Every 6-digit block is unique, so the window can be located in the page.
    page = "".join(f"{i:06d}" for i in range(9_000))
    window = window_text(page, pos=30_000, length=2_000, max_chars=10_000)
    assert len(window) == 10_000
    start = page.index(window)
    assert start <= 30_000 and start + 10_000 >= 32_000
    assert abs((start + 5_000) - (30_000 + 1_000)) <= 1


def test_window_text_clamps_at_the_page_edges():
    page = "x" * 50_000
    assert len(window_text(page, 0, 500, max_chars=10_000)) == 10_000
    assert len(window_text(page, 49_500, 500, max_chars=10_000)) == 10_000


def test_build_prompt_contains_page_chunk_and_the_fixed_instruction():
    prompt = build_prompt("THE PAGE", "THE CHUNK")
    assert "<document>\nTHE PAGE\n</document>" in prompt
    assert "<chunk>\nTHE CHUNK\n</chunk>" in prompt
    assert prompt.rstrip().endswith("Answer only with the succinct context and nothing else.")


def test_clean_context_strips_a_reasoning_block_and_collapses_whitespace():
    raw = "<think>\nlet me see\n</think>\n\n  Covers the  appeal\ndeadline.  "
    assert clean_context(raw) == "Covers the appeal deadline."


def test_clean_context_strips_an_unclosed_reasoning_block_to_empty():
    assert clean_context("<think>still thinking and never finished") == ""


def test_clean_context_keeps_pipes_and_non_ascii():
    assert clean_context("Rate | £50 per week – see café") == "Rate | £50 per week – see café"


def test_ctx_text_prepends_the_context_or_returns_the_plain_chunk_when_empty():
    assert ctx_text("About X.", "chunk body") == "About X.\n\nchunk body"
    assert ctx_text("", "chunk body") == "chunk body"


def test_context_key_changes_with_the_generator_and_the_prompt():
    base = context_key("gen-a", "prompt")
    assert base == context_key("gen-a", "prompt")
    assert base != context_key("gen-b", "prompt") and base != context_key("gen-a", "prompt2")


def test_context_cache_round_trips_and_survives_a_torn_last_line(tmp_path):
    path = tmp_path / "ctx.jsonl"
    cache = ContextCache(path)
    cache.put("k1", "line one\nwith a newline")
    cache.put("k2", "£ café")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "k3", "context": "tor')  # killed mid-write
    reloaded = ContextCache(path)
    assert reloaded.get("k2") == "£ café" and reloaded.get("k1") == "line one\nwith a newline"
    assert reloaded.get("k3") is None
    reloaded.put("k4", "after the tear")
    assert ContextCache(path).get("k4") == "after the tear"
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # the tear was dropped, so every line parses


def test_context_cache_without_a_path_is_in_memory():
    cache = ContextCache(None)
    cache.put("k", "v")
    assert cache.get("k") == "v"


PAGES = {"alpha": "# Alpha\n\n## Deadline\n\nYou have ten days.\n", "beta": "# Beta\n\ntext b\n"}
CHUNKS = [
    Chunk("alpha", 0, "# Alpha\n\n## Deadline\n\nYou have ten days.\n", "Alpha", ("Deadline",)),
    Chunk("beta", 0, "# Beta\n\ntext b\n", "Beta", ()),
]


def counting_generator(calls, answer="Context line."):
    def generate(prompt):
        calls.append(prompt)
        return answer

    return generate


def make(chunks=CHUNKS, generator="gen", fn=None, cache=None, **kwargs):
    cache = ContextCache(None) if cache is None else cache
    made = generate_contexts(
        chunks, PAGES, generator, fn or counting_generator([]), cache, Logger(None), **kwargs
    )
    return made, cache


def test_chunk_prompt_uses_the_pages_window_and_the_chunk():
    prompt = chunk_prompt(CHUNKS[1], PAGES)
    assert "text b" in prompt and "<chunk>\n# Beta\n\ntext b\n\n</chunk>" in prompt


def test_generate_contexts_writes_every_chunk_to_the_cache():
    calls = []
    made, cache = make(fn=counting_generator(calls))
    assert made == 2 and len(calls) == 2
    assert cached_contexts(CHUNKS, PAGES, "gen", cache) == ["Context line.", "Context line."]


def test_generate_contexts_resumes_without_regenerating_cached_chunks(tmp_path):
    path = tmp_path / "c.jsonl"
    make(cache=ContextCache(path), limit=1)
    calls = []
    made, _ = make(fn=counting_generator(calls), cache=ContextCache(path))
    assert made == 1 and len(calls) == 1 and "text b" in calls[0]


def test_generate_contexts_limit_processes_only_the_first_chunks():
    calls = []
    make(fn=counting_generator(calls), limit=1)
    assert len(calls) == 1


def test_a_different_generator_does_not_reuse_cached_lines():
    _, cache = make(generator="gen-a")
    calls = []
    make(generator="gen-b", fn=counting_generator(calls), cache=cache)
    assert len(calls) == 2


def test_generate_contexts_cleans_what_the_model_returned():
    fn = counting_generator([], "<think>x</think>\nAbout  the\nappeal.")
    _, cache = make(fn=fn)
    assert cached_contexts(CHUNKS, PAGES, "gen", cache)[0] == "About the appeal."


def test_an_empty_cleaned_context_is_cached_and_not_regenerated():
    _, cache = make(fn=counting_generator([], "<think>no end"))
    calls = []
    make(fn=counting_generator(calls), cache=cache)
    assert calls == [] and cached_contexts(CHUNKS, PAGES, "gen", cache) == ["", ""]


def test_cached_contexts_names_how_many_chunks_are_missing():
    _, cache = make(limit=1)
    with pytest.raises(ValueError, match=r"1 of 2 chunks.*generate"):
        cached_contexts(CHUNKS, PAGES, "gen", cache)


def test_generate_contexts_logs_progress_in_chunks(capsys):
    make(log_every=1)
    out = capsys.readouterr().out
    assert "2/2 chunks" in out and "questions" not in out


def test_arm_inputs_builds_the_four_arms_from_the_chunks_and_contexts():
    arms = arm_inputs(CHUNKS, {"small": ["About A.", ""], "mid": ["Mid A.", "Mid B."]})
    assert list(arms) == ["base", "crumbs", "ctx-small", "ctx-mid"]
    titles, texts = arms["base"]
    assert titles == ["Alpha", "Beta"] and texts == [c.text for c in CHUNKS]
    assert arms["crumbs"][0] == ["Alpha > Deadline", "Beta"]
    assert arms["ctx-small"][0] == ["Alpha", "Beta"]  # plain titles in the ctx arms
    assert arms["ctx-small"][1][0] == "About A.\n\n" + CHUNKS[0].text
    assert arms["ctx-small"][1][1] == CHUNKS[1].text  # empty context: the plain chunk


def test_contrast_is_a_minus_b_at_97_5_percent():
    rows = {
        "a": {"chunk_mrr": np.array([1.0, 1.0, 0.5, 0.5])},
        "b": {"chunk_mrr": np.array([0.5, 0.5, 0.5, 0.5])},
    }
    result = contrast(rows, "a", "b", ["p", "p", "q", "q"], 500, metrics=("chunk_mrr",))
    mean, lo, hi = result["chunk_mrr"]
    assert mean == pytest.approx(0.25) and lo <= mean <= hi


def test_verdict_one_needs_a_positive_bound_over_crumbs_and_no_ndcg_loss():
    vs_crumbs = {"chunk_mrr": (0.02, 0.005, 0.03), "page_ndcg10": (0.01, -0.01, 0.02)}
    vs_base = {"chunk_mrr": (0.04, 0.02, 0.06)}
    assert contextual_verdict(vs_crumbs, vs_base)[0] == 1


def test_verdict_one_fails_when_ndcg_is_lower_than_crumbs():
    vs_crumbs = {"chunk_mrr": (0.02, 0.005, 0.03), "page_ndcg10": (-0.001, -0.01, 0.01)}
    vs_base = {"chunk_mrr": (0.04, 0.02, 0.06)}
    assert contextual_verdict(vs_crumbs, vs_base)[0] == 2


def test_verdict_two_when_it_beats_base_but_not_crumbs():
    vs_crumbs = {"chunk_mrr": (0.0, -0.01, 0.01), "page_ndcg10": (0.0, -0.01, 0.01)}
    vs_base = {"chunk_mrr": (0.02, 0.01, 0.03)}
    assert contextual_verdict(vs_crumbs, vs_base)[0] == 2


def test_verdict_three_otherwise():
    vs_crumbs = {"chunk_mrr": (-0.01, -0.02, 0.0), "page_ndcg10": (0.0, -0.01, 0.01)}
    vs_base = {"chunk_mrr": (0.0, -0.01, 0.01)}
    assert contextual_verdict(vs_crumbs, vs_base)[0] == 3


def test_regressions_flags_a_point_estimate_below_minus_one_percent_against_base():
    vs_base = {
        "chunk_mrr": (0.02, 0.0, 0.04),
        "page_mrr": (-0.0101, -0.02, 0.0),
        "page_ndcg10": (-0.01, -0.02, 0.0),
    }
    assert regressions(vs_base) == ["page_mrr"]


def _filled_caches(pages):
    chunks = build_chunks(pages)
    caches = {}
    for size in ("small", "mid"):
        caches[size] = ContextCache(None)
        generate_contexts(
            chunks,
            pages,
            GENERATORS[size],
            counting_generator([], f"about {size}"),
            caches[size],
            Logger(None),
        )
    return caches


def test_score_run_end_to_end_reports_a_verdict_per_size(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    report = score_run(
        pages,
        dev,
        model="m",
        embed_fn=FakeEmbed(),
        cache_dir=None,
        caches=_filled_caches(pages),
        resamples=200,
    )
    assert "Verdict (small)" in report and "Verdict (mid)" in report
    assert "ctx-small" in report and "ctx-mid" in report and "crumbs" in report
    assert "empty contexts: 0" in report


def test_score_run_refuses_to_embed_when_a_generator_is_incomplete(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    caches = _filled_caches(pages)
    caches["mid"] = ContextCache(None)
    embedder = FakeEmbed()
    with pytest.raises(ValueError, match=r"6 of 6 chunks.*generate"):
        score_run(
            pages,
            dev,
            model="m",
            embed_fn=embedder,
            cache_dir=None,
            caches=caches,
            resamples=100,
        )
    assert embedder.calls == []


def test_score_run_counts_chunks_whose_context_came_back_empty(monkeypatch):
    monkeypatch.setattr(bench_breadcrumbs, "embedding_chunks", fake_chunks)
    pages, dev, _ = synthetic_corpus()
    chunks = build_chunks(pages)
    caches = {"small": ContextCache(None)}
    generate_contexts(
        chunks,
        pages,
        GENERATORS["small"],
        counting_generator([], "<think>x"),
        caches["small"],
        Logger(None),
    )
    report = score_run(
        pages, dev, model="m", embed_fn=FakeEmbed(), cache_dir=None, caches=caches, resamples=100
    )
    assert "empty contexts: 6" in report


def test_common_options_are_accepted_after_the_subcommand():
    args = build_parser().parse_args(["generate", "--generator", "small", "--max-pages", "10"])
    assert args.command == "generate" and args.max_pages == 10
    args = build_parser().parse_args(["score", "--max-pages", "10", "--generators", "small"])
    assert args.generators == ["small"] and args.max_pages == 10
    assert build_parser().parse_args(["score"]).max_pages is None


def test_generate_contexts_closing_line_reports_the_measured_rate(capsys):
    make()
    out = capsys.readouterr().out
    assert "2 chunks have a context (2 generated now" in out and "s/chunk" in out


def test_generate_contexts_closing_line_has_no_rate_when_nothing_was_generated(capsys):
    _, cache = make()
    capsys.readouterr()
    make(cache=cache)
    out = capsys.readouterr().out
    assert "0 generated now" in out and "s/chunk" not in out


def test_an_intact_context_cache_is_not_rewritten_on_load(tmp_path):
    path = tmp_path / "ctx.jsonl"
    ContextCache(path).put("k", "v")
    path.chmod(0o444)  # a rewrite would raise PermissionError
    try:
        assert ContextCache(path).get("k") == "v"
    finally:
        path.chmod(0o644)


def test_a_torn_context_cache_is_repaired_with_an_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "ctx.jsonl"
    ContextCache(path).put("k", "v")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "torn')
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append(b), real_replace(a, b)))
    ContextCache(path)
    assert replaced == [path]
    assert not list(tmp_path.glob("*.tmp"))
