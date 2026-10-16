import json

import pytest
from bench_breadcrumbs import Chunk
from bench_breadcrumbs_rerank import Logger
from bench_contextual import (
    ContextCache,
    build_prompt,
    cached_contexts,
    chunk_prompt,
    clean_context,
    context_key,
    ctx_text,
    generate_contexts,
    window_text,
)


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
