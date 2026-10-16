import json

from bench_contextual import (
    ContextCache,
    build_prompt,
    clean_context,
    context_key,
    ctx_text,
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
