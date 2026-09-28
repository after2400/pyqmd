from pyqmd_mlx.store._chunking import (
    CHUNK_OVERLAP_CHARS,
    chunk_document,
    find_best_cutoff,
    find_code_fences,
    is_inside_code_fence,
    scan_break_points,
)
from pyqmd_mlx.store._types import BreakPoint, CodeFenceRegion


def test_scan_break_points_finds_headings():
    text = "intro\n# Heading One\nbody text\n## Heading Two\nmore body"
    points = scan_break_points(text)
    types_at_pos = {p.pos: p.type for p in points}
    # "\n# Heading One" match starts at the newline before '#'
    h1_pos = text.index("\n# Heading One")
    h2_pos = text.index("\n## Heading Two")
    assert types_at_pos[h1_pos] == "h1"
    assert types_at_pos[h2_pos] == "h2"


def test_scan_break_points_sorted_by_position():
    text = "a\n\nb\n# C\nd"
    points = scan_break_points(text)
    positions = [p.pos for p in points]
    assert positions == sorted(positions)


def test_find_code_fences_detects_paired_fence():
    text = "before\n```\ncode here\n```\nafter"
    fences = find_code_fences(text)
    assert len(fences) == 1
    assert text[fences[0].start : fences[0].end] == "\n```\ncode here\n```"


def test_find_code_fences_ignores_lone_unmatched_marker():
    # A trailing unmatched ``` is ignored rather than treated as a fence
    # running to end-of-file: a lone marker (e.g. an unbalanced example in
    # a docstring) would otherwise swallow every candidate cut point for
    # the rest of the document. Deliberate divergence from Node's
    # store.ts, which extends an unclosed fence to EOF.
    text = "before\n```\nunterminated code"
    assert find_code_fences(text) == []


def test_chunk_document_with_lone_fence_marker_still_splits():
    # Regression test for the review finding: a 6215-char code file with
    # one stray ``` marker collapsed to a single chunk under
    # chunk_strategy="auto" (fence region ran to EOF, swallowing every
    # regex *and* AST cut point). Now it must split normally.
    doc = '"""Module docs with a stray ``` marker in an example."""\n' + "\n".join(
        f"def func_{i}():\n    return {i}\n" for i in range(200)
    )
    assert len(doc) > 6000  # sanity: long enough to force a split
    auto_chunks = chunk_document(doc, filepath="sample.py", chunk_strategy="auto")
    regex_chunks = chunk_document(doc, filepath="sample.py", chunk_strategy="regex")
    assert len(auto_chunks) > 1
    assert len(regex_chunks) > 1


def test_is_inside_code_fence():
    fences = [CodeFenceRegion(start=5, end=20)]
    assert is_inside_code_fence(10, fences) is True
    assert is_inside_code_fence(5, fences) is False  # boundary excluded
    assert is_inside_code_fence(20, fences) is False  # boundary excluded
    assert is_inside_code_fence(25, fences) is False


def test_find_best_cutoff_prefers_higher_scoring_break_point():
    # Two break points in the window; the higher-scored one, even if
    # farther from the target, should win over a low-score one right at
    # the target (squared-distance decay is gentle early on).
    points = [
        BreakPoint(pos=50, score=100, type="h1"),
        BreakPoint(pos=95, score=1, type="newline"),
    ]
    cutoff = find_best_cutoff(points, target_char_pos=100, window_chars=100)
    assert cutoff == 50


def test_find_best_cutoff_falls_back_to_target_with_no_break_points():
    cutoff = find_best_cutoff([], target_char_pos=100, window_chars=50)
    assert cutoff == 100


def test_find_best_cutoff_skips_points_inside_code_fences():
    points = [BreakPoint(pos=50, score=100, type="h1")]
    fences = [CodeFenceRegion(start=40, end=60)]
    cutoff = find_best_cutoff(points, target_char_pos=100, window_chars=100, code_fences=fences)
    assert cutoff == 100  # only break point available is inside the fence


def test_chunk_document_returns_whole_content_when_under_limit():
    text = "short document"
    chunks = chunk_document(text, max_chars=1000)
    assert chunks == [(text, 0)]


def test_chunk_document_splits_long_content_at_headings():
    section = "x" * 50
    text = f"# One\n{section}\n# Two\n{section}\n# Three\n{section}"
    chunks = chunk_document(text, max_chars=70, overlap_chars=0, window_chars=30)
    assert len(chunks) > 1
    # every chunk position is within bounds and text matches a slice of the source
    for chunk_text, pos in chunks:
        assert text[pos : pos + len(chunk_text)] == chunk_text


def test_chunk_document_never_splits_inside_code_fence():
    fence_body = "```\n" + ("code line\n" * 10) + "```"
    text = f"# Heading\nintro text here\n{fence_body}\nmore text after"
    chunks = chunk_document(text, max_chars=40, overlap_chars=5, window_chars=20)
    fence_start = text.index("```")
    fence_end = text.rindex("```") + 3
    for _, pos in chunks:
        assert not (fence_start < pos < fence_end)


def test_chunk_document_overlap_moves_start_backward():
    text = "# A\n" + ("y" * 100) + "\n# B\n" + ("z" * 100)
    chunks = chunk_document(text, max_chars=60, overlap_chars=CHUNK_OVERLAP_CHARS, window_chars=20)
    assert len(chunks) >= 2
    # second chunk's start position should be less than first chunk's end
    first_end = chunks[0][1] + len(chunks[0][0])
    assert chunks[1][1] < first_end


def test_chunk_document_regression_no_slivers_before_large_fence():
    """Regression test: clamping backward to fence.start caused a cascade of
    near-empty "sliver" chunks (1-10 chars) before finally clearing a large
    fence. This test ensures we extend forward instead, avoiding slivers."""
    # Short intro followed by large code fence: classic sliver-producing case
    intro = "# Header\nShort intro\n"
    fence_body = "```\n" + ("code line\n" * 200) + "```"
    text = intro + fence_body + "\n\nMore content after"

    # Use production-realistic defaults
    chunks = chunk_document(text, max_chars=3600, overlap_chars=540, window_chars=800)

    fence_start = text.index("```")
    fence_end = text.rindex("```") + 3

    # Assertion 1: No chunk should start strictly inside the fence
    for chunk_text, pos in chunks:
        assert not (fence_start < pos < fence_end), (
            f"Chunk at position {pos} starts inside fence ({fence_start}-{fence_end})"
        )

    # Assertion 2: No chunk before the last one should be absurdly small
    # (the sliver signature). Check all but the last chunk.
    for i, (chunk_text, pos) in enumerate(chunks[:-1]):
        chunk_size = len(chunk_text)
        assert chunk_size >= 20, (
            f"Sliver detected: chunk {i} at position {pos} is only {chunk_size} chars"
        )


def test_chunk_document_regex_strategy_unchanged_by_default():
    # A code file long enough to force a split. With chunk_strategy="regex"
    # (the default), behavior must be identical whether or not filepath is
    # given -- filepath is only consulted when chunk_strategy="auto".
    body = "\n".join(f"def func_{i}():\n    pass\n" for i in range(400))
    without_filepath = chunk_document(body, max_chars=800)
    with_filepath = chunk_document(body, max_chars=800, filepath="sample.py")
    assert without_filepath == with_filepath


def test_chunk_document_auto_strategy_splits_at_function_boundary():
    body = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(60))

    regex_chunks = chunk_document(body, max_chars=500, filepath="sample.py", chunk_strategy="regex")
    auto_chunks = chunk_document(body, max_chars=500, filepath="sample.py", chunk_strategy="auto")
    assert len(auto_chunks) > 1 and len(regex_chunks) > 1  # sanity: long enough to split

    # AST-aware chunking has more/better candidate break points (every
    # function start, not just newlines), so the split positions differ
    # from the regex-only result on code content.
    assert [pos for _text, pos in auto_chunks] != [pos for _text, pos in regex_chunks]
    # Every non-final auto chunk ends exactly at a "def " boundary -- the
    # whole point of the feature. (Chunk *starts* can't align to
    # boundaries: consecutive chunks overlap by overlap_chars, so each
    # chunk after the first begins mid-previous-chunk. The observable
    # alignment is at chunk ends, where find_best_cutoff lands on the
    # score-90 ast:func point instead of the adjacent score-1 newline.)
    for text, pos in auto_chunks[:-1]:
        assert body[pos + len(text) :].startswith("def ")
    # Regex-only ends land on the newline *before* the def instead.
    for text, pos in regex_chunks[:-1]:
        assert not body[pos + len(text) :].startswith("def ")


def test_chunk_document_auto_strategy_without_filepath_falls_back_to_regex():
    body = "\n".join(f"def func_{i}():\n    pass\n" for i in range(60))
    auto_no_filepath = chunk_document(body, max_chars=500, chunk_strategy="auto")
    regex = chunk_document(body, max_chars=500, chunk_strategy="regex")
    assert auto_no_filepath == regex


def test_chunk_document_rejects_invalid_chunk_strategy():
    # Literal["regex", "auto"] is a type-checker hint only -- a typo like
    # "Auto" must raise at runtime instead of silently chunking as regex.
    body = "x" * 5000
    for bad in ["Auto", "bogus", "", "REGEX"]:
        try:
            chunk_document(body, chunk_strategy=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_dedup_break_points_keeps_highest_score_and_sorts():
    from pyqmd_mlx.store._chunking import dedup_break_points

    points = [
        BreakPoint(pos=50, score=1, type="newline"),
        BreakPoint(pos=5, score=20, type="blank"),
        BreakPoint(pos=50, score=100, type="h1"),
    ]
    deduped = dedup_break_points(points)
    assert [(p.pos, p.score) for p in deduped] == [(5, 20), (50, 100)]
