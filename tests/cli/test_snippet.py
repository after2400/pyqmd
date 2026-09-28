from pyqmd_mlx.cli._snippet import extract_snippet


def test_extract_snippet_finds_best_matching_line():
    body = "line one\nline two mentions authentication\nline three\nline four"
    result = extract_snippet(body, "authentication")
    assert "authentication" in result.snippet
    assert result.line == 2


def test_extract_snippet_header_format():
    body = "\n".join(f"line {i}" for i in range(10))
    result = extract_snippet(body, "line 5")
    assert result.snippet.startswith("@@ -")
    assert "before" in result.snippet
    assert "after" in result.snippet


def test_extract_snippet_truncates_long_snippets():
    body = "match here\n" + ("x" * 1000)
    result = extract_snippet(body, "match", max_len=50)
    lines = result.snippet.split("\n", 1)
    assert len(lines[1]) <= 50
    assert lines[1].endswith("...")


def test_extract_snippet_chunk_pos_narrows_search_and_offsets_line_number():
    filler = "unrelated text\n" * 50
    target = "the authentication setup is here"
    body = filler + target + "\n" + ("more filler\n" * 50)
    chunk_pos = len(filler)

    result = extract_snippet(body, "authentication", chunk_pos=chunk_pos, chunk_len=100)
    assert result.line == 51  # filler is 50 lines (0-indexed line 50 -> 1-indexed 51)


def test_extract_snippet_chunk_pos_zero_with_no_match_falls_back_to_full_body():
    body = "no match in the narrow window\n" + ("filler\n" * 20) + "authentication mention here"
    result = extract_snippet(body, "authentication", chunk_pos=0, chunk_len=10)
    assert "authentication" in result.snippet


def test_extract_snippet_intent_terms_contribute_to_scoring():
    body = "line one plain\nline two mentions deployment\nline three plain\nline four plain"
    # Query term alone doesn't appear anywhere; intent term does.
    result = extract_snippet(body, "nonexistentterm", intent="deployment steps")
    assert result.line == 2


def test_extract_snippet_no_match_anywhere_defaults_to_first_lines():
    body = "alpha\nbeta\ngamma\ndelta"
    result = extract_snippet(body, "nonexistentterm")
    assert result.line == 1


def test_extract_snippet_lines_before_and_after_are_correct():
    body = "\n".join(f"line {i}" for i in range(20))
    result = extract_snippet(body, "line 10")
    assert result.lines_before >= 0
    assert result.lines_after >= 0
    assert result.lines_before + result.snippet_lines + result.lines_after == 20
