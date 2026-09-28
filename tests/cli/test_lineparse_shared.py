from pyqmd_mlx.cli._lineparse import parse_line_range


def test_parse_line_range_extracts_line_and_count_suffix():
    identifier, from_line, max_lines = parse_line_range("notes/a.md:100:40", None, None)
    assert identifier == "notes/a.md"
    assert from_line == 100
    assert max_lines == 40


def test_parse_line_range_extracts_line_only_suffix():
    identifier, from_line, max_lines = parse_line_range("notes/a.md:100", None, None)
    assert identifier == "notes/a.md"
    assert from_line == 100
    assert max_lines is None


def test_parse_line_range_no_suffix_leaves_identifier_untouched():
    identifier, from_line, max_lines = parse_line_range("notes/a.md", None, None)
    assert identifier == "notes/a.md"
    assert from_line is None
    assert max_lines is None


def test_parse_line_range_explicit_args_win_over_suffix():
    identifier, from_line, max_lines = parse_line_range("notes/a.md:100:40", 5, 10)
    assert identifier == "notes/a.md"
    assert from_line == 5
    assert max_lines == 10


def test_parse_line_range_clamps_from_line_to_at_least_one():
    identifier, from_line, max_lines = parse_line_range("notes/a.md:0", None, None)
    assert from_line == 1
