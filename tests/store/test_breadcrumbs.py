import pytest

from pyqmd_mlx.store._breadcrumbs import (
    BREADCRUMBS_ENV_VAR,
    MAX_BREADCRUMB_CHARS,
    Heading,
    breadcrumb_title,
    breadcrumbs_enabled,
    dominant_heading_path,
    find_headings,
    heading_path,
)

DOC = (
    "# Page\n\nintro\n\n## A\n\ntext a\n\n### A1\n\ntext a1\n\n"
    "### A2\n\ntext a2\n\n## B\n\ntext b\n"
)


def test_find_headings_returns_level_text_and_position():
    headings = find_headings(DOC)
    assert [(h.level, h.text) for h in headings] == [
        (1, "Page"),
        (2, "A"),
        (3, "A1"),
        (3, "A2"),
        (2, "B"),
    ]
    assert headings[1].pos == DOC.index("## A")


def test_find_headings_ignores_hash_lines_inside_code_fences():
    content = "# T\n\n## Real\n\n```\n# not a heading\n```\n\ntail"
    assert [h.text for h in find_headings(content)] == ["T", "Real"]


def test_find_headings_strips_closing_hashes_but_keeps_a_trailing_hash_in_the_text():
    content = "## Rates ##\n\n# C#\n"
    assert [h.text for h in find_headings(content)] == ["Rates", "C#"]


def test_heading_path_nests_and_pops_siblings():
    headings = find_headings(DOC)
    assert heading_path(headings, DOC.index("intro")) == []
    assert heading_path(headings, DOC.index("text a1")) == ["A", "A1"]
    assert heading_path(headings, DOC.index("text a2")) == ["A", "A2"]
    assert heading_path(headings, DOC.index("text b")) == ["B"]


def test_heading_path_includes_a_heading_that_starts_at_pos_and_drops_its_siblings():
    headings = find_headings(DOC)
    # The section that starts at "## B" is B, not the A2 section before it.
    assert heading_path(headings, DOC.index("## B")) == ["B"]
    assert heading_path(headings, DOC.index("## B") - 1) == ["A", "A2"]


SECTIONED = "# P\n\n## A\n\n" + "a" * 50 + "\n\n## B\n\n" + "b" * 500 + "\n"


def test_dominant_heading_path_picks_the_section_holding_most_of_the_chunk():
    headings = find_headings(SECTIONED)
    # Starts in A's last ten characters, then runs through all of B: B wins.
    pos = SECTIONED.index("a" * 50) + 40
    assert dominant_heading_path(headings, pos, len(SECTIONED) - pos) == ["B"]
    # Starts at the top of A's text and stops inside B's first lines: A wins.
    start = SECTIONED.index("a" * 50)
    assert dominant_heading_path(headings, start, 50 + 10) == ["A"]


def test_dominant_heading_path_without_a_heading_inside_the_chunk_is_the_path_at_pos():
    headings = find_headings(DOC)
    pos = DOC.index("text a1")
    assert dominant_heading_path(headings, pos, 5) == ["A", "A1"]


def test_dominant_heading_path_breaks_a_tie_toward_the_earlier_section():
    content = "# P\n\n## A\n\n" + "a" * 20 + "\n\n## B\n\n" + "b" * 20 + "\n"
    headings = find_headings(content)
    boundary = content.index("## B")
    start = boundary - 10
    assert dominant_heading_path(headings, start, 20) == ["A"]


def test_dominant_heading_path_for_a_page_without_headings_is_empty():
    assert dominant_heading_path(find_headings("plain text\n"), 0, 10) == []


def test_heading_path_after_a_code_fence_keeps_the_real_section():
    content = "# T\n\n## Real\n\n```\n# not a heading\n```\n\ntail"
    assert heading_path(find_headings(content), content.index("tail")) == ["Real"]


def test_heading_path_without_headings_is_empty():
    assert heading_path(find_headings("just some text\n"), 5) == []


def test_heading_path_never_contains_level_one_headings():
    content = "# One\n\n## Two\n\n# Three\n\ntext"
    assert heading_path(find_headings(content), content.index("text")) == []


def test_heading_is_a_plain_value():
    assert Heading(3, 2, "x") == Heading(pos=3, level=2, text="x")


def test_breadcrumb_title_joins_the_title_and_path():
    assert breadcrumb_title("Page", ["A", "A1"]) == "Page > A > A1"


def test_breadcrumb_title_without_a_path_is_the_title_unchanged():
    assert breadcrumb_title("Page", []) == "Page"


def test_breadcrumb_title_at_exactly_the_cap_is_kept():
    heading = "h" * (MAX_BREADCRUMB_CHARS - len(" > "))
    title = breadcrumb_title("Page", [heading])
    assert title == f"Page > {heading}" and len(title) - len("Page") == MAX_BREADCRUMB_CHARS


def test_breadcrumb_title_drops_outer_headings_first_when_over_the_cap():
    inner = "i" * 200
    title = breadcrumb_title("Page", ["o" * 100, inner])
    assert title == f"Page > {inner}"


def test_breadcrumb_title_truncates_a_single_enormous_heading_with_an_ellipsis():
    title = breadcrumb_title("Page", ["x" * 5000])
    added = title[len("Page") :]
    assert title.startswith("Page > x") and title.endswith("…")
    assert len(added) == MAX_BREADCRUMB_CHARS


def test_breadcrumb_title_drops_a_leading_heading_equal_to_the_doc_title():
    # extract_title takes the first "#" or "##" heading, so a note that opens
    # with "## Overview" is titled "Overview" and must not read "Overview > Overview".
    assert breadcrumb_title("Overview", ["Overview", "Sub"]) == "Overview > Sub"
    assert breadcrumb_title("Overview", ["Overview"]) == "Overview"
    # Only the first entry: a later heading with the same text is real.
    assert breadcrumb_title("A", ["B", "A"]) == "A > B > A"


def test_find_headings_strips_closing_hashes_in_crlf_files():
    assert [h.text for h in find_headings("## Rates ##\r\n\r\ntext\r\n")] == ["Rates"]


def test_find_headings_skips_headings_with_no_text():
    content = "# T\n\n##  \n\ntext\n\n### Sub\n\nx"
    assert [h.text for h in find_headings(content)] == ["T", "Sub"]


def test_breadcrumb_title_keeps_the_doc_title_whatever_its_length():
    assert breadcrumb_title("T" * 400, ["A"]) == "T" * 400 + " > A"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "On"])
def test_breadcrumbs_enabled_by_env(value):
    assert breadcrumbs_enabled({BREADCRUMBS_ENV_VAR: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "off", "", "anything"])
def test_breadcrumbs_not_enabled_by_other_env_values(value):
    assert breadcrumbs_enabled({BREADCRUMBS_ENV_VAR: value}) is False


def test_breadcrumbs_are_off_when_unset():
    assert breadcrumbs_enabled({}) is False
