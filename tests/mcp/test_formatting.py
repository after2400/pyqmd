from pyqmd_mlx.mcp._formatting import format_query_result, format_query_summary
from pyqmd_mlx.store._types import HybridQueryResult


def _make_result(**overrides) -> HybridQueryResult:
    defaults = dict(
        file="qmd://notes/a.md",
        display_path="notes/a.md",
        title="A",
        body="# A\n\nThis is about authentication and login flows.",
        best_chunk="This is about authentication and login flows.",
        best_chunk_pos=0,
        score=0.87,
        context=None,
        docid="abc123",
    )
    defaults.update(overrides)
    return HybridQueryResult(**defaults)


def test_format_query_result_has_hash_prefixed_docid():
    result = format_query_result(_make_result(), "authentication", intent=None)
    assert result["docid"] == "#abc123"


def test_format_query_result_rounds_score_to_two_decimals():
    result = format_query_result(_make_result(score=0.8734), "authentication", intent=None)
    assert result["score"] == 0.87


def test_format_query_result_includes_display_path_and_title():
    # Deliberately the bare display_path, not the qmd://-prefixed file --
    # Node's real MCP `query` tool emits the bare form (verified against a
    # live Node MCP server); see pyqmd_mlx/mcp/_formatting.py's comment.
    result = format_query_result(_make_result(), "authentication", intent=None)
    assert result["file"] == "notes/a.md"
    assert result["title"] == "A"


def test_format_query_result_includes_line_and_snippet():
    result = format_query_result(_make_result(), "authentication", intent=None)
    assert isinstance(result["line"], int)
    assert "authentication" in result["snippet"].lower()


def test_format_query_result_omits_context_when_none():
    result = format_query_result(_make_result(context=None), "authentication", intent=None)
    assert result["context"] is None


def test_format_query_result_includes_context_when_present():
    result = format_query_result(
        _make_result(context="meeting notes"), "authentication", intent=None
    )
    assert result["context"] == "meeting notes"


def test_format_query_result_includes_non_empty_metadata():
    result = _make_result(metadata={"status": "published"})
    formatted = format_query_result(result, "alpha", None)
    assert formatted["metadata"] == {"status": "published"}


def test_format_query_result_omits_empty_metadata():
    result = _make_result(metadata={})
    formatted = format_query_result(result, "alpha", None)
    assert "metadata" not in formatted


def test_format_query_summary_reports_zero_results():
    text = format_query_summary([], "authentication")
    assert "No results" in text
    assert "authentication" in text


def test_format_query_summary_lists_each_result():
    results = [
        format_query_result(_make_result(docid="aaa111", title="A"), "authentication", intent=None),
        format_query_result(
            _make_result(
                docid="bbb222", title="B", display_path="notes/b.md", file="qmd://notes/b.md"
            ),
            "authentication",
            intent=None,
        ),
    ]
    text = format_query_summary(results, "authentication")
    assert "#aaa111" in text
    assert "#bbb222" in text
    assert "notes/a.md" in text
    assert "notes/b.md" in text
