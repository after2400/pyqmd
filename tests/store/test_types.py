from pyqmd_mlx.store._types import (  # noqa: F401
    BreakPoint,
    CodeFenceRegion,
    ExpandedQueryPart,
    HybridQueryResult,
    RankedListMeta,
    RankedResult,
    SearchResult,
)


def test_break_point_fields():
    bp = BreakPoint(pos=10, score=100, type="h1")
    assert bp.pos == 10
    assert bp.score == 100
    assert bp.type == "h1"


def test_search_result_fields():
    result = SearchResult(
        filepath="qmd://mycol/a.md",
        display_path="mycol/a.md",
        title="A",
        hash="abc123",
        docid="abc123",
        collection_name="mycol",
        body="hello",
        score=0.5,
        source="fts",
        chunk_pos=None,
    )
    assert result.docid == "abc123"
    assert result.source == "fts"


def test_hybrid_query_result_defaults_empty_metadata():
    result = HybridQueryResult(
        file="qmd://mycol/a.md",
        display_path="mycol/a.md",
        title="A",
        body="hello",
        best_chunk="hello",
        best_chunk_pos=0,
        score=0.9,
        context=None,
        docid="abc123",
    )
    assert result.metadata == {}
