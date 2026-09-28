import pytest

from pyqmd_mlx.mcp._errors import error_result, tool_result_or_error


def test_error_result_produces_is_error_text_block():
    result = error_result("boom")
    assert result.is_error is True
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "boom"


def test_tool_result_or_error_catches_value_error():
    def _raises():
        raise ValueError("bad input")

    result = tool_result_or_error(_raises)
    assert result.is_error is True
    assert "bad input" in result.content[0].text


def test_tool_result_or_error_catches_sqlite_integrity_error():
    import sqlite3

    def _raises():
        raise sqlite3.IntegrityError("duplicate")

    result = tool_result_or_error(_raises)
    assert result.is_error is True
    assert "duplicate" in result.content[0].text


def test_tool_result_or_error_catches_sqlite_operational_error():
    import sqlite3

    def _raises():
        raise sqlite3.OperationalError("database is locked")

    result = tool_result_or_error(_raises)
    assert result.is_error is True
    assert "database is locked" in result.content[0].text


def test_tool_result_or_error_returns_fn_result_on_success():
    from mcp.types import CallToolResult, TextContent

    expected = CallToolResult(content=[TextContent(type="text", text="fine")])

    def _ok():
        return expected

    assert tool_result_or_error(_ok) is expected


def test_tool_result_or_error_propagates_unexpected_exception():
    def _crashes():
        raise RuntimeError("real bug")

    with pytest.raises(RuntimeError):
        tool_result_or_error(_crashes)


def test_tool_result_or_error_catches_expansion_model_error():
    from pyqmd_mlx.llm import ExpansionModelError

    def _raises():
        raise ExpansionModelError("Could not load query-expansion model 'x/y': boom")

    result = tool_result_or_error(_raises)
    assert result.is_error is True
    assert "Could not load query-expansion model 'x/y'" in result.content[0].text
