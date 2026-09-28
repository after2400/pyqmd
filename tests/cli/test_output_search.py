import csv
import io
import json

import pytest

from pyqmd_mlx.cli._output_search import format_search_results
from pyqmd_mlx.cli._types import DisplayResult


def _row(**overrides) -> DisplayResult:
    defaults = dict(
        docid="abc123",
        score=0.8734,
        display_path="notes/auth.md",
        title="Auth",
        body="# Auth\nHow to configure authentication for your app.",
        chunk_pos=None,
        context=None,
    )
    defaults.update(overrides)
    return DisplayResult(**defaults)


def test_json_format_includes_docid_score_file_title():
    output = format_search_results([_row()], "json", query="authentication")
    parsed = json.loads(output)
    assert parsed[0]["docid"] == "#abc123"
    assert parsed[0]["score"] == 0.87
    assert parsed[0]["file"] == "notes/auth.md"
    assert parsed[0]["title"] == "Auth"
    assert "snippet" in parsed[0]
    assert "body" not in parsed[0]


def test_json_format_full_includes_body_not_snippet():
    output = format_search_results([_row()], "json", query="authentication", full=True)
    parsed = json.loads(output)
    assert "body" in parsed[0]
    assert "snippet" not in parsed[0]


def test_json_format_includes_context_when_present():
    output = format_search_results([_row(context="Auth docs")], "json", query="auth")
    parsed = json.loads(output)
    assert parsed[0]["context"] == "Auth docs"


def test_csv_format_has_header_and_one_row_per_result():
    output = format_search_results([_row(), _row(docid="def456")], "csv", query="auth")
    # Parse with csv.reader rather than a naive split on "\n": the snippet
    # column can legitimately contain embedded newlines (RFC 4180 quoting),
    # as it does here since both body lines match "auth" and the snippet
    # window ends up spanning the whole 2-line body.
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == ["docid", "score", "file", "title", "context", "line", "snippet"]
    assert len(rows) == 3


def test_files_format_is_docid_score_path():
    output = format_search_results([_row()], "files")
    assert output == "#abc123,0.87,notes/auth.md"


def test_markdown_format_includes_heading_and_docid():
    output = format_search_results([_row()], "md", query="auth")
    assert "# Auth" in output
    assert "**docid:** `#abc123`" in output


def test_xml_format_is_well_formed_with_file_element():
    output = format_search_results([_row()], "xml", query="auth")
    assert '<file docid="#abc123"' in output
    assert "</file>" in output


def test_cli_format_includes_path_title_score_and_content():
    output = format_search_results([_row()], "cli", query="auth")
    assert "notes/auth.md" in output
    assert "Auth" in output
    assert "87%" in output


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError):
        format_search_results([_row()], "yaml")


def test_empty_results_produces_empty_or_minimal_output():
    assert format_search_results([], "files") == ""
    assert json.loads(format_search_results([], "json")) == []


def test_json_format_includes_non_empty_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "json")
    assert '"metadata"' in output
    assert '"status": "published"' in output


def test_json_format_omits_empty_metadata():
    row = _row(metadata={})
    output = format_search_results([row], "json")
    assert '"metadata"' not in output


def test_cli_format_includes_non_empty_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "cli")
    assert "status" in output and "published" in output


def test_cli_format_omits_empty_metadata_line():
    row = _row(metadata={})
    output = format_search_results([row], "cli")
    assert "Metadata:" not in output


def test_markdown_format_includes_non_empty_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "md")
    assert "status" in output and "published" in output


def test_xml_format_includes_non_empty_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "xml")
    assert "<metadata>" in output
    assert "status" in output


def test_csv_format_never_includes_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "csv")
    assert "status" not in output


def test_files_format_never_includes_metadata():
    row = _row(metadata={"status": "published"})
    output = format_search_results([row], "files")
    assert "status" not in output


def test_json_format_omits_docid_when_none():
    output = format_search_results([_row(docid=None)], "json", query="authentication")
    parsed = json.loads(output)
    assert "docid" not in parsed[0]


def test_csv_format_leaves_docid_column_blank_when_none():
    output = format_search_results([_row(docid=None)], "csv", query="authentication")
    reader = csv.reader(io.StringIO(output))
    rows = list(reader)
    assert rows[1][0] == ""


def test_files_format_omits_docid_when_none():
    output = format_search_results([_row(docid=None)], "files")
    assert not output.startswith("#")


def test_markdown_format_omits_docid_line_when_none():
    output = format_search_results([_row(docid=None)], "md", query="authentication")
    assert "**docid:**" not in output


def test_xml_format_omits_docid_attribute_when_none():
    output = format_search_results([_row(docid=None)], "xml", query="authentication")
    assert "docid=" not in output


def test_cli_format_omits_docid_suffix_when_none():
    output = format_search_results([_row(docid=None)], "cli", query="authentication")
    assert "#" not in output.split("\n")[0]
