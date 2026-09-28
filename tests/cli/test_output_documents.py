import json

import pytest

from pyqmd_mlx.cli._output_documents import format_documents
from pyqmd_mlx.cli._types import DocumentEntry


def _entry(**overrides) -> DocumentEntry:
    defaults = dict(
        display_path="qmd://notes/a.md",
        title="A",
        body="hello world",
        context=None,
        skipped=False,
        skip_reason=None,
        docid="abc123",
    )
    defaults.update(overrides)
    return DocumentEntry(**defaults)


def test_json_format_includes_file_docid_title_body():
    output = format_documents([_entry()], "json")
    parsed = json.loads(output)
    assert parsed[0]["file"] == "qmd://notes/a.md"
    assert parsed[0]["docid"] == "#abc123"
    assert parsed[0]["title"] == "A"
    assert parsed[0]["body"] == "hello world"


def test_json_format_omits_docid_when_none():
    output = format_documents([_entry(docid=None)], "json")
    parsed = json.loads(output)
    assert "docid" not in parsed[0]


def test_json_format_skipped_entry_has_no_body():
    output = format_documents([_entry(skipped=True, skip_reason="too large")], "json")
    parsed = json.loads(output)
    assert parsed[0]["skipped"] is True
    assert parsed[0]["reason"] == "too large"
    assert "body" not in parsed[0]


def test_csv_format_has_header_and_row_per_entry():
    output = format_documents([_entry(), _entry(display_path="qmd://notes/b.md")], "csv")
    lines = output.split("\n")
    assert lines[0] == "docid,file,title,context,skipped,body"
    assert len(lines) == 3
    assert lines[1].startswith("#abc123,qmd://notes/a.md,")


def test_files_format_marks_skipped_entries():
    output = format_documents([_entry(skipped=True, skip_reason="x")], "files")
    assert "[SKIPPED]" in output
    assert output.startswith("#abc123,qmd://notes/a.md")


def test_markdown_format_includes_heading_docid_and_body_fence():
    output = format_documents([_entry()], "md")
    assert "## qmd://notes/a.md" in output
    assert "**docid:** `#abc123`" in output
    assert "```\nhello world\n```" in output


def test_xml_format_is_well_formed():
    output = format_documents([_entry()], "xml")
    assert '<document docid="#abc123">' in output
    assert "<file>qmd://notes/a.md</file>" in output


def test_cli_format_uses_dedicated_banner_renderer_not_markdown():
    # Node's `qmd multi-get` CLI format is NOT the markdown formatter --
    # it's a separate `=`-bordered banner renderer (multiGet() in
    # src/cli/qmd.ts). Porting `cli` as a documents_to_markdown fallback
    # was a real bug: it produced a different line count
    # than Node's actual output for the same documents.
    output = format_documents([_entry()], "cli")
    assert output != format_documents([_entry()], "md")
    bar = "=" * 60
    assert output == f"\n{bar}\nFile: qmd://notes/a.md  #abc123\n{bar}\n\nhello world"


def test_cli_format_two_documents_matches_node_byte_for_byte():
    # Verified directly against a real frozen Node `qmd multi-get`
    # invocation on the same two real scifact documents: identical output,
    # 18 lines on both sides (was 22 for pyqmd before this fix).
    bar = "=" * 60
    entries = [
        _entry(display_path="qmd://scifact/a.md", docid="975c9b", body="line1\nline2\nline3"),
        _entry(display_path="qmd://scifact/b.md", docid="ea185a", body="line1\nline2\nline3"),
    ]
    output = format_documents(entries, "cli")
    expected = (
        f"\n{bar}\nFile: qmd://scifact/a.md  #975c9b\n{bar}\n\n"
        "line1\nline2\nline3\n"
        f"\n{bar}\nFile: qmd://scifact/b.md  #ea185a\n{bar}\n\n"
        "line1\nline2\nline3"
    )
    assert output == expected
    assert len(output.splitlines()) == 16


def test_cli_format_skipped_entry_shows_skip_reason_no_body():
    output = format_documents([_entry(skipped=True, skip_reason="too large")], "cli")
    assert "[SKIPPED: too large]" in output
    assert "hello world" not in output


def test_cli_format_includes_folder_context_when_present():
    output = format_documents([_entry(context="some folder note")], "cli")
    assert "Folder Context: some folder note" in output


def test_cli_format_omits_docid_suffix_when_none():
    output = format_documents([_entry(docid=None)], "cli")
    assert "File: qmd://notes/a.md\n" in output
    assert "#" not in output.split("File:")[1].split("\n")[0]


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError):
        format_documents([_entry()], "yaml")
