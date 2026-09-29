import pytest

from pyqmd_mlx.store._title import extract_title


# Expected values are Node's own extractTitle output at 8262698, run in bun.
@pytest.mark.parametrize(
    ("content", "filename", "expected"),
    [
        ("# Hello\nbody", "a.md", "Hello"),
        ("intro\n## Sub\n# Top\n", "a.md", "Sub"),
        ("### Deep\ntext", "a.md", "a"),
        ("# Notes\n\n## Weekly\n", "n.md", "Weekly"),
        ("# \U0001f4dd Notes\n## Garden\n", "n.md", "Garden"),
        ("# Notes\ntext", "n.md", "Notes"),
        ("#\n\nfoo bar", "h.md", "foo bar"),
        ("# Loud", "x.MD", "Loud"),
        ("#+title: Reading\n* Heading", "r.org", "Reading"),
        ("* Plan\n** Tasks", "p.org", "Plan"),
        ("#+TITLE:   \n* Heading", "x.org", "* Heading"),
        ("#+TITLE:   ", "x.org", "x"),
        ("#+TITLE:\n\n", "y.org", "y"),
        ("# setup helpers\n", "src/helpers.py", "helpers"),
        ("# T", "a.markdown", "a"),
        ("text", "README", "README"),
        ("text", ".env", ".env"),
        ("text", "archive.v2/README", "archive"),
        ("text", "notes/2024.01.md", "2024.01"),
        ("# Title\r\nbody", "crlf.md", "Title"),
        ("# a b\n", "ls.md", "a"),
    ],
)
def test_extract_title_matches_nodes_rules(content, filename, expected):
    assert extract_title(content, filename) == expected
