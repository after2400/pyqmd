import os
from dataclasses import dataclass

from pyqmd_mlx.cli._fullpath import apply_full_path, format_fullpath_warning, render_full_path


def test_render_full_path_returns_dot_slash_for_cwd():
    assert render_full_path(os.path.realpath(os.getcwd())) == "./"


def test_render_full_path_returns_relative_for_cwd_subpath(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "sub" / "a.md"
    sub.parent.mkdir()
    sub.write_text("x")

    assert render_full_path(str(sub.resolve())) == "./sub/a.md"


def test_render_full_path_returns_absolute_outside_cwd(tmp_path, monkeypatch):
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    outside = tmp_path / "elsewhere" / "a.md"
    outside.parent.mkdir()
    outside.write_text("x")

    assert render_full_path(str(outside.resolve())) == str(outside.resolve())


@dataclass
class _FakeRow:
    display_path: str
    docid: str | None


class _FakeStore:
    def __init__(self, resolutions: dict[tuple[str, str], str | None]):
        self._resolutions = resolutions

    def resolve_full_path(self, collection, path):
        return self._resolutions.get((collection, path))


def test_apply_full_path_swaps_display_path_and_clears_docid():
    rows = [_FakeRow(display_path="qmd://notes/a.md", docid="abc123")]
    store = _FakeStore({("notes", "a.md"): "/real/notes/a.md"})

    unresolved = apply_full_path(rows, store)

    assert unresolved == 0
    assert rows[0].display_path == "/real/notes/a.md"
    assert rows[0].docid is None


def test_apply_full_path_leaves_unresolved_row_untouched_and_counts_it():
    rows = [_FakeRow(display_path="qmd://notes/gone.md", docid="def456")]
    store = _FakeStore({("notes", "gone.md"): None})

    unresolved = apply_full_path(rows, store)

    assert unresolved == 1
    assert rows[0].display_path == "qmd://notes/gone.md"
    assert rows[0].docid == "def456"


def test_format_fullpath_warning_singular_and_plural():
    assert "1 document " in format_fullpath_warning(1)
    assert "2 documents " in format_fullpath_warning(2)
