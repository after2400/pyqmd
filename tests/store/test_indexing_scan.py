from pathlib import Path

from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import ReindexResult, scan_and_register_collection, split_glob_mask


def test_split_glob_mask_splits_on_top_level_comma():
    assert split_glob_mask("a.md,*.txt") == ["a.md", "*.txt"]


def test_split_glob_mask_leaves_braces_intact():
    assert split_glob_mask("**/*.{md,txt}") == ["**/*.{md,txt}"]


def test_split_glob_mask_single_pattern_no_comma():
    assert split_glob_mask("**/*.md") == ["**/*.md"]


def test_scan_and_register_collection_indexes_new_files(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nworld content")
    (tmp_path / "b.md").write_text("# Second\nmore content")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert isinstance(result, ReindexResult)
    assert result.indexed == 2
    assert result.updated == 0
    assert result.unchanged == 0
    assert result.removed == 0

    doc = store.find_active_document("notes", "a.md")
    assert doc["title"] == "Hello"
    store.close()


def test_scan_and_register_collection_extracts_title_from_h1_or_h2():
    pass  # covered by test_scan_and_register_collection_indexes_new_files above ("Hello")


def test_scan_and_register_collection_falls_back_to_filename_stem(tmp_path):
    (tmp_path / "no-heading.md").write_text("just plain text, no heading")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    doc = store.find_active_document("notes", "no-heading.md")
    assert doc["title"] == "no-heading"
    store.close()


def test_scan_and_register_collection_notes_heading_skips_to_next_h2(tmp_path):
    (tmp_path / "journal.md").write_text("# Notes\n## Real Title\nbody text")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    doc = store.find_active_document("notes", "journal.md")
    assert doc["title"] == "Real Title"
    store.close()


def test_scan_and_register_collection_updates_changed_content(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Hello\noriginal")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    f.write_text("# Hello\nchanged")
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.updated == 1
    assert result.indexed == 0
    doc = store.find_active_document("notes", "a.md")
    assert (
        store.conn.execute("SELECT doc FROM content WHERE hash = ?", (doc["hash"],)).fetchone()[
            "doc"
        ]
        == "# Hello\nchanged"
    )
    store.close()


def test_scan_and_register_collection_unchanged_content_is_a_noop(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nsame")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.unchanged == 1
    assert result.updated == 0
    assert result.indexed == 0
    store.close()


def test_scan_and_register_collection_deactivates_removed_files(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Hello\nworld")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    f.unlink()
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.removed == 1
    assert store.find_active_document("notes", "a.md") is None
    store.close()


def test_scan_and_register_collection_skips_hidden_files_and_dirs(tmp_path):
    (tmp_path / ".hidden.md").write_text("# Hidden\nshould not be indexed")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config.md").write_text("# Config\nshould not be indexed")
    (tmp_path / "visible.md").write_text("# Visible\nshould be indexed")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.indexed == 1
    assert store.find_active_document("notes", "visible.md") is not None
    store.close()


def test_scan_and_register_collection_skips_excluded_dirs_at_root_and_nested(tmp_path):
    # Root-level excluded directory (the common case: a JS project checkout
    # with node_modules/ directly under the collection root).
    root_excluded = tmp_path / "node_modules"
    root_excluded.mkdir()
    (root_excluded / "pkg.md").write_text("# Pkg\nshould not be indexed")

    # Nested excluded directories, one level down.
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.md").write_text("# Lib\nshould not be indexed")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "dist").mkdir()
    (tmp_path / "sub" / "dist" / "f.md").write_text("# F\nshould not be indexed")
    (tmp_path / "sub" / "build").mkdir()
    (tmp_path / "sub" / "build" / "f.md").write_text("# F\nshould not be indexed")

    (tmp_path / "visible.md").write_text("# Visible\nshould be indexed")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.indexed == 1
    assert store.find_active_document("notes", "visible.md") is not None
    assert store.find_active_document("notes", "node_modules/pkg.md") is None
    assert store.find_active_document("notes", "vendor/lib.md") is None
    assert store.find_active_document("notes", "sub/dist/f.md") is None
    assert store.find_active_document("notes", "sub/build/f.md") is None
    store.close()


def test_scan_and_register_collection_respects_comma_separated_mask(tmp_path):
    (tmp_path / "a.md").write_text("# A\ncontent")
    (tmp_path / "b.txt").write_text("plain text content")
    (tmp_path / "c.py").write_text("# not indexed")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "*.md,*.txt", "notes")

    assert result.indexed == 2
    store.close()


def test_scan_and_register_collection_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("   \n  ")
    (tmp_path / "real.md").write_text("# Real\ncontent")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.indexed == 1
    store.close()


def test_scan_and_register_collection_extracts_metadata_for_new_files(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\nqmd:\n  metadata:\n    status: published\n---\n# Hello\nbody"
    )

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    doc = store.find_active_document("notes", "a.md")
    row = store.conn.execute(
        "SELECT text_value FROM document_metadata_values WHERE document_id = ? AND key = 'status'",
        (doc["id"],),
    ).fetchone()
    assert row["text_value"] == "published"
    store.close()


def test_scan_and_register_collection_re_extracts_metadata_for_changed_files(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("---\nqmd:\n  metadata:\n    status: draft\n---\n# Hello\nbody")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    f.write_text("---\nqmd:\n  metadata:\n    status: published\n---\n# Hello\nchanged body")
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    doc = store.find_active_document("notes", "a.md")
    row = store.conn.execute(
        "SELECT text_value FROM document_metadata_values WHERE document_id = ? AND key = 'status'",
        (doc["id"],),
    ).fetchone()
    assert row["text_value"] == "published"
    store.close()


def test_scan_and_register_collection_backfills_metadata_for_unchanged_files(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("---\nqmd:\n  metadata:\n    status: published\n---\n# Hello\nbody")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    doc = store.find_active_document("notes", "a.md")
    # Simulate a pre-this-feature database: metadata was never extracted for
    # this unchanged document.
    store.conn.execute("DELETE FROM document_metadata WHERE document_id = ?", (doc["id"],))
    store.conn.execute("DELETE FROM document_metadata_values WHERE document_id = ?", (doc["id"],))
    store.conn.commit()

    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.unchanged == 1
    row = store.conn.execute(
        "SELECT text_value FROM document_metadata_values WHERE document_id = ? AND key = 'status'",
        (doc["id"],),
    ).fetchone()
    assert row["text_value"] == "published"
    store.close()


def test_scan_and_register_collection_reports_orphaned_cleaned(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\noriginal content")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    (tmp_path / "a.md").write_text("# Hello\nchanged content")
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.updated == 1
    assert result.orphaned_cleaned == 1
    content_row = store.conn.execute("SELECT COUNT(*) as n FROM content").fetchone()
    assert content_row["n"] == 1  # only the new hash remains
    store.close()


def test_scan_and_register_collection_orphaned_cleaned_zero_when_unchanged(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nsame content")

    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.unchanged == 1
    assert result.orphaned_cleaned == 0
    store.close()


def test_scan_falls_back_to_index_time_when_birthtime_missing(tmp_path, monkeypatch):
    # st_birthtime exists on macOS/BSD but not on Linux -- scan must fall
    # back to the indexing time instead of raising AttributeError (caught
    # via ubuntu CI, where every collection add/update failed).
    real_stat = Path.stat

    class _NoBirthtime:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "st_birthtime":
                raise AttributeError("st_birthtime")
            return getattr(self._inner, name)

    def _stat_no_birthtime(self, *, follow_symlinks=True):
        # NOTE: must accept (and forward) follow_symlinks -- pathlib
        # internals (exists()/is_symlink()/lstat()) pass it, including
        # pytest's own traceback rendering if this test fails. A bare
        # (self) signature turns any such call into a TypeError that
        # crashes the whole pytest session (INTERNALERROR).
        return _NoBirthtime(real_stat(self, follow_symlinks=follow_symlinks))

    monkeypatch.setattr(Path, "stat", _stat_no_birthtime)

    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert result.indexed == 1
    doc = store.find_active_document("notes", "a.md")
    assert doc is not None
    assert doc["created_at"]  # fell back to index time, not a crash
    store.close()


def test_rescan_refreshes_a_title_that_changed_without_a_content_change(tmp_path):
    (tmp_path / "helpers.py").write_text("# setup helpers\n\ndef setup():\n    pass\n")
    store = Store(":memory:")
    store.add_collection("code", str(tmp_path), pattern="**/*.py")
    scan_and_register_collection(store, str(tmp_path), "**/*.py", "code")
    doc = store.find_active_document("code", "helpers.py")
    # An index written before the title fix stored a different title.
    store.update_document(doc["id"], "Stale Zebra Title", doc["hash"], doc["modified_at"])

    result = scan_and_register_collection(store, str(tmp_path), "**/*.py", "code")

    assert (result.indexed, result.updated, result.unchanged) == (0, 1, 0)
    refreshed = store.find_active_document("code", "helpers.py")
    assert refreshed["title"] == "helpers"
    assert refreshed["hash"] == doc["hash"]
    assert refreshed["modified_at"] != doc["modified_at"]
    assert store.search_fts("zebra") == []
    assert [r.title for r in store.search_fts("helpers")] == ["helpers"]
    store.close()


def test_rescan_leaves_a_current_title_unchanged(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nworld")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    result = scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert (result.updated, result.unchanged) == (0, 1)
    store.close()
