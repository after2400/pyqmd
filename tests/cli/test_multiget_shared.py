from pyqmd_mlx.cli._multiget import resolve_multi_get
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_resolve_multi_get_returns_entry_per_comma_separated_token():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")

    entries = resolve_multi_get(store, "notes/a.md, notes/b.md", max_bytes=10 * 1024)

    assert len(entries) == 2
    assert entries[0].display_path == "qmd://notes/a.md"
    assert entries[0].body == "hello world"
    assert entries[0].skipped is False
    assert entries[0].not_found is None
    assert entries[0].docid is not None
    assert entries[1].display_path == "qmd://notes/b.md"


def test_resolve_multi_get_marks_unresolved_token_as_not_found():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    entries = resolve_multi_get(store, "notes/a.md, notes/missing.md", max_bytes=10 * 1024)

    assert len(entries) == 2
    assert entries[1].not_found == "notes/missing.md"


def test_resolve_multi_get_skips_files_over_max_bytes_with_reason():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "big.md", "Big", "x" * 100)

    entries = resolve_multi_get(store, "notes/big.md", max_bytes=10)

    assert len(entries) == 1
    assert entries[0].skipped is True
    assert "too large" in entries[0].skip_reason.lower()
    assert "notes/big.md" in entries[0].skip_reason


def test_resolve_multi_get_empty_pattern_returns_no_entries():
    store = Store(":memory:")
    entries = resolve_multi_get(store, "  ,  ", max_bytes=10 * 1024)
    assert entries == []


def test_resolve_multi_get_matches_glob_pattern():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")
    _seed_doc(store, "notes", "c.txt", "C", "not markdown")

    entries = resolve_multi_get(store, "*.md", max_bytes=10 * 1024)

    assert [e.display_path for e in entries] == ["qmd://notes/a.md", "qmd://notes/b.md"]
    assert entries[0].body == "hello world"


def test_resolve_multi_get_glob_respects_path_segments():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "journals/2024-01.md", "J", "one")
    _seed_doc(store, "notes", "docs/readme.md", "R", "readme")

    entries = resolve_multi_get(store, "journals/*.md", max_bytes=10 * 1024)

    assert [e.display_path for e in entries] == ["qmd://notes/journals/2024-01.md"]


def test_resolve_multi_get_expands_braces():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "readme.md", "R", "readme")
    _seed_doc(store, "notes", "changelog.md", "C", "changelog")
    _seed_doc(store, "notes", "license.md", "L", "license")

    entries = resolve_multi_get(store, "{readme,changelog}.md", max_bytes=10 * 1024)

    assert {e.display_path for e in entries} == {
        "qmd://notes/readme.md",
        "qmd://notes/changelog.md",
    }


def test_resolve_multi_get_comma_with_glob_char_matches_nothing():
    """Ports Node's real dispatch quirk: a pattern with both a comma and
    a glob metacharacter is treated as one glob (not a comma-list),
    which in practice matches nothing since a bare comma isn't special
    outside braces. See the design spec's findings."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    entries = resolve_multi_get(store, "notes/a.md,17*.md", max_bytes=10 * 1024)

    assert entries == []


def test_resolve_multi_get_glob_with_no_matches_returns_empty_list():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    entries = resolve_multi_get(store, "*.txt", max_bytes=10 * 1024)

    assert entries == []
