from pyqmd_mlx.mcp.server import _multi_get_impl
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_multi_get_impl_returns_one_resource_block_per_resolved_document():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello")
    _seed_doc(store, "notes", "b.md", "B", "world")

    result = _multi_get_impl(store, "notes/a.md, notes/b.md", None, 10 * 1024, True)

    assert result.is_error is not True
    assert len(result.content) == 2
    assert all(block.type == "resource" for block in result.content)


def test_multi_get_impl_reports_not_found_token_as_text_block():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello")

    result = _multi_get_impl(store, "notes/a.md, notes/missing.md", None, 10 * 1024, True)

    text_blocks = [b for b in result.content if b.type == "text"]
    assert any("missing.md" in b.text for b in text_blocks)


def test_multi_get_impl_reports_skipped_file_as_text_block():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "big.md", "Big", "x" * 100)

    result = _multi_get_impl(store, "notes/big.md", None, 10, True)

    text_blocks = [b for b in result.content if b.type == "text"]
    assert any("SKIPPED" in b.text for b in text_blocks)


def test_multi_get_impl_truncates_to_max_lines_with_marker():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "one\ntwo\nthree\nfour\nfive")

    result = _multi_get_impl(store, "notes/a.md", 2, 10 * 1024, False)

    text = result.content[0].resource.text
    assert "one" in text
    assert "two" in text
    assert "three" not in text
    assert "truncated 3 more lines" in text


def test_multi_get_impl_errors_when_nothing_matches():
    store = Store(":memory:")
    result = _multi_get_impl(store, "notes/missing.md", None, 10 * 1024, True)
    assert result.is_error is True


def test_multi_get_impl_skip_pointer_names_qmd_get():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "big.md", "Big", "x" * 100)

    result = _multi_get_impl(store, "notes/big.md", None, 10, True)

    text_blocks = [b for b in result.content if b.type == "text"]
    assert len(text_blocks) == 1
    assert (
        text_blocks[0].text == "[SKIPPED: notes/big.md - File too large (0KB > 0KB). "
        "Use 'qmd_get' with file=\"notes/big.md\" to retrieve.]"
    )
    assert "pyqmd get" not in text_blocks[0].text
    assert ".." not in text_blocks[0].text
    assert "qmd://" not in text_blocks[0].text


def test_multi_get_impl_zero_max_bytes_falls_back_to_default():
    """Node: maxBytes || DEFAULT -- an explicit 0 means the default, not
    "skip everything". A 100-byte doc must come back as a resource."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "x" * 100)

    result = _multi_get_impl(store, "notes/a.md", None, 0, True)

    assert result.is_error is not True
    assert result.content[0].type == "resource"


def test_multi_get_impl_default_max_bytes_is_64kb():
    import inspect

    from pyqmd_mlx.mcp.server import _multi_get_impl as impl

    assert inspect.signature(impl).parameters["max_bytes"].default == 65536

    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "medium.md", "M", "x" * (20 * 1024))

    result = _multi_get_impl(store, "notes/medium.md", None)

    assert result.is_error is not True
    assert all(b.type == "resource" for b in result.content)


def test_multi_get_impl_prepends_context_and_encodes_uri_per_entry():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")
    _seed_doc(store, "notes", "my note.md", "T", "hello")

    result = _multi_get_impl(store, "notes/my note.md", None, 65536, False)

    assert result.content[0].resource.uri == "qmd://notes/my%20note.md"
    assert result.content[0].resource.text == "<!-- Context: Root context -->\n\nhello"
