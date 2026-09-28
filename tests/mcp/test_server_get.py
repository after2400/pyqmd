from pyqmd_mlx.mcp.server import _get_impl
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_get_impl_returns_resource_content_block_on_success():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "line one\nline two\nline three")

    result = _get_impl(store, "notes/a.md", None, None, True)

    assert result.is_error is not True
    assert result.content[0].type == "resource"
    assert result.content[0].resource.uri == "qmd://notes/a.md"
    assert "1: line one" in result.content[0].resource.text


def test_get_impl_returns_error_when_not_found():
    store = Store(":memory:")
    result = _get_impl(store, "notes/missing.md", None, None, True)
    assert result.is_error is True
    assert "not found" in result.content[0].text.lower()


def test_get_impl_uri_leaves_encodeuricomponent_safe_chars_raw():
    """Node's encodeQmdPath is encodeURIComponent per segment: !*'() stay
    raw while spaces encode. A doc whose path uses both pins the set."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "we'ird (1)!.md", "W", "body")

    result = _get_impl(store, "notes/we'ird (1)!.md", None, None, True)

    assert result.is_error is not True
    assert result.content[0].resource.uri == "qmd://notes/we'ird%20(1)!.md"


def test_get_impl_supports_line_range_suffix_on_the_file_argument():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "one\ntwo\nthree\nfour\nfive")

    result = _get_impl(store, "notes/a.md:2:2", None, None, True)

    text = result.content[0].resource.text
    assert "two" in text
    assert "three" in text
    assert "one" not in text
    assert "four" not in text


def test_get_impl_without_line_numbers_omits_them():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    result = _get_impl(store, "notes/a.md", None, None, False)

    assert result.content[0].resource.text == "hello world"


def test_get_impl_prepends_context_header_when_collection_has_context():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes about people")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    result = _get_impl(store, "notes/a.md", None, None, False)

    assert result.content[0].resource.text == "<!-- Context: Notes about people -->\n\nhello world"


def test_get_impl_context_header_stays_unline_numbered():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes about people")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    result = _get_impl(store, "notes/a.md", None, None, True)

    text = result.content[0].resource.text
    assert text.startswith("<!-- Context: Notes about people -->\n\n1: hello world")


def test_get_impl_segment_encodes_special_characters_in_uri():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "my note.md", "Titled", "body")

    result = _get_impl(store, "notes/my note.md", None, None, False)

    assert result.content[0].resource.uri == "qmd://notes/my%20note.md"
