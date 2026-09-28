from pyqmd_mlx.mcp._instructions import build_instructions
from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )


def test_instructions_mention_document_count():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")

    text = build_instructions(store)

    assert "2" in text
    assert "notes" in text


def test_instructions_on_empty_store_do_not_crash_and_mention_zero_documents():
    store = Store(":memory:")
    text = build_instructions(store)
    assert "0" in text


def test_instructions_note_missing_embeddings_when_no_vectors_exist():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    text = build_instructions(store)

    assert "embed" in text.lower()


def test_instructions_mention_all_four_tools():
    store = Store(":memory:")
    text = build_instructions(store)
    for tool_name in ("query", "get", "multi_get", "status"):
        assert tool_name in text
