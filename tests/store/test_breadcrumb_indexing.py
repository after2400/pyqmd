from pyqmd_mlx.store import Store
from pyqmd_mlx.store._fingerprint import embedding_fingerprint

MODEL = "fake-model"
LONG = (
    "# Guide\n\n## A\n\n"
    + "alpha words here. " * 220
    + "\n\n## B\n\n"
    + "beta words here. " * 220
    + "\n"
)
SHORT = "# Guide\n\nintro\n\n## Setup\n\ninstall it\n"


def _store(calls, **kwargs):
    def recording_embed(texts, model, kind="query", title=None):
        calls.append({"texts": list(texts), "kind": kind, "title": title})
        return [[1.0, 0.0] for _ in texts]

    return Store(":memory:", embed_fn=recording_embed, embed_model=MODEL, **kwargs)


def _seed(store, path, body):
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document("notes", path, "t", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    return h


def test_markdown_chunks_get_per_chunk_titles_with_the_section_path():
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    titles = calls[0]["title"]
    assert isinstance(titles, list) and len(titles) == len(calls[0]["texts"]) >= 2
    # The first chunk sits in "## A", the last in "## B".
    assert titles[0] == "Guide > A" and titles[-1] == "Guide > B"
    store.close()


NO_H1 = (
    "## Overview\n\n"
    + "alpha words here. " * 220
    + "\n\n## Details\n\n"
    + "beta words here. " * 220
    + "\n"
)


def test_a_note_that_opens_at_h2_does_not_repeat_its_title_in_the_path():
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    h = _seed(store, "note.md", NO_H1)
    store.index_content(h, NO_H1, filepath="note.md")
    titles = calls[0]["title"]
    # extract_title is "Overview": the first chunk sits in that section (plain
    # title), the last in "## Details".
    assert titles[0] == "Overview" and titles[-1] == "Overview > Details"
    store.close()


def test_a_note_whose_only_section_is_its_title_gets_the_plain_title_string():
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    body = "## Overview\n\nsome prose under the only heading\n"
    h = _seed(store, "one.md", body)
    store.index_content(h, body, filepath="one.md")
    assert calls[0]["title"] == "Overview"
    store.close()


def test_a_document_with_no_headings_gets_the_plain_title_string():
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    body = "just prose, no headings at all\n"
    h = _seed(store, "plain.md", body)
    store.index_content(h, body, filepath="plain.md")
    assert calls[0]["title"] == "plain"
    store.close()


def test_non_markdown_files_get_the_plain_title_string():
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    body = "# Guide\n\n## Setup\n\nx = 1\n"
    h = _seed(store, "helpers.py", body)
    store.index_content(h, body, filepath="helpers.py")
    assert calls[0]["title"] == "helpers"
    store.close()


def test_breadcrumbs_off_calls_the_embedder_exactly_as_before():
    calls = []
    store = _store(calls, breadcrumbs=False)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    assert calls[0]["title"] == "Guide"
    store.close()


def test_the_keyword_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("PYQMD_EMBED_BREADCRUMBS", "0")
    calls = []
    store = _store(calls, breadcrumbs=True)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    assert isinstance(calls[0]["title"], list)
    store.close()

    monkeypatch.setenv("PYQMD_EMBED_BREADCRUMBS", "1")
    calls = []
    store = _store(calls, breadcrumbs=False)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    assert calls[0]["title"] == "Guide"
    store.close()


def test_the_environment_turns_breadcrumbs_on(monkeypatch):
    monkeypatch.setenv("PYQMD_EMBED_BREADCRUMBS", "1")
    calls = []
    store = _store(calls)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    assert isinstance(calls[0]["title"], list)
    store.close()


def test_default_is_off_so_nothing_changes_for_existing_indexes(monkeypatch):
    monkeypatch.delenv("PYQMD_EMBED_BREADCRUMBS", raising=False)
    calls = []
    store = _store(calls)
    store.add_collection("notes", "/notes")
    h = _seed(store, "guide.md", LONG)
    store.index_content(h, LONG, filepath="guide.md")
    assert calls[0]["title"] == "Guide"
    stored = store.conn.execute("SELECT DISTINCT embed_fingerprint FROM content_vectors").fetchall()
    assert [r[0] for r in stored] == [embedding_fingerprint(MODEL)]
    store.close()


def test_rows_store_the_fingerprint_that_matches_the_mode():
    for flag in (False, True):
        calls = []
        store = _store(calls, breadcrumbs=flag)
        store.add_collection("notes", "/notes")
        h = _seed(store, "guide.md", SHORT)
        store.index_content(h, SHORT, filepath="guide.md")
        stored = store.conn.execute(
            "SELECT DISTINCT embed_fingerprint FROM content_vectors"
        ).fetchall()
        assert [r[0] for r in stored] == [embedding_fingerprint(MODEL, flag)]
        assert store.count_pending_embed() == 0
        store.close()


def test_switching_breadcrumbs_makes_documents_pending_and_embed_clears_them(tmp_path):
    db = str(tmp_path / "idx.sqlite")
    calls = []

    def recording_embed(texts, model, kind="query", title=None):
        calls.append(title)
        return [[1.0, 0.0] for _ in texts]

    off = Store(db, embed_fn=recording_embed, embed_model=MODEL, breadcrumbs=False)
    off.add_collection("notes", "/notes")
    h = _seed(off, "guide.md", SHORT)
    off.index_content(h, SHORT, filepath="guide.md")
    off.close()

    on = Store(db, embed_fn=recording_embed, embed_model=MODEL, breadcrumbs=True)
    assert on.count_pending_embed() == 1  # stale: embedded under the old fingerprint
    on.index_content(h, SHORT, filepath="guide.md")
    assert on.count_pending_embed() == 0
    n = len(calls)
    on.index_content(h, SHORT, filepath="guide.md")
    assert len(calls) == n  # already current: no second embed
    on.close()

    off_again = Store(db, embed_fn=recording_embed, embed_model=MODEL, breadcrumbs=False)
    assert off_again.count_pending_embed() == 1  # toggling back also re-embeds
    off_again.close()
