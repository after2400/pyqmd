from pyqmd_mlx.store import Store


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    return content_hash


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


# --- find_document_by_identifier ---


def test_find_document_by_identifier_resolves_collection_slash_path():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    doc = store.find_document_by_identifier("notes/a.md")
    assert doc["title"] == "A"
    assert doc["doc"] == "hello world"
    store.close()


def test_find_document_by_identifier_strips_qmd_scheme():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    doc = store.find_document_by_identifier("qmd://notes/a.md")
    assert doc["title"] == "A"
    store.close()


def test_find_document_by_identifier_resolves_docid_with_hash_prefix():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = _seed_doc(store, "notes", "a.md", "A", "hello world")

    doc = store.find_document_by_identifier(content_hash[:6])
    assert doc["title"] == "A"
    store.close()


def test_find_document_by_identifier_strips_hash_prefix_marker():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = _seed_doc(store, "notes", "a.md", "A", "hello world")

    doc = store.find_document_by_identifier(f"#{content_hash[:6]}")
    assert doc["title"] == "A"
    store.close()


def test_find_document_by_identifier_returns_none_when_not_found():
    store = Store(":memory:")
    assert store.find_document_by_identifier("notes/missing.md") is None
    assert store.find_document_by_identifier("zzzzzz") is None
    store.close()


def test_find_document_by_identifier_resolves_bare_filename_with_no_collection_prefix():
    """A bare filename with no '/' and no matching docid hash should still
    resolve by falling back to a path lookup across any collection --
    mirrors Node's findDocument() (src/store.ts), which does the same via
    its own per-collection relative-path fallback. This is what makes
    `pyqmd get some-file.md` work without requiring the caller to know
    (or type) which collection it lives in."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    doc = store.find_document_by_identifier("a.md")
    assert doc["title"] == "A"
    assert doc["doc"] == "hello world"
    store.close()


def test_find_document_by_identifier_prefers_docid_match_over_bare_filename_fallback():
    """If the bare string happens to be BOTH a valid docid-hash-prefix
    match AND coincidentally equal to some document's path, the existing
    docid interpretation wins -- the path fallback only kicks in once the
    docid lookup has already failed to find anything."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    content_hash = _seed_doc(store, "notes", "a.md", "Docid Match", "hash content")
    docid_prefix = content_hash[:6]
    # A second document whose *path* happens to equal the first doc's docid
    # prefix -- a contrived but possible collision to prove precedence.
    _seed_doc(store, "notes", docid_prefix, "Path Match", "path content")

    doc = store.find_document_by_identifier(docid_prefix)
    assert doc["title"] == "Docid Match"
    store.close()


def test_find_document_by_identifier_bare_filename_fallback_returns_none_when_no_collection_has_it():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    assert store.find_document_by_identifier("nope.md") is None
    store.close()


def test_find_document_by_identifier_treats_percent_and_underscore_literally():
    """'%' and '_' are SQL LIKE wildcards. A bare '%' or '_' identifier must
    not match arbitrary documents by accident -- it should behave as a
    literal (non-matching) search string when no hash actually starts with
    that literal character."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")

    assert store.find_document_by_identifier("%") is None
    assert store.find_document_by_identifier("_") is None
    store.close()


# --- find_documents_by_glob ---


def test_find_documents_by_glob_matches_pattern_in_one_collection():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "alpha")
    _seed_doc(store, "notes", "b.md", "B", "bravo")
    _seed_doc(store, "notes", "c.txt", "C", "charlie")

    docs = store.find_documents_by_glob("*.md")

    assert [d["path"] for d in docs] == ["a.md", "b.md"]
    store.close()


def test_find_documents_by_glob_respects_path_segments():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "journals/2024-01.md", "J1", "one")
    _seed_doc(store, "notes", "journals/2024-02.md", "J2", "two")
    _seed_doc(store, "notes", "docs/readme.md", "R", "readme")

    top_level = store.find_documents_by_glob("*.md")
    assert top_level == []

    nested = store.find_documents_by_glob("journals/*.md")
    assert [d["path"] for d in nested] == ["journals/2024-01.md", "journals/2024-02.md"]
    store.close()


def test_find_documents_by_glob_matches_collection_slash_path_pattern():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "alpha")
    _seed_doc(store, "notes", "b.md", "B", "bravo")

    docs = store.find_documents_by_glob("notes/*.md")

    assert [d["path"] for d in docs] == ["a.md", "b.md"]
    store.close()


def test_find_documents_by_glob_expands_braces():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "readme.md", "R", "readme")
    _seed_doc(store, "notes", "changelog.md", "C", "changelog")
    _seed_doc(store, "notes", "license.md", "L", "license")

    docs = store.find_documents_by_glob("{readme,changelog}.md")

    assert [d["path"] for d in docs] == ["changelog.md", "readme.md"]
    store.close()


def test_find_documents_by_glob_sorted_by_collection_then_path():
    store = Store(":memory:")
    store.add_collection("zeta", "/zeta")
    store.add_collection("alpha", "/alpha")
    _seed_doc(store, "zeta", "a.md", "ZA", "za")
    _seed_doc(store, "alpha", "b.md", "AB", "ab")

    docs = store.find_documents_by_glob("*.md")

    assert [(d["collection"], d["path"]) for d in docs] == [("alpha", "b.md"), ("zeta", "a.md")]
    store.close()


def test_find_documents_by_glob_returns_empty_list_when_nothing_matches():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "alpha")

    docs = store.find_documents_by_glob("*.txt")

    assert docs == []
    store.close()


# --- clear_embeddings ---


def test_clear_embeddings_makes_hash_eligible_for_reembedding():
    calls = []

    def counting_embed(texts, model, kind="query", title=None):
        calls.append(texts)
        return _fake_embed(texts, model, kind, title)

    store = Store(":memory:", embed_fn=counting_embed)
    store.add_collection("notes", "/notes")
    content_hash = _seed_doc(store, "notes", "a.md", "A", "hello world")
    store.index_content(content_hash, "hello world", model="fake-model")
    assert len(calls) == 1

    cleared = store.clear_embeddings("notes")
    assert cleared > 0

    store.index_content(content_hash, "hello world", model="fake-model")
    assert len(calls) == 2  # re-embedded, not skipped as already-current
    store.close()


def test_clear_embeddings_scoped_to_collection_leaves_others_untouched():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    hash_a = _seed_doc(store, "a", "x.md", "X", "content a")
    hash_b = _seed_doc(store, "b", "y.md", "Y", "content b")
    store.index_content(hash_a, "content a", model="fake-model")
    store.index_content(hash_b, "content b", model="fake-model")

    store.clear_embeddings("a")

    remaining = store.conn.execute(
        "SELECT COUNT(*) as n FROM content_vectors WHERE hash = ?", (hash_b,)
    ).fetchone()["n"]
    assert remaining == 1
    store.close()


def test_clear_embeddings_returns_zero_when_nothing_embedded():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    assert store.clear_embeddings("notes") == 0
    store.close()


# --- get_indexable_content ---


def test_get_indexable_content_returns_distinct_hash_doc_pairs():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")

    content = store.get_indexable_content("notes")
    assert {row["doc"] for row in content} == {"hello world", "goodbye world"}
    store.close()


def test_get_indexable_content_scoped_by_collection():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "X", "content a")
    _seed_doc(store, "b", "y.md", "Y", "content b")

    content = store.get_indexable_content("a")
    assert [row["doc"] for row in content] == ["content a"]
    store.close()


def test_get_indexable_content_unscoped_returns_all():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "X", "content a")
    _seed_doc(store, "b", "y.md", "Y", "content b")

    content = store.get_indexable_content()
    assert {row["doc"] for row in content} == {"content a", "content b"}
    store.close()


# --- get_status_counts ---


def test_get_status_counts_reports_documents_vectors_and_pending():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    hash_a = _seed_doc(store, "notes", "a.md", "A", "hello world")
    _seed_doc(store, "notes", "b.md", "B", "goodbye world")
    store.index_content(hash_a, "hello world", model="fake-model")

    counts = store.get_status_counts(model="fake-model")
    assert counts["active_documents"] == 2
    assert counts["embedded_vectors"] == 1
    assert counts["pending_embed"] == 1
    assert counts["most_recent_modified_at"] == "2026-01-01T00:00:00Z"
    store.close()


# --- remove_collection (now cascade-deletes) ---


def test_remove_collection_deletes_its_documents():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    assert store.remove_collection("notes") is True
    assert store.get_collection("notes") is None
    assert store.find_active_document("notes", "a.md") is None
    row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents WHERE collection = ?", ("notes",)
    ).fetchone()
    assert row["n"] == 0
    store.close()


def test_remove_collection_cleans_up_orphaned_content_and_fts():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "a.md", "A", "hello world")

    store.remove_collection("notes")

    content_row = store.conn.execute("SELECT COUNT(*) as n FROM content").fetchone()
    assert content_row["n"] == 0
    fts_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM documents_fts WHERE documents_fts MATCH 'hello'"
    ).fetchone()
    assert fts_row["n"] == 0
    store.close()


def test_remove_collection_preserves_content_shared_with_another_collection():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    shared_body = "shared content"
    content_hash = store.hash_content(shared_body)
    store.insert_content(content_hash, shared_body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "a", "x.md", "X", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.insert_document(
        "b", "y.md", "Y", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )

    store.remove_collection("a")

    assert store.find_active_document("b", "y.md") is not None
    content_row = store.conn.execute(
        "SELECT COUNT(*) as n FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert content_row["n"] == 1
    store.close()


def test_remove_collection_returns_false_when_missing():
    store = Store(":memory:")
    assert store.remove_collection("nope") is False
    store.close()
