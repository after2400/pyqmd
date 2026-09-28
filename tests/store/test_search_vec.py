from pyqmd_mlx.store import Store
from pyqmd_mlx.store._metadata_filter import parse_metadata_filter


def _fake_embed_similarity(texts, model, kind="query"):
    # Encode "closeness to a target concept" as a 2D vector so cosine
    # distance produces a predictable ranking in tests: texts containing
    # "auth" point toward (1, 0), everything else toward (0, 1).
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")
    return content_hash


def test_search_vec_returns_empty_when_no_vectors_indexed():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    results = store.search_vec("authentication", model="fake-model")
    assert results == []
    store.close()


def test_search_vec_finds_semantically_close_document():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication setup guide")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "recipe for pasta dinner")

    results = store.search_vec("auth credentials", model="fake-model")
    assert len(results) >= 1
    assert results[0].title == "Auth"
    assert results[0].source == "vec"
    store.close()


def test_search_vec_populates_context_when_configured():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication setup guide")

    results = store.search_vec("auth credentials", model="fake-model")

    assert results[0].context == "Notes root context"
    store.close()


def test_search_vec_respects_limit():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    for i in range(5):
        _seed_doc(store, "notes", f"auth{i}.md", f"Auth {i}", "authentication guide content")

    results = store.search_vec("auth", limit=2, model="fake-model")
    assert len(results) <= 2
    store.close()


def test_search_vec_scopes_to_collection():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    _seed_doc(store, "a", "x.md", "X", "authentication in collection a")
    _seed_doc(store, "b", "y.md", "Y", "authentication in collection b")

    results = store.search_vec("auth", collection="a", model="fake-model")
    assert all(r.collection_name == "a" for r in results)
    store.close()


def test_search_vec_dedupes_multi_chunk_document_by_filepath():
    # Regression test for finding 1: a multi-chunk document must not occupy
    # multiple slots in the result list (store.ts:4370-4382 dedupes by
    # filepath, keeping the chunk with the best/lowest distance).
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")

    # Long enough (> CHUNK_SIZE_CHARS = 3600) to be split into several
    # chunks by chunk_document; every chunk contains "auth" so every chunk
    # embeds close to the query under _fake_embed_similarity.
    long_body = "This document explains authentication flows in detail. " * 100
    assert len(long_body) > 3600
    _seed_doc(store, "notes", "auth-long.md", "Auth Long", long_body)
    _seed_doc(store, "notes", "other.md", "Other", "totally unrelated cooking content")

    results = store.search_vec("auth credentials", limit=10, model="fake-model")

    filepaths = [r.filepath for r in results]
    assert len(filepaths) == len(set(filepaths)), (
        f"expected each filepath at most once, got {filepaths}"
    )
    assert sum(1 for r in results if r.title == "Auth Long") == 1
    store.close()


def _fake_embed_starvation(texts, model, kind="query"):
    # Direction-only 2D vectors (cosine distance ignores magnitude): the
    # "small" collection's one relevant document embeds a bit further from
    # the query direction than every "big" collection document, so a global
    # ANN scan with a small k picks up only "big" chunks and never reaches
    # "small"'s document.
    vecs = []
    for t in texts:
        if "SMALL_DOC_MARKER" in t:
            vecs.append([0.9, 0.44])
        else:
            vecs.append([1.0, 0.0001])
    return vecs


def test_search_vec_collection_scope_does_not_starve_small_collection():
    # Regression test for finding 2: a global ANN scan (k = limit*3)
    # followed by a post-filter can silently exclude a small scoped
    # collection's real candidates before the filter ever runs
    # (store.ts:4196-4203, issues #791, #803). Exact-scanning the collection's
    # own eligible vectors when the set is small avoids this.
    store = Store(":memory:", embed_fn=_fake_embed_starvation)
    store.add_collection("small", "/small")
    store.add_collection("big", "/big")

    _seed_doc(
        store,
        "small",
        "relevant.md",
        "Relevant",
        "SMALL_DOC_MARKER this is the one relevant document in the small collection",
    )
    for i in range(80):
        _seed_doc(
            store,
            "big",
            f"noise{i}.md",
            f"Noise {i}",
            f"unrelated filler content number {i} with no relevant keywords",
        )

    limit = 5
    results = store.search_vec(
        "authentication credentials please", collection="small", limit=limit, model="fake-model"
    )

    assert len(results) == 1
    assert results[0].title == "Relevant"
    assert results[0].collection_name == "small"
    store.close()


def _seed_with_metadata_and_embedding(store, collection, path, title, body, metadata_yaml):
    full_body = f"---\nqmd:\n  metadata:\n{metadata_yaml}\n---\n{body}"
    content_hash = store.hash_content(full_body)
    store.insert_content(content_hash, full_body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.sync_document_metadata(doc_id, full_body, path)
    store.index_content(content_hash, full_body, model="fake-model")
    return doc_id


def test_search_vec_filter_excludes_non_matching_documents():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    _seed_with_metadata_and_embedding(
        store, "notes", "a.md", "Auth Published", "auth content here", "    status: published"
    )
    _seed_with_metadata_and_embedding(
        store, "notes", "b.md", "Auth Draft", "auth content here", "    status: draft"
    )

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_vec("auth", model="fake-model", filter=filter_)

    assert [r.title for r in results] == ["Auth Published"]
    store.close()


def test_search_vec_filter_with_no_eligible_documents_returns_empty():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    _seed_with_metadata_and_embedding(
        store, "notes", "a.md", "Auth Draft", "auth content here", "    status: draft"
    )

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_vec("auth", model="fake-model", filter=filter_)

    assert results == []
    store.close()


def test_search_vec_filter_excludes_documents_without_extraction():
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    body = "auth content here"
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "a.md", "Auth", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")
    # No sync_document_metadata call -- this document was never extracted.

    filter_ = parse_metadata_filter({"key": "status", "operator": "exists", "value": False})
    results = store.search_vec("auth", model="fake-model", filter=filter_)

    assert results == []
    store.close()


def test_search_vec_filter_reapplied_at_doc_lookup_for_shared_content_hash():
    # Two documents that share one content hash (identical body) but differ
    # in extraction state: one has a document_metadata row matching the
    # filter, the other has none at all (never extracted). The eligible-set
    # query alone can't distinguish them -- it operates on distinct
    # hash_seq values derived from the shared hash, and only needs ONE of
    # the two documents to pass its JOIN in order to admit that hash_seq --
    # so the doc-lookup query's own independent filter/collection guard is
    # what excludes the unextracted document from the final results. This
    # proves that second guard (store.py's search_vec doc-lookup query) is
    # load-bearing, not redundant.
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")

    shared_body = "---\nqmd:\n  metadata:\n    status: published\n---\nauth content here"
    content_hash = store.hash_content(shared_body)
    store.insert_content(content_hash, shared_body, "2026-01-01T00:00:00Z")

    doc_extracted = store.insert_document(
        "notes",
        "a.md",
        "A extracted",
        content_hash,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    store.sync_document_metadata(doc_extracted, shared_body, "a.md")

    # This second document shares the exact same content hash as
    # doc_extracted but intentionally never gets sync_document_metadata
    # called -- it has no document_metadata row at all.
    store.insert_document(
        "notes",
        "b.md",
        "B unextracted",
        content_hash,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )

    store.index_content(content_hash, shared_body, model="fake-model")

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_vec("auth", model="fake-model", filter=filter_)

    assert [r.title for r in results] == ["A extracted"]
    store.close()


def test_search_vec_filter_combines_with_collection_scope():
    # Both documents have status:published metadata -- only the collection
    # scope should exclude "Other Auth", proving `collection` and `filter`
    # compose through the same eligible-set query rather than either one
    # accidentally doing all the work.
    store = Store(":memory:", embed_fn=_fake_embed_similarity)
    store.add_collection("notes", "/notes")
    store.add_collection("other", "/other")
    _seed_with_metadata_and_embedding(
        store, "notes", "a.md", "Notes Auth", "auth content here", "    status: published"
    )
    _seed_with_metadata_and_embedding(
        store, "other", "b.md", "Other Auth", "auth content here too", "    status: published"
    )

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.search_vec("auth", model="fake-model", collection="notes", filter=filter_)

    assert [r.title for r in results] == ["Notes Auth"]
    store.close()
