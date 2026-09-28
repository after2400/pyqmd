import pytest

from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] for _ in texts]


def _seed(store):
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    store.add_collection("c", "/c")
    for name, word in (("a", "alpha"), ("b", "bravo"), ("c", "charlie")):
        content = f"# {word}\n\nThis document is about {word} topics."
        h = store.hash_content(content)
        store.insert_content(h, content, "2026-01-01T00:00:00Z")
        store.insert_document(
            name, "doc.md", word, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.index_content(h, content)


def test_search_fts_single_collection_still_works():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    results = store.search_fts("alpha", collection="a")
    assert len(results) == 1
    assert results[0].collection_name == "a"


def test_search_fts_accepts_list_of_collections():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    # "document" (not "alpha OR bravo OR charlie") -- build_fts5_query has no
    # boolean-OR support; it ANDs every term (including a literal "or"), so
    # an OR-syntax query matches nothing regardless of collection filtering.
    # "document" is a word common to all three seeded docs and actually
    # exercises the collection IN-filter this test is checking.
    results = store.search_fts("document", collection=["a", "b"])
    names = {r.collection_name for r in results}
    assert names == {"a", "b"}
    assert "c" not in names


def test_search_fts_no_collection_filter_searches_everything():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    results = store.search_fts("document")
    names = {r.collection_name for r in results}
    assert names == {"a", "b", "c"}


def test_search_vec_accepts_list_of_collections():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    results = store.search_vec("anything", collection=["a", "c"])
    names = {r.collection_name for r in results}
    assert names <= {"a", "c"}
    assert "b" not in names


def test_search_vec_list_of_collections_below_exact_scan_threshold_uses_exact_scan():
    # Below FILTERED_VEC_EXACT_SCAN_MAX, search_vec exact-scans the eligible
    # set rather than falling back to ANN -- confirm this path still works
    # (not just the ANN fallback) when given a list of collections.
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    results = store.search_vec("anything", collection=["a", "b", "c"])
    assert len(results) == 3


@pytest.mark.requires_expansion_weights
def test_query_accepts_list_of_collections():
    def _fake_rerank(query, documents, model):
        return [0.9 for _ in documents]

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)
    # See test_search_fts_accepts_list_of_collections above for why
    # "document" is used instead of OR-syntax.
    results = store.query("document", collection=["a", "b"])
    files = {r.file for r in results}
    assert all("qmd://a/" in f or "qmd://b/" in f for f in files)
    assert not any("qmd://c/" in f for f in files)


def test_search_fts_no_collection_filter_skips_excluded_collection():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    store.set_collection_include_by_default("c", False)
    results = store.search_fts("document")
    names = {r.collection_name for r in results}
    assert names == {"a", "b"}


def test_search_fts_explicit_collection_overrides_exclusion():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    store.set_collection_include_by_default("c", False)
    results = store.search_fts("charlie", collection="c")
    assert len(results) == 1
    assert results[0].collection_name == "c"


def test_search_vec_no_collection_filter_skips_excluded_collection():
    store = Store(":memory:", embed_fn=_fake_embed)
    _seed(store)
    store.set_collection_include_by_default("b", False)
    results = store.search_vec("anything")
    names = {r.collection_name for r in results}
    assert "b" not in names


@pytest.mark.requires_expansion_weights
def test_query_no_collection_filter_skips_excluded_collection():
    def _fake_rerank(query, documents, model):
        return [0.9 for _ in documents]

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    _seed(store)
    store.set_collection_include_by_default("c", False)
    results = store.query("document")
    files = {r.file for r in results}
    assert not any("qmd://c/" in f for f in files)
