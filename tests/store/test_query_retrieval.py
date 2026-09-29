from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _fake_expand(query, model):
    return [
        f"lex: {query} setup",
        f"vec: how to {query}",
        f"hyde: {query} is configured via steps.",
    ]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")


def test_retrieve_and_fuse_finds_document_via_expansion():
    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")
    _seed_doc(store, "notes", "unrelated.md", "Unrelated", "cooking recipes")

    fused = store._retrieve_and_fuse("auth", collection=None, candidate_limit=40)
    files = [r.file for r in fused]
    assert any("auth.md" in f for f in files)
    store.close()


def test_retrieve_and_fuse_strong_bm25_signal_skips_expansion():
    calls = []

    def counting_expand(query, model):
        calls.append(query)
        return _fake_expand(query, model)

    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=counting_expand)
    store.add_collection("notes", "/notes")
    # A very distinctive term with only one match gives a strong, unambiguous
    # BM25 hit -- expansion should be skipped entirely. BM25's magnitude
    # scales with corpus size via IDF, so this needs a large-enough sea of
    # unrelated filler documents (none containing the term) plus the term
    # appearing in both title and body of the one matching doc for the
    # normalized score to clear STRONG_SIGNAL_MIN_SCORE (0.85) in practice
    # (measured ~0.88 with 200 filler docs; a couple with just one filler
    # doc measured ~9e-07 -- nowhere close).
    _seed_doc(
        store,
        "notes",
        "a.md",
        "zzqqxxunique_distinctive_term_here",
        "zzqqxxunique_distinctive_term_here present",
    )
    for i in range(200):
        _seed_doc(store, "notes", f"b{i}.md", f"B{i}", "completely different unrelated content")

    store._retrieve_and_fuse(
        "zzqqxxunique_distinctive_term_here", collection=None, candidate_limit=40
    )
    assert calls == []  # expansion was never called
    store.close()


def test_retrieve_and_fuse_respects_candidate_limit():
    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand)
    store.add_collection("notes", "/notes")
    for i in range(10):
        _seed_doc(store, "notes", f"auth{i}.md", f"Auth {i}", "authentication guide content shared")

    fused = store._retrieve_and_fuse("auth", collection=None, candidate_limit=3)
    assert len(fused) <= 3
    store.close()


def test_retrieve_and_fuse_falls_back_to_hyde_when_expansion_is_prose(monkeypatch):
    store = Store(
        ":memory:",
        embed_fn=_fake_embed,
        expand_fn=lambda query, model: ["Here is a cooking guide.", "Boil the water."],
    )
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    vec_queries, fts_queries = [], []
    real_vec, real_fts = store.search_vec, store.search_fts
    monkeypatch.setattr(
        store, "search_vec", lambda q, **kw: vec_queries.append(q) or real_vec(q, **kw)
    )
    monkeypatch.setattr(
        store, "search_fts", lambda q, **kw: fts_queries.append(q) or real_fts(q, **kw)
    )

    # "zebra" matches nothing, so the BM25 probe can't take the strong-signal
    # shortcut and expansion runs.
    store._retrieve_and_fuse("zebra", collection=None, candidate_limit=40)

    assert vec_queries == ["zebra", "Information about zebra"]
    assert fts_queries == ["zebra", "zebra"]  # BM25 probe + original-query list
    store.close()
