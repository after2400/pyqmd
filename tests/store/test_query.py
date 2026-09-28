from pyqmd_mlx.store import Store
from pyqmd_mlx.store._metadata_filter import parse_metadata_filter


def _fake_embed(texts, model, kind="query"):
    return [[1.0, 0.0] if "auth" in t.lower() else [0.0, 1.0] for t in texts]


def _fake_expand(query, model):
    return [f"lex: {query} setup", f"vec: how to {query}"]


def _fake_rerank(query, documents, model):
    # Score by whether the query terms literally appear -- deterministic
    # and good enough to test ordering/finalization without a real model.
    return [1.0 if query.lower() in doc.lower() else 0.1 for doc in documents]


def _seed_doc(store, collection, path, title, body):
    content_hash = store.hash_content(body)
    store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(content_hash, body, model="fake-model")


def _make_store():
    return Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand, rerank_fn=_fake_rerank)


def test_query_returns_hybrid_query_results():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")
    _seed_doc(store, "notes", "cooking.md", "Cooking", "pasta recipe instructions")

    results = store.query("authentication")
    assert len(results) >= 1
    assert results[0].title == "Auth"
    assert results[0].docid == store.find_active_document("notes", "auth.md")["hash"][:6]
    store.close()


def test_query_populates_context_when_configured():
    store = _make_store()
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Notes root context")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide content")

    results = store.query("authentication")

    assert results[0].context == "Notes root context"
    store.close()


def test_query_best_chunk_is_substring_of_body():
    store = _make_store()
    store.add_collection("notes", "/notes")
    body = "# Intro\nunrelated filler text.\n# Auth\nauthentication configuration guide."
    _seed_doc(store, "notes", "auth.md", "Auth", body)

    results = store.query("authentication")
    assert results[0].best_chunk in results[0].body
    store.close()


def test_query_respects_min_score():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")
    _seed_doc(store, "notes", "other.md", "Other", "something else entirely unrelated")

    results = store.query("authentication", min_score=0.99)
    assert all(r.score >= 0.99 for r in results)
    store.close()


def test_query_respects_limit():
    store = _make_store()
    store.add_collection("notes", "/notes")
    for i in range(5):
        _seed_doc(store, "notes", f"auth{i}.md", f"Auth {i}", "authentication guide content shared")

    results = store.query("authentication", limit=2)
    assert len(results) <= 2
    store.close()


def test_query_dedups_by_file():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    results = store.query("authentication")
    files = [r.file for r in results]
    assert len(files) == len(set(files))
    store.close()


def test_query_skip_rerank_uses_rrf_score_only():
    calls = []

    def counting_rerank(query, documents, model):
        calls.append(documents)
        return _fake_rerank(query, documents, model)

    store = Store(
        ":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand, rerank_fn=counting_rerank
    )
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    store.query("authentication", skip_rerank=True)
    assert calls == []
    store.close()


def test_query_skip_rerank_scores_top_result_as_one():
    # skip_rerank scores must land in the same ~0..1 scale min_score is
    # calibrated against for the reranked branch (matches store.ts's
    # skipRerank branch: score = 1 / rank, not the raw internal RRF fusion
    # score, whose range is much smaller, e.g. ~0.03-0.17). The raw RRF
    # score was never meant to be user-facing.
    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    results = store.query("authentication", skip_rerank=True)
    assert results[0].score == 1.0
    store.close()


def test_query_skip_rerank_min_score_uses_rank_based_scale_not_raw_rrf():
    # A min_score threshold picked for the reranked branch's ~0..1 scale
    # (e.g. 0.3) must not silently zero out every result just because
    # skip_rerank was also passed -- this was the actual bug.
    store = Store(":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand)
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")
    _seed_doc(store, "notes", "auth2.md", "Auth2", "authentication login flow details")

    results = store.query("authentication", skip_rerank=True, min_score=0.3)
    assert len(results) >= 1
    store.close()


def test_query_metadata_defaults_to_empty_dict():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    results = store.query("authentication")
    assert results[0].metadata == {}
    store.close()


def test_query_intent_disables_strong_signal_bypass():
    calls = []

    def counting_expand(query, model):
        calls.append(query)
        return _fake_expand(query, model)

    store = Store(
        ":memory:", embed_fn=_fake_embed, expand_fn=counting_expand, rerank_fn=_fake_rerank
    )
    store.add_collection("notes", "/notes")
    for i in range(200):
        _seed_doc(store, "notes", f"filler{i}.md", f"Filler {i}", "completely unrelated content")
    _seed_doc(
        store,
        "notes",
        "a.md",
        "zzqqxxunique_distinctive_term_here",
        "zzqqxxunique_distinctive_term_here present",
    )

    store.query("zzqqxxunique_distinctive_term_here", intent="something")
    assert calls != []  # expansion WAS called -- strong-signal bypass was disabled
    store.close()


def test_query_intent_biases_chunk_selection():
    # This document must be long enough that chunk_document() (3600-char
    # default chunk size) splits it into multiple distinct chunks -- a
    # single-chunk body would make best_chunk always equal the whole body
    # regardless of scoring, and the test would pass even with
    # INTENT_WEIGHT_CHUNK's contribution deleted.
    store = _make_store()
    store.add_collection("notes", "/notes")
    filler_sentence = (
        "This is unrelated filler content about gardening and weather patterns across the region. "
    )
    # "setup" appears only in this first section, well before the first
    # chunk boundary (~3600 chars in), so the bare query term "setup" alone
    # favors this chunk.
    section_a = "## Filler\nRemember to complete the setup before continuing.\n" + (
        filler_sentence * 45
    )
    # This section contains no occurrence of "setup" at all, but repeats
    # several intent terms -- only INTENT_WEIGHT_CHUNK-weighted scoring can
    # tip chunk selection here.
    section_b = "## Auth\n" + (
        "Authentication and credentials management guide for the deployment pipeline. " * 45
    )
    body = section_a + "\n\n" + section_b
    _seed_doc(store, "notes", "auth.md", "Auth", body)

    # Companion assertion: without intent, the bare query term "setup" only
    # occurs in the filler section, so plain query-term overlap selects that
    # chunk -- proving the auth/credentials chunk is NOT the query's own
    # preferred pick.
    baseline = store.query("setup")
    assert "setup" in baseline[0].best_chunk.lower()
    assert "authentication" not in baseline[0].best_chunk.lower()

    # With intent, INTENT_WEIGHT_CHUNK-weighted intent-term hits (authentication,
    # credentials, deployment, pipeline -- all absent from the filler chunk)
    # outweigh the lone "setup" match and tip selection to the chunk that
    # doesn't contain "setup" at all.
    biased = store.query("setup", intent="authentication credentials deployment pipeline")
    assert "authentication" in biased[0].best_chunk.lower()
    assert "credentials" in biased[0].best_chunk.lower()
    assert "setup" not in biased[0].best_chunk.lower()
    assert biased[0].best_chunk_pos != baseline[0].best_chunk_pos
    store.close()


def test_query_intent_is_prepended_to_rerank_input():
    seen_queries = []

    def capturing_rerank(query, documents, model):
        seen_queries.append(query)
        return _fake_rerank(query, documents, model)

    store = Store(
        ":memory:", embed_fn=_fake_embed, expand_fn=_fake_expand, rerank_fn=capturing_rerank
    )
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    store.query("authentication", intent="deployment steps")
    assert seen_queries[0] == "deployment steps\n\nauthentication"
    store.close()


def test_query_without_intent_still_behaves_as_before():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    results = store.query("authentication")
    assert len(results) >= 1
    store.close()


def _seed_doc_with_metadata(store, collection, path, title, body, metadata_yaml):
    full_body = f"---\nqmd:\n  metadata:\n{metadata_yaml}\n---\n{body}"
    content_hash = store.hash_content(full_body)
    store.insert_content(content_hash, full_body, "2026-01-01T00:00:00Z")
    doc_id = store.insert_document(
        collection, path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.sync_document_metadata(doc_id, full_body, path)
    store.index_content(content_hash, full_body, model="fake-model")
    return doc_id


def test_query_filter_narrows_hybrid_results():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc_with_metadata(
        store,
        "notes",
        "a.md",
        "Auth Published",
        "authentication configuration guide",
        "    status: published",
    )
    _seed_doc_with_metadata(
        store,
        "notes",
        "b.md",
        "Auth Draft",
        "authentication configuration guide",
        "    status: draft",
    )

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    results = store.query("authentication", filter=filter_)

    assert [r.title for r in results] == ["Auth Published"]
    store.close()


def test_query_populates_real_metadata_on_results():
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc_with_metadata(
        store,
        "notes",
        "a.md",
        "Auth",
        "authentication configuration guide",
        "    status: published\n    priority: 3",
    )

    results = store.query("authentication")

    assert results[0].metadata == {"status": "published", "priority": 3}
    store.close()


def test_query_probe_passes_filter_to_the_strong_signal_probe_call():
    # The BM25 probe must apply `filter` too -- store.ts:5579-5582: "the
    # strong-signal decision must be based only on eligible documents."
    #
    # An earlier version of this test tried to prove this end-to-end by
    # engineering a corpus where an unfiltered probe would see a false
    # strong signal from a filter-excluded document. That approach turned
    # out to be unreliable: this codebase's BM25 scoring (bm25() with a
    # tiny/synthetic corpus) makes STRONG_SIGNAL_MIN_SCORE=0.85 very hard to
    # cross deterministically -- IDF collapses toward zero once a term's
    # document frequency is a large fraction of a small corpus, regardless
    # of term frequency within the matching document. A live experiment
    # confirmed a corpus built exactly this way never crossed the threshold,
    # making the test pass identically whether or not the probe actually
    # received `filter` -- i.e. it didn't test what it claimed to.
    #
    # This version instead asserts the mechanism directly: it spies on
    # `store.search_fts` and confirms the very first call `query()` makes
    # (always the `limit=2` probe -- see `_retrieve_and_fuse`'s first
    # statement) receives the same `filter` object passed to `query()`.
    # This is deterministic and cannot be defeated by corpus/scoring
    # quirks, and it still fails immediately if a regression drops
    # `filter=filter` from that specific call site.
    store = _make_store()
    store.add_collection("notes", "/notes")
    _seed_doc(store, "notes", "auth.md", "Auth", "authentication configuration guide")

    calls = []
    real_search_fts = store.search_fts

    def spying_search_fts(*args, **kwargs):
        calls.append(kwargs)
        return real_search_fts(*args, **kwargs)

    store.search_fts = spying_search_fts

    filter_ = parse_metadata_filter({"key": "status", "operator": "eq", "value": "published"})
    store.query("authentication", filter=filter_)

    assert len(calls) >= 1
    probe_call = calls[0]  # the probe is always the first search_fts call
    assert probe_call.get("limit") == 2
    assert probe_call.get("filter") is filter_
    store.close()


def test_query_auto_chunk_strategy_picks_best_chunk_at_function_boundary():
    store = _make_store()
    store.add_collection("code", "/code")
    # Long enough to force a real split at the default chunk size; the
    # needle function sits mid-file so overlap/padding can't accidentally
    # align a chunk start with it.
    body = "\n".join(
        [f"def noop_{i}():\n    pass\n" for i in range(80)]
        + ["def target_function():\n    return 'needle value'\n"]
        + [f"def noop_{i}():\n    pass\n" for i in range(80, 160)]
    )
    assert len(body) > 3600  # sanity: chunk_document really splits this
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "code", "sample.py", "Sample", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(h, body, filepath="sample.py", chunk_strategy="auto")

    results = store.query("needle value", chunk_strategy="auto")

    assert results
    assert "needle value" in results[0].best_chunk
    # The whole needle function survives in one chunk -- no split between
    # its def line and its body.
    assert "def target_function():\n    return 'needle value'" in results[0].best_chunk
