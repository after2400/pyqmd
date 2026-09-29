from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    # deterministic, distinguishable fake vectors: encode text length so
    # tests can assert on which vectors got inserted.
    return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def test_ensure_vec_table_creates_table_with_requested_dimensions():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.ensure_vec_table(4)
    row = store.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vectors_vec'").fetchone()
    assert row is not None
    assert "float[4]" in row["sql"]
    store.close()


def test_ensure_vec_table_raises_on_dimension_mismatch():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.ensure_vec_table(4)
    try:
        store.ensure_vec_table(8)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "dimension" in str(e).lower()
    store.close()


def test_index_content_inserts_content_vectors_rows():
    store = Store(":memory:", embed_fn=_fake_embed)
    content_hash = store.hash_content("short doc")
    chunk_count = store.index_content(content_hash, "short doc", model="fake-model")
    assert chunk_count == 1
    rows = store.conn.execute(
        "SELECT * FROM content_vectors WHERE hash = ?", (content_hash,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "fake-model"
    store.close()


def test_index_content_inserts_into_vectors_vec():
    store = Store(":memory:", embed_fn=_fake_embed)
    content_hash = store.hash_content("short doc")
    store.index_content(content_hash, "short doc", model="fake-model")
    row = store.conn.execute(
        "SELECT hash_seq FROM vectors_vec WHERE hash_seq = ?", (f"{content_hash}_0",)
    ).fetchone()
    assert row is not None
    store.close()


def test_index_content_is_a_noop_when_already_embedded_with_same_model():
    calls = []

    def counting_embed(texts, model, kind="query", title=None):
        calls.append(texts)
        return _fake_embed(texts, model, kind, title)

    store = Store(":memory:", embed_fn=counting_embed)
    content_hash = store.hash_content("doc")
    store.index_content(content_hash, "doc", model="fake-model")
    store.index_content(content_hash, "doc", model="fake-model")
    assert len(calls) == 1  # second call is a no-op
    store.close()


def test_index_content_multi_chunk_document_inserts_multiple_vectors():
    store = Store(":memory:", embed_fn=_fake_embed)
    long_doc = "# Section\n" + ("word " * 2000)
    content_hash = store.hash_content(long_doc)
    chunk_count = store.index_content(content_hash, long_doc, model="fake-model")
    assert chunk_count > 1
    rows = store.conn.execute(
        "SELECT COUNT(*) as n FROM content_vectors WHERE hash = ?", (content_hash,)
    ).fetchone()
    assert rows["n"] == chunk_count
    store.close()


def test_rebuild_fts_repopulates_from_content_table():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    content_hash = store.hash_content("# Title\nbody")
    store.insert_content(content_hash, "# Title\nbody", "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "a.md", "Title", content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    # Corrupt the FTS index directly, then confirm rebuild fixes it.
    store.conn.execute("DELETE FROM documents_fts")
    store.conn.commit()
    store.rebuild_fts()
    row = store.conn.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH 'title'"
    ).fetchone()
    assert row is not None
    store.close()


def test_get_indexable_content_includes_path():
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("hello world")
    store.insert_content(h, "hello world", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")

    content = store.get_indexable_content()

    assert len(content) == 1
    assert content[0]["path"] == "a.md"
    store.close()


def test_get_indexable_content_path_is_min_across_duplicate_hash():
    # Same content hash indexed at two different paths -- path must be the
    # deterministic MIN(path), matching Node's getPendingEmbeddingDocs.
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("notes", "/notes")
    h = store.hash_content("shared body")
    store.insert_content(h, "shared body", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "z.md", "Z", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.insert_document("notes", "a.md", "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")

    content = store.get_indexable_content()

    assert len(content) == 1
    assert content[0]["path"] == "a.md"
    store.close()


def test_index_content_auto_chunk_strategy_uses_filepath():
    store = Store(":memory:", embed_fn=_fake_embed)
    body = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(400))
    h = store.hash_content(body)

    chunk_count = store.index_content(h, body, filepath="sample.py", chunk_strategy="auto")

    assert chunk_count > 1
    store.close()


def test_index_content_regex_strategy_ignores_filepath():
    # filepath is only consulted when chunk_strategy="auto" -- passing one
    # under the default "regex" strategy must produce the same chunk count
    # chunk_document() itself produces without a filepath. (Comparing
    # against a second index_content() call would risk hitting its
    # already-embedded no-op branch instead of re-chunking -- comparing
    # against chunk_document() directly avoids that pitfall.)
    from pyqmd_mlx.store._chunking import chunk_document

    store = Store(":memory:", embed_fn=_fake_embed)
    body = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(400))
    h = store.hash_content(body)
    expected_chunk_count = len(chunk_document(body))
    assert expected_chunk_count > 1  # sanity: body forces a real split

    chunk_count = store.index_content(h, body, filepath="sample.py")

    assert chunk_count == expected_chunk_count
    store.close()


def _seed_code_doc(store, collection="code", path="sample.py", funcs=400):
    body = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(funcs))
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document(collection, path, path, h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    return h, body


def test_count_pending_embed_detects_strategy_switch():
    # Finding #1: embed (regex) then embed --chunk-strategy auto must not
    # no-op -- the auto boundaries differ, so the hash counts as pending.
    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("code", "/code")
    h, body = _seed_code_doc(store)
    store.index_content(h, body, filepath="sample.py", chunk_strategy="regex")

    assert store.count_pending_embed("code", chunk_strategy="regex") == 0
    assert store.count_pending_embed("code", chunk_strategy="auto") == 1
    store.close()


def test_index_content_updates_positions_on_strategy_switch():
    # Finding #2: same chunk *count* under both strategies must not keep
    # stale regex boundaries -- stored positions must move to the fresh
    # AST-derived ones.
    from pyqmd_mlx.store._chunking import chunk_document

    store = Store(":memory:", embed_fn=_fake_embed)
    store.add_collection("code", "/code")
    h, body = _seed_code_doc(store)
    regex_count = store.index_content(h, body, filepath="sample.py", chunk_strategy="regex")
    auto_count = store.index_content(h, body, filepath="sample.py", chunk_strategy="auto")
    assert regex_count == auto_count  # the count coincidence that hid the bug

    stored = [
        r["pos"]
        for r in store.conn.execute(
            "SELECT pos FROM content_vectors WHERE hash = ? ORDER BY seq", (h,)
        ).fetchall()
    ]
    expected = [
        pos for _text, pos in chunk_document(body, filepath="sample.py", chunk_strategy="auto")
    ]
    assert stored == expected
    assert store.count_pending_embed("code", chunk_strategy="auto") == 0
    store.close()


def test_index_content_skips_reembed_when_boundaries_identical():
    # Position comparison cuts both ways: when both strategies chunk
    # identically, switching strategy must NOT burn embedding work.
    calls = []

    def counting_embed(texts, model, kind="query", title=None):
        calls.append(texts)
        return _fake_embed(texts, model, kind, title)

    store = Store(":memory:", embed_fn=counting_embed)
    body = "short markdown doc"
    h = store.hash_content(body)
    store.index_content(h, body, filepath="notes.md", chunk_strategy="regex")
    store.index_content(h, body, filepath="notes.md", chunk_strategy="auto")

    assert len(calls) == 1
    store.close()


def test_store_rejects_invalid_chunk_strategy():
    store = Store(":memory:", embed_fn=_fake_embed)
    h = store.hash_content("doc")
    for fn in [
        lambda: store.index_content(h, "doc", chunk_strategy="Auto"),
        lambda: store.count_pending_embed(chunk_strategy="bogus"),
        lambda: store.query("hello", chunk_strategy="AUTO"),
    ]:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    store.close()


def _recording_store(calls):
    def recording_embed(texts, model, kind="query", title=None):
        calls.append({"texts": list(texts), "kind": kind, "title": title})
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    return Store(":memory:", embed_fn=recording_embed)


def test_index_content_passes_the_title_from_the_filepath():
    calls = []
    store = _recording_store(calls)
    body = "# setup helpers\n\ndef setup():\n    pass\n"
    store.index_content(store.hash_content(body), body, filepath="src/helpers.py")
    assert calls == [{"texts": [body], "kind": "document", "title": "helpers"}]
    store.close()


def test_index_content_without_filepath_uses_the_documents_min_path():
    calls = []
    store = _recording_store(calls)
    store.add_collection("notes", "/notes")
    body = "# Rivers\nwater"
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    # Inserted MAX-first, so neither "first inserted" nor MAX(path) passes.
    for path in ("b/copy.md", "a/rivers.txt"):
        store.insert_document("notes", path, "t", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    store.index_content(h, body)
    # MIN(path) is a/rivers.txt: no .txt extractor, so the file name wins.
    assert calls[0]["title"] == "rivers"
    store.close()


def test_index_content_with_no_document_and_no_filepath_embeds_untitled():
    calls = []
    store = _recording_store(calls)
    store.index_content(store.hash_content("bare"), "bare")
    assert calls[0]["title"] is None
    store.close()
