from pyqmd_mlx.store import Store
from pyqmd_mlx.store._fingerprint import embedding_fingerprint

MODEL = "fake-model"


def _counting_store(calls):
    def counting_embed(texts, model, kind="query", title=None):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    return Store(":memory:", embed_fn=counting_embed, embed_model=MODEL)


def _seed(store, path="a.md", body="# A\nalpha"):
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document("notes", path, "A", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    return h, body


def test_fingerprint_values_are_pinned():
    # Port of store.ts getEmbeddingFingerprint: sha256 over the model id,
    # the formatted probes and the chunk token sizes, first 6 hex chars.
    assert embedding_fingerprint("mlx-community/embeddinggemma-300m-8bit") == "aa71c5"
    assert embedding_fingerprint("mlx-community/Qwen3-Embedding-0.6B") == "1bee40"


def test_index_content_stores_the_current_fingerprint_on_every_row():
    store = _counting_store([])
    store.add_collection("notes", "/notes")
    h, body = _seed(store, body="# A\n" + "filler words here. " * 400)
    store.index_content(h, body)
    rows = store.conn.execute(
        "SELECT DISTINCT embed_fingerprint FROM content_vectors WHERE hash = ?", (h,)
    ).fetchall()
    assert [r["embed_fingerprint"] for r in rows] == [embedding_fingerprint(MODEL)]
    store.close()


def test_rows_without_the_current_fingerprint_are_re_embedded():
    calls = []
    store = _counting_store(calls)
    store.add_collection("notes", "/notes")
    h, body = _seed(store)
    store.index_content(h, body)
    # Simulate an index written before fingerprints existed.
    store.conn.execute("UPDATE content_vectors SET embed_fingerprint = ''")
    store.conn.commit()

    store.index_content(h, body)
    store.index_content(h, body)

    assert len(calls) == 2  # re-embedded once, then current
    store.close()


def test_pending_counts_include_stale_and_partial_hashes():
    store = _counting_store([])
    store.add_collection("notes", "/notes")
    current, body_c = _seed(store, "c.md", "# C\ncurrent")
    stale, body_s = _seed(store, "s.md", "# S\nstale")
    partial, body_p = _seed(store, "p.md", "# P\n" + "filler words here. " * 400)
    for h, body in ((current, body_c), (stale, body_s), (partial, body_p)):
        store.index_content(h, body)
    store.conn.execute("UPDATE content_vectors SET embed_fingerprint = '' WHERE hash = ?", (stale,))
    store.conn.execute("DELETE FROM content_vectors WHERE hash = ? AND seq = 1", (partial,))
    store.conn.commit()

    assert store.count_pending_embed() == 2
    assert store.count_pending_embed("notes") == 2
    assert store.get_status_counts()["pending_embed"] == 2
    store.close()
