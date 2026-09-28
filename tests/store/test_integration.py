import pytest

from pyqmd_mlx.store import Store


@pytest.mark.slow
def test_store_query_end_to_end_with_real_models():
    store = Store(":memory:")
    store.add_collection("docs", "/docs")

    body_a = "# Authentication\nHow to configure authentication for your application using API keys and OAuth tokens."
    body_b = "# Cooking\nA recipe for making fresh pasta from scratch with flour and eggs."

    for path, title, body in (
        ("auth.md", "Authentication", body_a),
        ("cooking.md", "Cooking", body_b),
    ):
        content_hash = store.hash_content(body)
        store.insert_content(content_hash, body, "2026-01-01T00:00:00Z")
        store.insert_document(
            "docs", path, title, content_hash, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
        )
        store.index_content(content_hash, body)

    results = store.query("how do I set up authentication")
    assert len(results) >= 1
    assert results[0].title == "Authentication"
    store.close()
