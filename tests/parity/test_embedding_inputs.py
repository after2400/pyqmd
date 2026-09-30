"""Model-input parity: the exact strings pyqmd hands the embedding model
must equal Node's, for every document in parity/fixtures/embedding_inputs/.
Node's side is pinned in node_expected.json by
`python -m parity.capture_node_snapshots --phase inputs` (manual).

The model ids differ (MLX vs. GGUF builds of embeddinggemma); the formats
don't depend on the id for non-Qwen models, so pyqmd's default model is
used on this side.
"""

import json
import shutil
from pathlib import Path

import pytest

from pyqmd_mlx.llm._constants import DEFAULT_EMBED_MODEL
from pyqmd_mlx.llm._prompts import format_embedding_inputs
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._chunking import embedding_chunks
from pyqmd_mlx.store._indexing import scan_and_register_collection
from pyqmd_mlx.store._title import extract_title

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "parity" / "fixtures" / "embedding_inputs"
NOT_DOCUMENTS = {"node_expected.json", "queries.json"}
EXPECTED = json.loads((FIXTURES / "node_expected.json").read_text(encoding="utf-8"))


def _read(relpath: str) -> str:
    # Bytes decoded as-is, like Node's readFileSync(path, "utf-8").
    return (FIXTURES / relpath).read_bytes().decode("utf-8")


def _pyqmd_record(relpath: str) -> dict:
    content = _read(relpath)
    title = extract_title(content, relpath)
    chunks = embedding_chunks(content, relpath)
    formatted = format_embedding_inputs(
        [text for text, _pos in chunks], DEFAULT_EMBED_MODEL, "document", title
    )
    return {
        "title": title,
        "chunks": [
            {"pos": pos, "text": text, "formatted": f}
            for (text, pos), f in zip(chunks, formatted, strict=True)
        ],
    }


def test_node_inputs_come_from_the_pinned_node_commit():
    pinned = (REPO / "parity/node_ref/scifact/COMMIT.txt").read_text().splitlines()[0].strip()
    assert EXPECTED["node_commit"] == pinned


def test_every_fixture_was_captured():
    on_disk = sorted(
        p.relative_to(FIXTURES).as_posix()
        for p in FIXTURES.rglob("*")
        if p.is_file() and p.name not in NOT_DOCUMENTS
    )
    assert on_disk == sorted(EXPECTED["documents"])


@pytest.mark.parametrize("relpath", sorted(EXPECTED["documents"]))
def test_document_inputs_match_node(relpath):
    assert _pyqmd_record(relpath) == EXPECTED["documents"][relpath]


def test_long_fixture_really_splits():
    assert len(EXPECTED["documents"]["long-sections.md"]["chunks"]) >= 3


def test_query_inputs_match_node():
    queries = list(EXPECTED["queries"])
    assert format_embedding_inputs(queries, DEFAULT_EMBED_MODEL, "query") == list(
        EXPECTED["queries"].values()
    )


def test_index_content_hands_the_embedder_node_matching_inputs(tmp_path):
    """The call-site link that broke for gap 1: the formatter was right,
    the caller never passed a title. Runs the real scan + index path."""
    calls = []

    def recording_embed(texts, model, kind="query", title=None):
        calls.append((list(texts), kind, title))
        return [[1.0, 0.0] for _ in texts]

    root = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, root, ignore=shutil.ignore_patterns(*NOT_DOCUMENTS))
    store = Store(":memory:", embed_fn=recording_embed)
    store.add_collection("fx", str(root), pattern="**/*")
    scan_and_register_collection(store, str(root), "**/*", "fx")
    for row in store.get_indexable_content("fx"):
        store.index_content(row["hash"], row["doc"], filepath=row["path"])
    store.close()

    assert {kind for _texts, kind, _title in calls} == {"document"}
    actual = {
        tuple(format_embedding_inputs(texts, DEFAULT_EMBED_MODEL, kind, title))
        for texts, kind, title in calls
    }
    expected = {
        tuple(chunk["formatted"] for chunk in doc["chunks"])
        for doc in EXPECTED["documents"].values()
    }
    assert actual == expected


def test_fingerprint_port_matches_nodes_value():
    from pyqmd_mlx.store._fingerprint import embedding_fingerprint

    assert embedding_fingerprint(EXPECTED["model"]) == EXPECTED["fingerprint"]
