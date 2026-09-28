"""Run the full Store.query() pipeline over the existing BEIR SciFact
fixture (built in sub-project #1) and report IR metrics. Not a strict
"must match Node exactly" comparison like sub-project #1's rerank harness
-- the individual pieces (embed/rerank/expand) are already validated
there; this confirms the *assembled* pipeline behaves sensibly.

Relevance judgments are NOT embedded in rerank-fixture.json (its candidate
docs carry no `relevant` field) -- they live in the separate BEIR qrels
file `data/scifact/qrels-test.tsv` (tab-separated, header
`query-id\tcorpus-id\tscore`, one row per relevant (query, doc) pair).
This script loads that file and builds a `query_id -> {relevant doc ids}`
map, used instead of a nonexistent `relevant` field.

Usage: cd python && uv run scripts/validate_store_query.py
"""

import csv
import json
from pathlib import Path

from pyqmd_mlx.store import Store

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def load_fixture() -> list[dict]:
    return json.loads((DATA_DIR / "rerank-fixture.json").read_text())


def load_qrels() -> dict[str, set[str]]:
    """Build query_id -> {relevant corpus/doc ids} from qrels-test.tsv."""
    relevance: dict[str, set[str]] = {}
    with (DATA_DIR / "qrels-test.tsv").open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            relevance.setdefault(query_id, set()).add(corpus_id)
    return relevance


def build_store(fixture: list[dict]) -> Store:
    store = Store(":memory:")
    store.add_collection("scifact", "/scifact")
    seen_docs: set[str] = set()
    for query_entry in fixture:
        for doc in query_entry["candidate_docs"]:
            doc_id = doc["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            content_hash = store.hash_content(doc["text"])
            store.insert_content(content_hash, doc["text"], "2026-01-01T00:00:00Z")
            store.insert_document(
                "scifact",
                f"{doc_id}.md",
                doc_id,
                content_hash,
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            )
            store.index_content(content_hash, doc["text"])
    return store


def mean_reciprocal_rank(fixture: list[dict], store: Store, qrels: dict[str, set[str]]) -> float:
    reciprocal_ranks = []
    for query_entry in fixture:
        results = store.query(query_entry["query"], limit=10, collection="scifact")
        relevant_doc_ids = qrels.get(query_entry["query_id"], set())
        if not relevant_doc_ids:
            continue
        rank = next((i + 1 for i, r in enumerate(results) if r.title in relevant_doc_ids), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def main() -> None:
    fixture = load_fixture()
    qrels = load_qrels()
    print(f"Building store from {len(fixture)} queries' candidate docs...")
    store = build_store(fixture)
    print("Running Store.query() over all queries...")
    mrr = mean_reciprocal_rank(fixture, store, qrels)
    print(f"MRR: {mrr:.3f}")
    store.close()


if __name__ == "__main__":
    main()
