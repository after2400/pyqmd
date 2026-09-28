"""Score the rerank-fixture.json queries with the MLX candidate reranker.

Usage: cd python && uv run scripts/score_rerank_fixture.py
  [--fixture data/scifact/rerank-fixture.json]
  [--out data/scifact/candidate-scores.json] [--model <mlx model id>]
"""

import argparse
import json
import time
from pathlib import Path

from pyqmd_mlx.llm import DEFAULT_RERANK_MODEL, rerank

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def build_score_record(doc_ids: list[str], scores: list[float]) -> list[dict]:
    return [{"doc_id": doc_id, "score": score} for doc_id, score in zip(doc_ids, scores)]


def score_fixture(fixture: list[dict], model: str = DEFAULT_RERANK_MODEL) -> dict:
    output = {}
    for q in fixture:
        doc_ids = [d["doc_id"] for d in q["candidate_docs"]]
        texts = [d["text"] for d in q["candidate_docs"]]

        start = time.monotonic()
        scores = rerank(q["query"], texts, model=model) if texts else []
        latency_ms = (time.monotonic() - start) * 1000

        output[q["query_id"]] = {
            "scores": build_score_record(doc_ids, scores),
            "latency_ms": latency_ms,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(DEFAULT_DATA_DIR / "rerank-fixture.json"))
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR / "candidate-scores.json"))
    parser.add_argument("--model", default=DEFAULT_RERANK_MODEL)
    args = parser.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    output = score_fixture(fixture, model=args.model)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"Wrote candidate scores for {len(fixture)} queries to {args.out}")


if __name__ == "__main__":
    main()
