"""Does the heading-breadcrumbs gain survive fusion with full-text search?

Builds two real in-memory Stores over the ConditionalQA corpus, one embedding
`.md` chunks with their section-heading path and one without, then asks each
the train questions through the product's own FTS + vector + RRF fusion
(`Store.query(skip_rerank=True)` with query expansion switched off). Compares
the gold page's rank at the page level with a page-clustered bootstrap. The
rule is fixed in advance in the feature spec's "Amendment 1: hybrid-stage
check". Never touches the user's index.

Usage: uv run scripts/bench_breadcrumbs_hybrid.py [--data-dir data/conditionalqa]
  [--out PATH] [--resamples 10000] [--max-pages N]  (N pages: smoke run)
"""

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from bench_breadcrumbs import (
    DEFAULT_DATA_DIR,
    build_chunks,
    check_exclusions,
    cluster_bootstrap,
    load_dataset,
    select_questions,
)

from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL, embed
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import scan_and_register_collection

LIMIT = 40
METRICS = ("page_mrr", "page_ndcg10", "page_recall10")
REGRESSION = -0.01


def page_rank_metrics(ranked: list[str], gold: str) -> dict[str, float]:
    """Page-level metrics for one question: `ranked` is the result pages, best
    first. A gold page that is absent scores 0."""
    if gold not in ranked:
        return dict.fromkeys(METRICS, 0.0)
    rank = ranked.index(gold) + 1
    return {
        "page_mrr": 1 / rank,
        "page_ndcg10": 1 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "page_recall10": float(rank <= 10),
    }


def no_expansion(query: str, model: str) -> list[str]:
    """An expand_fn that adds nothing, so only the original query is searched."""
    return []


def hybrid_ranked_pages(store, query: str, limit: int = LIMIT) -> list[str]:
    """The pages FTS + vector + RRF return for `query`, best first."""
    results = store.query(query, limit=limit, skip_rerank=True)
    return [Path(r.file).stem for r in results]


def build_store(corpus_dir: Path, *, breadcrumbs: bool, embed_fn: Callable, model: str) -> Store:
    """An in-memory Store with the corpus indexed and embedded the product's way."""
    store = Store(
        ":memory:",
        embed_fn=embed_fn,
        expand_fn=no_expansion,
        embed_model=model,
        breadcrumbs=breadcrumbs,
    )
    store.add_collection("corpus", str(corpus_dir), pattern="**/*")
    scan_and_register_collection(store, str(corpus_dir), "**/*", "corpus")
    for row in store.get_indexable_content("corpus"):
        store.index_content(row["hash"], row["doc"], filepath=row["path"])
    return store


def evaluate(store, questions) -> dict[str, np.ndarray]:
    rows = [page_rank_metrics(hybrid_ranked_pages(store, q.query), q.page) for q in questions]
    return {m: np.array([r[m] for r in rows]) for m in METRICS}


def verdict(diffs: dict[str, tuple[float, float, float]]) -> tuple[int, str]:
    """The rule from the spec, in its stated precedence (3 over 1 over 2).
    `diffs` maps each metric to (mean, lower, upper) of on minus off."""
    worst = min(mean for mean, _lo, _hi in diffs.values())
    if worst < REGRESSION:
        return 3, (
            f"3: a metric's point estimate fell below {REGRESSION}; ship with the default off"
        )
    _mean, lo, _hi = diffs["page_ndcg10"]
    if lo > 0:
        return 1, "1: nDCG@10's interval is above zero; the gain survives fusion, ship default on"
    return 2, "2: inconclusive at this stage (interval spans zero, no regression); the user decides"


def run(corpus_dir, pages, questions_source, *, model, embed_fn, resamples, strict):
    chunks = build_chunks(pages)
    selection = select_questions(questions_source, pages, chunks)
    if strict:
        check_exclusions(selection, 0.15)
    questions = selection.questions
    results = {}
    for name, flag in (("off", False), ("on", True)):
        store = build_store(corpus_dir, breadcrumbs=flag, embed_fn=embed_fn, model=model)
        results[name] = evaluate(store, questions)
        store.close()
    clusters = [q.page for q in questions]
    diffs = {
        m: cluster_bootstrap(results["on"][m] - results["off"][m], clusters, resamples)
        for m in METRICS
    }
    means = {n: {m: float(results[n][m].mean()) for m in METRICS} for n in results}
    return selection, means, diffs


def format_report(selection, means, diffs) -> str:
    _code, text = verdict(diffs)
    lines = [
        "# Heading breadcrumbs: hybrid-stage check",
        "",
        f"- Questions scored: {len(selection.questions)} (strict labelling; "
        f"{selection.ambiguous} excluded as ambiguous)",
        "- Pipeline: FTS + vector + RRF, no query expansion, no reranker, top 40",
        "",
        "| Metric | off | on | on − off (95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for m in METRICS:
        mean, lo, hi = diffs[m]
        lines.append(
            f"| {m} | {means['off'][m]:.4f} | {means['on'][m]:.4f} | "
            f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}] |"
        )
    lines += ["", f"## Verdict {text}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Breadcrumbs: hybrid-stage check")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--max-pages", type=int, default=None, help="Smoke run on N pages")
    args = parser.parse_args()

    pages, questions, _subset = load_dataset(args.data_dir, args.max_pages, "train")
    selection, means, diffs = run(
        args.data_dir / "corpus",
        pages,
        questions,
        model=args.model,
        embed_fn=embed,
        resamples=args.resamples,
        strict=args.max_pages is None,
    )
    report = format_report(selection, means, diffs)
    out = args.out or args.data_dir / (
        "breadcrumbs-hybrid.md" if args.max_pages is None else "breadcrumbs-hybrid-smoke.md"
    )
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
