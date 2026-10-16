"""Does the heading-breadcrumbs gain reach the reranked ranking?

Same corpus, questions, stores and metrics as bench_breadcrumbs_hybrid.py, but
each question goes through `Store.query(limit=40)` with the real reranker on
(query expansion off). The rule is fixed in advance in the feature spec's
"Amendment 2: reranker-stage check". Never touches the user's index.

The reranker is slow, so this is built to be run by hand and left alone:
progress goes to the terminal and to a log file, every question's ranking is
appended to a JSONL checkpoint as it finishes, and running it again resumes
where it stopped. Ctrl-C is safe. Prefix with `caffeinate -i` to keep the Mac
awake.

Usage: uv run scripts/bench_breadcrumbs_rerank.py [--data-dir data/conditionalqa]
  [--out PATH] [--log PATH] [--checkpoint PATH] [--resamples 10000]
  [--max-pages N]      smoke run on N pages (own checkpoint)
  [--max-questions N]  first N questions only (timing run; resumable into the full run)
"""

import argparse
import json
import time
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
from bench_breadcrumbs_hybrid import (
    LIMIT,
    METRICS,
    build_store,
    page_rank_metrics,
    verdict,
)

from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL, embed
from pyqmd_mlx.store._fingerprint import embedding_fingerprint

ARMS = ("off", "on")


class Logger:
    """Prints each line and appends it to a log file (flushed, so `tail -f` works)."""

    def __init__(self, path: Path | None):
        self.path = path

    def say(self, line: str = "") -> None:
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def reranked_pages(store, query: str, limit: int = LIMIT) -> list[str]:
    """The pages the full pipeline (FTS + vector + RRF + reranker, no expansion)
    returns for `query`, best first."""
    results = store.query(query, limit=limit)
    return [Path(r.file).stem for r in results]


def format_progress(done: int, total: int, elapsed: float, resumed: int) -> str:
    """`done` counts every finished question; `resumed` of them came from the
    checkpoint, so the rate uses only the ones this session did."""
    this_session = max(done - resumed, 1)
    rate = elapsed / this_session
    left = round(rate * (total - done) / 60)
    return f"{done}/{total} questions, {rate:.1f} s/question, ~{left} min left"


def load_checkpoint(path: Path, header: dict) -> dict[int, dict]:
    """Finished questions by index. A header that differs from `header` means
    the file belongs to another setup; a torn last line (killed mid-write) is
    ignored."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    if json.loads(lines[0]) != header:
        raise ValueError(f"checkpoint {path} was made by a different setup; move it aside")
    done = {}
    for position, line in enumerate(lines[1:], start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1:
                break
            raise
        done[record["i"]] = record
    return done


def run_questions(
    stores: dict, questions, checkpoint: Path, header: dict, log: Logger, *, log_every: int = 25
) -> dict[str, dict[str, np.ndarray]]:
    """Both arms for every question, asking only the ones the checkpoint lacks."""
    done = load_checkpoint(checkpoint, header)
    for i, record in done.items():
        if i < len(questions) and record["q"] != questions[i].query:
            raise ValueError(f"checkpoint question {i} is not the one being run now")
    # Rewriting also drops a torn trailing line, so appends start on a clean line.
    lines = [json.dumps(header)] + [json.dumps(done[i]) for i in sorted(done)]
    checkpoint.write_text("\n".join(lines) + "\n", encoding="utf-8")
    resumed = sum(1 for i in done if i < len(questions))
    started = time.monotonic()
    rows: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
    for i, question in enumerate(questions):
        record = done.get(i)
        if record is None:
            record = {
                "i": i,
                "q": question.query,
                **{arm: reranked_pages(stores[arm], question.query) for arm in ARMS},
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            finished = i + 1
            fresh = finished - resumed
            if fresh == 3 or fresh % log_every == 0:  # an early rate estimate
                elapsed = time.monotonic() - started
                log.say(format_progress(finished, len(questions), elapsed, resumed))
        for arm in ARMS:
            rows[arm].append(page_rank_metrics(record[arm], question.page))
    return {arm: {m: np.array([r[m] for r in rows[arm]]) for m in METRICS} for arm in ARMS}


def rerank_verdict(diffs: dict[str, tuple[float, float, float]]) -> tuple[int, str]:
    """Amendment 2's rule: Amendment 1's thresholds and precedence, different
    consequences (the default is already decided as opt-in)."""
    code, _text = verdict(diffs)
    texts = {
        1: "1: nDCG@10's interval is above zero; the gain reaches the final ranking. "
        "Reopen the default question with the user (nothing flips automatically)",
        2: "2: no evidence of a final-ranking gain; stays opt-in, README keeps saying "
        "the benefit is established at the retrieval stages only",
        3: "3: a metric's point estimate fell below -0.01; record a regression in the README",
    }
    return code, texts[code]


def build_stores(corpus_dir: Path, *, model: str, embed_fn: Callable, log: Logger) -> dict:
    stores = {}
    for arm in ARMS:
        started = time.monotonic()
        log.say(f"Indexing and embedding the corpus ({arm})...")
        stores[arm] = build_store(
            corpus_dir, breadcrumbs=arm == "on", embed_fn=embed_fn, model=model
        )
        log.say(f"  done in {time.monotonic() - started:.0f} s")
    return stores


def format_report(selection, scored: int, means, diffs) -> str:
    _code, text = rerank_verdict(diffs)
    partial = scored < len(selection.questions)
    lines = [
        "# Heading breadcrumbs: reranker-stage check",
        "",
        f"- Questions scored: {scored} of {len(selection.questions)} (strict labelling; "
        f"{selection.ambiguous} excluded as ambiguous)",
        "- Pipeline: FTS + vector + RRF + reranker over the top 40, no query expansion",
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
    if partial:
        lines += ["", "## Not a verdict: partial run (timing only)", ""]
    else:
        lines += ["", f"## Verdict {text}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Breadcrumbs: reranker-stage check")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--max-pages", type=int, default=None, help="Smoke run on N pages")
    parser.add_argument("--max-questions", type=int, default=None, help="First N questions only")
    args = parser.parse_args()

    stem = "breadcrumbs-rerank" if args.max_pages is None else "breadcrumbs-rerank-smoke"
    out = args.out or args.data_dir / f"{stem}.md"
    log = Logger(args.log or args.data_dir / f"{stem}.log")
    checkpoint = args.checkpoint or args.data_dir / f"{stem}.jsonl"
    log.say(f"--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    pages, source, _subset = load_dataset(args.data_dir, args.max_pages, "train")
    chunks = build_chunks(pages)
    selection = select_questions(source, pages, chunks)
    if args.max_pages is None:
        check_exclusions(selection, 0.15)
    questions = selection.questions[: args.max_questions]
    log.say(f"{len(questions)} of {len(selection.questions)} questions, {len(pages)} pages")

    header = {
        "model": args.model,
        "limit": LIMIT,
        "pages": len(pages),
        "fingerprint_off": embedding_fingerprint(args.model, False),
        "fingerprint_on": embedding_fingerprint(args.model, True),
    }
    stores = build_stores(args.data_dir / "corpus", model=args.model, embed_fn=embed, log=log)
    results = run_questions(stores, questions, checkpoint, header, log)
    for store in stores.values():
        store.close()

    clusters = [q.page for q in questions]
    diffs = {
        m: cluster_bootstrap(results["on"][m] - results["off"][m], clusters, args.resamples)
        for m in METRICS
    }
    means = {a: {m: float(results[a][m].mean()) for m in METRICS} for a in ARMS}
    report = format_report(selection, len(questions), means, diffs)
    out.write_text(report, encoding="utf-8")
    log.say(report)
    log.say(f"Wrote {out}")


if __name__ == "__main__":
    main()
