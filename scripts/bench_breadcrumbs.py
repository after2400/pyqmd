"""Heading-breadcrumbs experiment: does prefixing a chunk's embedded title
with its section-heading path make vector search find the right chunk more
often? Embeds every chunk of the ConditionalQA corpus three ways (`base`,
`full`, `last`), searches by cosine similarity in memory, and reports
chunk-level and page-level metrics with paired, page-clustered bootstrap
intervals. Vector-only: no FTS, RRF, expansion or rerank. Never opens the
user's index.

Usage: uv run scripts/bench_breadcrumbs.py [--data-dir data/conditionalqa]
  [--out PATH] [--model <mlx embed model id>] [--resamples 10000]
  [--max-pages N]   (N pages only, for a smoke run; skips the exclusion check)

See docs/specs/2026-10-15-heading-breadcrumbs-experiment-design.md.
"""

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from prepare_conditionalqa_corpus import _parse_item, page_slug

from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL, embed
from pyqmd_mlx.store._breadcrumbs import dominant_heading_path, find_headings
from pyqmd_mlx.store._chunking import embedding_chunks
from pyqmd_mlx.store._fingerprint import embedding_fingerprint
from pyqmd_mlx.store._title import extract_title

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "conditionalqa"
EMBED_BATCH = 16
VARIANTS = ("base", "full", "last")


def variant_title(variant: str, doc_title: str, path: list[str]) -> str:
    """The `title` handed to embed(kind="document") for one chunk."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "base" or not path:
        return doc_title
    if variant == "full":
        return " > ".join([doc_title, *path])
    return f"{doc_title} > {path[-1]}"


@dataclass(frozen=True)
class Chunk:
    page: str
    pos: int
    text: str
    title: str  # the document title `embed` uses today
    path: tuple[str, ...]  # heading path of the section holding most of the chunk


def build_chunks(pages: dict[str, str]) -> list[Chunk]:
    """Every page's embed-time chunks (the same ones `pyqmd embed` stores),
    sorted by page slug so a page's chunks are contiguous."""
    chunks: list[Chunk] = []
    for slug in sorted(pages):
        content = pages[slug]
        filename = f"{slug}.md"
        title = extract_title(content, filename)
        headings = find_headings(content)
        for text, pos in embedding_chunks(content, filename):
            path = tuple(dominant_heading_path(headings, pos, len(text)))
            chunks.append(Chunk(slug, pos, text, title, path))
    return chunks


def chunks_by_page(chunks: list[Chunk]) -> dict[str, list[int]]:
    by_page: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        by_page.setdefault(chunk.page, []).append(index)
    return by_page


def _evidence_text(item: str) -> str:
    """The evidence item's text as prepare_conditionalqa_corpus writes it
    (entities unescaped, whitespace collapsed); "" if it cannot be parsed."""
    try:
        _, text = _parse_item(item)
    except ValueError:
        return ""
    return text


def evidence_spans(content: str, evidences: list[str]) -> tuple[list[tuple[int, int]], int]:
    """Character spans of every occurrence of each evidence item in the page's
    markdown, plus how many items were not found (or could not be parsed).
    The runs 1 and 2 labelling ("all")."""
    spans: list[tuple[int, int]] = []
    missing = 0
    for item in evidences:
        text = _evidence_text(item)
        found = [m.start() for m in re.finditer(re.escape(text), content)] if text else []
        if not found:
            missing += 1
            continue
        spans.extend((start, start + len(text)) for start in found)
    return spans, missing


def strict_evidence_spans(
    content: str, evidences: list[str]
) -> tuple[list[tuple[int, int]], int, int]:
    """Spans of the evidence items that occur exactly once on the page, matched
    on word boundaries, plus the number of items missing (not found or
    unparseable) and ambiguous (more than one occurrence). Amendment 1's
    labelling: a repeated item such as "You must:" cannot say which of its
    occurrences answers the question, so it is dropped, not spread over every
    chunk that contains it."""
    spans: list[tuple[int, int]] = []
    missing = 0
    ambiguous = 0
    for item in evidences:
        text = _evidence_text(item)
        pattern = rf"(?<!\w){re.escape(text)}(?!\w)"
        found = [m.start() for m in re.finditer(pattern, content)] if text else []
        if not found:
            missing += 1
        elif len(found) > 1:
            ambiguous += 1
        else:
            spans.append((found[0], found[0] + len(text)))
    return spans, missing, ambiguous


@dataclass(frozen=True)
class EvalQuestion:
    query_id: str
    query: str
    page: str
    gold: frozenset[int]  # indices into the chunk list


@dataclass
class Selection:
    questions: list = field(default_factory=list)
    unanswerable: int = 0
    no_evidence: int = 0
    page_missing: int = 0
    unmatched: int = 0
    ambiguous: int = 0  # every located evidence item occurs more than once
    items_missing: int = 0
    items_ambiguous: int = 0

    @property
    def eligible(self) -> int:
        """Answerable questions with evidence: the pool the exclusion check
        is measured against."""
        return len(self.questions) + self.page_missing + self.unmatched + self.ambiguous


def select_questions(
    dev: list[dict], pages: dict[str, str], chunks: list[Chunk], labelling: str = "strict"
) -> Selection:
    if labelling not in ("strict", "all"):
        raise ValueError(f"unknown labelling {labelling!r}; expected 'strict' or 'all'")
    by_page = chunks_by_page(chunks)
    selection = Selection()
    for question in dev:
        if question["not_answerable"]:
            selection.unanswerable += 1
            continue
        if not question["evidences"]:
            selection.no_evidence += 1
            continue
        try:
            slug = page_slug(question["url"])
        except ValueError:
            slug = ""
        if slug not in by_page:
            selection.page_missing += 1
            continue
        if labelling == "strict":
            spans, missing, ambiguous = strict_evidence_spans(pages[slug], question["evidences"])
        else:
            spans, missing = evidence_spans(pages[slug], question["evidences"])
            ambiguous = 0
        selection.items_missing += missing
        selection.items_ambiguous += ambiguous
        gold = frozenset(
            index
            for index in by_page[slug]
            if any(
                chunks[index].pos < end and start < chunks[index].pos + len(chunks[index].text)
                for start, end in spans
            )
        )
        if not gold:
            if ambiguous:
                selection.ambiguous += 1
            else:
                selection.unmatched += 1
            continue
        selection.questions.append(
            EvalQuestion(question["id"], question["question"].strip(), slug, gold)
        )
    return selection


def check_exclusions(selection: Selection, max_fraction: float = 0.10) -> None:
    excluded = selection.page_missing + selection.unmatched + selection.ambiguous
    if selection.eligible and excluded / selection.eligible > max_fraction:
        raise RuntimeError(
            f"{excluded} of {selection.eligible} eligible questions excluded "
            f"({selection.page_missing} page missing, {selection.unmatched} evidence not "
            f"located, {selection.ambiguous} ambiguous): over {max_fraction:.0%}, "
            f"so the matching is suspect"
        )


METRICS = (
    "chunk_mrr",
    "chunk_hit1",
    "chunk_hit5",
    "chunk_hit10",
    "in_page_hit1",
    "page_mrr",
    "page_ndcg10",
    "page_recall10",
)
PAGE_GUARD_METRICS = ("page_mrr", "page_ndcg10", "page_recall10")


def query_metrics(
    scores: np.ndarray, bounds: np.ndarray, page_index: int, gold: list[int]
) -> dict[str, float]:
    """Metrics for one query. `scores` holds one cosine score per chunk and
    `bounds[i]:bounds[i + 1]` are page i's chunks. A rank is 1 plus the number
    of items scoring strictly higher, so a tie with the gold chunk is not a
    loss (real float embeddings essentially never tie)."""
    chunk_rank = 1 + int((scores > scores[gold].max()).sum())
    lo, hi = int(bounds[page_index]), int(bounds[page_index + 1])
    in_page_top = lo + int(np.argmax(scores[lo:hi]))
    page_scores = np.maximum.reduceat(scores, bounds[:-1])
    page_rank = 1 + int((page_scores > page_scores[page_index]).sum())
    return {
        "chunk_mrr": 1 / chunk_rank,
        "chunk_hit1": float(chunk_rank <= 1),
        "chunk_hit5": float(chunk_rank <= 5),
        "chunk_hit10": float(chunk_rank <= 10),
        "in_page_hit1": float(in_page_top in gold),
        "page_mrr": 1 / page_rank,
        "page_ndcg10": 1 / math.log2(page_rank + 1) if page_rank <= 10 else 0.0,
        "page_recall10": float(page_rank <= 10),
    }


def cluster_bootstrap(
    values: np.ndarray,
    clusters: list[str],
    resamples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of `values` with a `level` (95% by default) percentile interval from resampling whole
    clusters (gold pages) with replacement: several dev questions share a
    page and are not independent."""
    names = sorted(set(clusters))
    index = {name: i for i, name in enumerate(names)}
    ids = np.array([index[c] for c in clusters])
    sums = np.bincount(ids, weights=values, minlength=len(names))
    counts = np.bincount(ids, minlength=len(names)).astype(float)
    draw = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(values.mean()), float(lo), float(hi)


def decide(
    chunk_mrr_ci: tuple[float, float, float],
    subset_deltas: dict[str, float],
    n_subset: int,
    tolerance: float | None = None,
    label: str = "quality-test queries",
) -> tuple[bool, list[str]]:
    """The spec's rule, fixed in advance: write the feature spec only if the
    chunk-level MRR gain's 95% interval is above zero AND no page-level metric
    on the guard set drops by more than the tolerance. The tolerance is one
    query's worth (1 / n_subset) unless given (Amendment 1: a fixed 0.01 over
    all scored train questions)."""
    if n_subset <= 0:
        raise ValueError("no quality-test queries among the eligible questions")
    limit = 1 / n_subset if tolerance is None else tolerance
    lo = chunk_mrr_ci[1]
    gain_ok = lo > 0
    reasons = [
        f"chunk-level MRR gain: 95% interval lower bound {lo:+.4f} ({'> 0' if gain_ok else '<= 0'})"
    ]
    regressed = [m for m in PAGE_GUARD_METRICS if subset_deltas[m] < -(limit + 1e-9)]
    for metric in PAGE_GUARD_METRICS:
        reasons.append(
            f"{metric} on {n_subset} {label}: {subset_deltas[metric]:+.4f} "
            f"({'regression' if metric in regressed else 'ok'}; limit -{limit:.4f})"
        )
    return gain_ok and not regressed, reasons


EmbedFn = Callable[..., list[list[float]]]


def _normalise(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _cache_key(model: str, *parts: object) -> str:
    """The embedding fingerprint (model, input format, chunk sizes) is part of
    the key, so a change to how pyqmd formats embedding input never reuses
    vectors made under the old format."""
    key = (embedding_fingerprint(model), model, *parts)
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def embed_variant(
    chunks: list[Chunk], variant: str, model: str, embed_fn: EmbedFn, cache_dir: Path | None
) -> tuple[np.ndarray, float]:
    """`embed_inputs` for one breadcrumb variant's titles over the chunks' own text."""
    titles = [variant_title(variant, c.title, list(c.path)) for c in chunks]
    texts = [c.text for c in chunks]
    return embed_inputs(titles, texts, variant, model, embed_fn, cache_dir)


def embed_inputs(
    titles: list[str],
    texts: list[str],
    label: str,
    model: str,
    embed_fn: EmbedFn,
    cache_dir: Path | None,
) -> tuple[np.ndarray, float]:
    """Normalised document vectors for every (title, text) pair, and the
    seconds spent embedding (from the cache's sidecar on a hit). Inputs that
    share a title string go through one embed() call (it takes one title per
    call). The cache key covers the model, every title and every text; `label`
    only names the cache file."""
    stem = None
    if cache_dir is not None:
        stem = cache_dir / f"{label}-{_cache_key(model, titles, texts)}"
        if stem.with_suffix(".npy").exists() and stem.with_suffix(".json").exists():
            meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            return np.load(stem.with_suffix(".npy")), float(meta["seconds"])

    groups: dict[str, list[int]] = {}
    for index, title in enumerate(titles):
        groups.setdefault(title, []).append(index)
    out: list[list[float] | None] = [None] * len(texts)
    seconds = 0.0
    for title, indices in groups.items():
        for start in range(0, len(indices), EMBED_BATCH):
            batch = indices[start : start + EMBED_BATCH]
            began = time.perf_counter()
            vectors = embed_fn([texts[i] for i in batch], model, kind="document", title=title)
            seconds += time.perf_counter() - began
            for i, vector in zip(batch, vectors):
                out[i] = vector
    matrix = _normalise(out)  # type: ignore[arg-type]
    if stem is not None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.save(stem.with_suffix(".npy"), matrix)
        stem.with_suffix(".json").write_text(json.dumps({"seconds": seconds}), encoding="utf-8")
    return matrix, seconds


def embed_queries(
    queries: list[str], model: str, embed_fn: EmbedFn, cache_dir: Path | None
) -> np.ndarray:
    stem = None
    if cache_dir is not None:
        stem = cache_dir / f"queries-{_cache_key(model, queries)}.npy"
        if stem.exists():
            return np.load(stem)
    vectors: list[list[float]] = []
    for start in range(0, len(queries), EMBED_BATCH):
        vectors.extend(embed_fn(queries[start : start + EMBED_BATCH], model, kind="query"))
    matrix = _normalise(vectors)
    if stem is not None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.save(stem, matrix)
    return matrix


def load_dataset(
    data_dir: Path, max_pages: int | None, split: str = "dev"
) -> tuple[dict[str, str], list[dict], set[str]]:
    """Corpus pages by slug (sorted), the questions of `split` ("dev" or
    "train"), and, for dev, the ids of the parity quality test's queries
    (queries.json); the train split has none."""
    if split not in ("dev", "train"):
        raise ValueError(f"unknown split {split!r}; expected 'dev' or 'train'")
    paths = sorted((data_dir / "corpus").glob("*.md"))
    if max_pages is not None:
        paths = paths[:max_pages]
    pages = {path.stem: path.read_text(encoding="utf-8") for path in paths}
    split_file = data_dir / "raw" / f"{split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found (download it from the pinned dataset)")
    questions = json.loads(split_file.read_text(encoding="utf-8"))
    subset: set[str] = set()
    if split == "dev":
        queries = json.loads((data_dir / "queries.json").read_text(encoding="utf-8"))
        subset = {q["query_id"] for q in queries}
    return pages, questions, subset


@dataclass
class Result:
    selection: Selection
    n_chunks: int
    means: dict[str, dict[str, float]]
    diffs: dict[str, dict[str, tuple[float, float, float]]]  # vs base, per metric
    subset_n: int
    subset_means: dict[str, dict[str, float]]
    subset_deltas: dict[str, dict[str, float]]  # vs base, page-level guard metrics
    embed_seconds: dict[str, float]
    breadcrumb_stats: dict[str, tuple[float, int, float]]  # mean chars, max chars, share
    decision: bool
    reasons: list[str]
    guard_all: bool  # the guard covers every scored question (a fixed tolerance)
    labelling: str


def run_experiment(
    pages: dict[str, str],
    dev: list[dict],
    subset_ids: set[str],
    *,
    model: str,
    embed_fn: EmbedFn,
    cache_dir: Path | None,
    resamples: int = 10_000,
    strict: bool = True,
    labelling: str = "strict",
    max_excluded: float = 0.15,
    guard_tolerance: float | None = None,
) -> Result:
    chunks = build_chunks(pages)
    selection = select_questions(dev, pages, chunks, labelling)
    if strict:
        check_exclusions(selection, max_excluded)
    questions = selection.questions
    if not questions:
        raise RuntimeError("no eligible questions")

    page_names = list(dict.fromkeys(c.page for c in chunks))
    page_index = {name: i for i, name in enumerate(page_names)}
    starts = [i for i, c in enumerate(chunks) if i == 0 or chunks[i - 1].page != c.page]
    bounds = np.array([*starts, len(chunks)])
    query_vectors = embed_queries([q.query for q in questions], model, embed_fn, cache_dir)

    rows: dict[str, dict[str, np.ndarray]] = {}
    embed_seconds: dict[str, float] = {}
    for variant in VARIANTS:
        vectors, embed_seconds[variant] = embed_variant(chunks, variant, model, embed_fn, cache_dir)
        scores = query_vectors @ vectors.T
        per_query = [
            query_metrics(scores[i], bounds, page_index[q.page], sorted(q.gold))
            for i, q in enumerate(questions)
        ]
        rows[variant] = {m: np.array([r[m] for r in per_query]) for m in METRICS}

    clusters = [q.page for q in questions]
    if guard_tolerance is not None:
        in_subset = np.ones(len(questions), dtype=bool)
    else:
        in_subset = np.array([q.query_id in subset_ids for q in questions])
    n_subset = int(in_subset.sum())
    means = {v: {m: float(rows[v][m].mean()) for m in METRICS} for v in VARIANTS}
    diffs: dict[str, dict[str, tuple[float, float, float]]] = {}
    subset_means: dict[str, dict[str, float]] = {}
    subset_deltas: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        subset_means[variant] = {
            m: float(rows[variant][m][in_subset].mean()) if n_subset else float("nan")
            for m in PAGE_GUARD_METRICS
        }
        if variant == "base":
            continue
        delta = {m: rows[variant][m] - rows["base"][m] for m in METRICS}
        diffs[variant] = {m: cluster_bootstrap(delta[m], clusters, resamples) for m in METRICS}
        subset_deltas[variant] = {
            m: float(delta[m][in_subset].mean()) if n_subset else float("nan")
            for m in PAGE_GUARD_METRICS
        }

    breadcrumb_stats: dict[str, tuple[float, int, float]] = {}
    for variant in ("full", "last"):
        extra = [len(variant_title(variant, c.title, list(c.path))) - len(c.title) for c in chunks]
        with_path = [n for n in extra if n > 0]
        breadcrumb_stats[variant] = (
            float(np.mean(with_path)) if with_path else 0.0,
            max(extra),
            len(with_path) / len(chunks),
        )

    if n_subset == 0 and not strict:
        # A --max-pages smoke run: most quality-test pages are missing.
        decision, reasons = False, ["no quality-test queries in this smoke run"]
    else:
        decision, reasons = decide(
            diffs["full"]["chunk_mrr"],
            subset_deltas["full"],
            n_subset,
            guard_tolerance,
            "scored questions" if guard_tolerance is not None else "quality-test queries",
        )
    return Result(
        selection=selection,
        n_chunks=len(chunks),
        means=means,
        diffs=diffs,
        subset_n=n_subset,
        subset_means=subset_means,
        subset_deltas=subset_deltas,
        embed_seconds=embed_seconds,
        breadcrumb_stats=breadcrumb_stats,
        decision=decision,
        reasons=reasons,
        guard_all=guard_tolerance is not None,
        labelling=labelling,
    )


def format_report(result: Result) -> str:
    s = result.selection
    lines = [
        "# Heading breadcrumbs experiment: results",
        "",
        f"- Questions scored: {len(s.questions)} (eligible {s.eligible}; excluded: "
        f"{s.page_missing} page missing, {s.unmatched} evidence not located, "
        f"{s.ambiguous} ambiguous; "
        f"not counted as eligible: {s.unanswerable} unanswerable, {s.no_evidence} without evidence)",
        f"- Evidence items not found on their page: {s.items_missing}",
        f"- Labelling: {result.labelling}; evidence items ambiguous: {s.items_ambiguous}; "
        f"questions excluded as ambiguous: {s.ambiguous}",
        f"- Chunks embedded per variant: {result.n_chunks}",
        f"- Quality-test queries among the scored ones: {result.subset_n}",
        "",
        "## All scored questions (means)",
        "",
        "| Metric | base | full | last | full − base (95% CI) | last − base (95% CI) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in METRICS:
        cells = [f"{result.means[v][m]:.4f}" for v in VARIANTS]
        for variant in ("full", "last"):
            mean, lo, hi = result.diffs[variant][m]
            cells.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    guard_heading = (
        f"## Guard set (all {result.subset_n} scored questions), page level"
        if result.guard_all
        else f"## Quality-test subset ({result.subset_n} queries), page level"
    )
    lines += [
        "",
        guard_heading,
        "",
        "| Metric | base | full | last | full − base | last − base |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in PAGE_GUARD_METRICS:
        cells = [f"{result.subset_means[v][m]:.4f}" for v in VARIANTS]
        cells += [f"{result.subset_deltas[v][m]:+.4f}" for v in ("full", "last")]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Cost",
        "",
        "| Variant | Embed seconds | Breadcrumb chars (mean / max, chunks with a path) | Chunks with a path |",
        "| --- | --- | --- | --- |",
        f"| base | {result.embed_seconds['base']:.1f} | - | - |",
    ]
    for variant in ("full", "last"):
        mean_chars, max_chars, share = result.breadcrumb_stats[variant]
        lines.append(
            f"| {variant} | {result.embed_seconds[variant]:.1f} | "
            f"{mean_chars:.0f} / {max_chars} | {share:.0%} |"
        )
    lines += ["", f"## Decision: {'YES' if result.decision else 'NO'} (`full` vs `base`)", ""]
    lines += [f"- {reason}" for reason in result.reasons]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Heading breadcrumbs embedding experiment")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Report path (default: in data-dir)")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--max-pages", type=int, default=None, help="Smoke run on N pages")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--labelling", choices=["strict", "all"], default="strict")
    parser.add_argument("--max-excluded", type=float, default=0.15)
    args = parser.parse_args()

    pages, questions, subset_ids = load_dataset(args.data_dir, args.max_pages, args.split)
    result = run_experiment(
        pages,
        questions,
        subset_ids,
        model=args.model,
        embed_fn=embed,
        cache_dir=args.data_dir / "breadcrumbs-cache",
        resamples=args.resamples,
        strict=args.max_pages is None,
        labelling=args.labelling,
        max_excluded=args.max_excluded,
        guard_tolerance=0.01 if args.split == "train" else None,
    )
    report = format_report(result)
    out = args.out or args.data_dir / (
        f"breadcrumbs-{args.split}-{args.labelling}.md"
        if args.max_pages is None
        else "breadcrumbs-smoke.md"
    )
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
