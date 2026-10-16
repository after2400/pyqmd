"""Does an LLM-written context line per chunk beat heading breadcrumbs?

Two subcommands, both resumable and meant to be run by hand:
  generate --generator small|mid   write a context line for every chunk
  score                            embed base / crumbs / ctx-small / ctx-mid and
                                   apply the rule fixed in the spec
The rule and every choice here are fixed in advance in the spec
(2026-10-16-contextual-embedding-experiment-design.md). Never touches the
user's index.
"""

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from bench_breadcrumbs import (
    DEFAULT_DATA_DIR,
    METRICS,
    Chunk,
    build_chunks,
    check_exclusions,
    cluster_bootstrap,
    embed_inputs,
    embed_queries,
    load_dataset,
    query_metrics,
    select_questions,
    variant_title,
)
from bench_breadcrumbs_rerank import Logger, format_progress

from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL, embed

GENERATORS = {
    "small": "mlx-community/Qwen3-1.7B-4bit",
    "mid": "mlx-community/Qwen3-4B-4bit",
}
MAX_WINDOW_CHARS = 18_000
MAX_NEW_TOKENS = 60

_INSTRUCTION = (
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purpose of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def window_text(page: str, pos: int, length: int, max_chars: int = MAX_WINDOW_CHARS) -> str:
    """The page, or max_chars of it centred on the chunk [pos, pos + length)."""
    if len(page) <= max_chars:
        return page
    centre = pos + length // 2
    start = min(max(centre - max_chars // 2, 0), len(page) - max_chars)
    return page[start : start + max_chars]


def build_prompt(window: str, chunk: str) -> str:
    return (
        f"<document>\n{window}\n</document>\n"
        "Here is the chunk we want to situate within the whole document\n"
        f"<chunk>\n{chunk}\n</chunk>\n"
        f"{_INSTRUCTION}"
    )


def clean_context(raw: str) -> str:
    """One line: reasoning blocks (closed, or open to the end) removed, whitespace collapsed."""
    text = _THINK_BLOCK.sub("", raw)
    text = text.split("<think>", 1)[0]
    return " ".join(text.split())


def ctx_text(context: str, chunk: str) -> str:
    return f"{context}\n\n{chunk}" if context else chunk


def context_key(generator_id: str, prompt: str) -> str:
    """Covers the generator, the instruction, the page window and the chunk
    (the prompt contains all of them)."""
    return hashlib.sha256(f"{generator_id}\x00{prompt}".encode()).hexdigest()


class ContextCache:
    """key -> context line, appended to a JSONL file as it is generated. A line
    torn by a kill is dropped on load."""

    def __init__(self, path: Path | None):
        self.path = path
        self._items: dict[str, str] = {}
        if path is not None and path.exists():
            good = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._items[record["key"]] = record["context"]
                good.append(line)
            path.write_text("".join(f"{line}\n" for line in good), encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self._items.get(key)

    def put(self, key: str, context: str) -> None:
        self._items[key] = context
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"key": key, "context": context}, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def chunk_prompt(chunk: Chunk, pages: dict[str, str]) -> str:
    window = window_text(pages[chunk.page], chunk.pos, len(chunk.text))
    return build_prompt(window, chunk.text)


def generate_contexts(
    chunks: list[Chunk],
    pages: dict[str, str],
    generator_id: str,
    generate_fn: Callable[[str], str],
    cache: ContextCache,
    log: Logger,
    limit: int | None = None,
    log_every: int = 25,
) -> int:
    """Generate and cache a cleaned context for each chunk the cache lacks (the
    first `limit` chunks only, when given). Returns how many were generated
    this session."""
    targets = chunks[:limit]
    prompts = [chunk_prompt(c, pages) for c in targets]
    keys = [context_key(generator_id, p) for p in prompts]
    resumed = sum(1 for k in keys if cache.get(k) is not None)
    started = time.monotonic()
    made = 0
    for done, (prompt, key) in enumerate(zip(prompts, keys, strict=True), start=1):
        if cache.get(key) is not None:
            continue
        cache.put(key, clean_context(generate_fn(prompt)))
        made += 1
        if made == 3 or done % log_every == 0:
            elapsed = time.monotonic() - started
            log.say(format_progress(done, len(targets), elapsed, resumed, unit="chunks"))
    log.say(f"{len(targets)} chunks have a context ({made} generated now)")
    return made


def cached_contexts(
    chunks: list[Chunk], pages: dict[str, str], generator_id: str, cache: ContextCache
) -> list[str]:
    """One cleaned context per chunk, in order; every chunk must be cached."""
    contexts = [cache.get(context_key(generator_id, chunk_prompt(c, pages))) for c in chunks]
    missing = sum(1 for c in contexts if c is None)
    if missing:
        raise ValueError(
            f"{missing} of {len(chunks)} chunks have no context from {generator_id}; "
            "run `bench_contextual.py generate` for this generator first"
        )
    return contexts  # type: ignore[return-value]


def load_generator(model_id: str) -> Callable[[str], str]:
    """A greedy, thinking-off chat generator over an MLX model (loads now,
    downloads on the first run). Not covered by unit tests; the smoke run
    exercises it."""
    from mlx_lm import generate, load

    model, tokenizer = load(model_id)

    def generate_context(prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return generate(model, tokenizer, prompt=text, max_tokens=MAX_NEW_TOKENS)

    return generate_context


REGRESSION = -0.01
CONTRAST_LEVEL = 0.975  # two generator sizes are tested


def arm_inputs(
    chunks: list[Chunk], contexts: dict[str, list[str]]
) -> dict[str, tuple[list[str], list[str]]]:
    """arm name -> (titles, texts) handed to embed(kind="document"). The ctx
    arms keep the plain document title: only the text changes."""
    plain_titles = [c.title for c in chunks]
    texts = [c.text for c in chunks]
    arms = {
        "base": (plain_titles, texts),
        "crumbs": ([variant_title("full", c.title, list(c.path)) for c in chunks], texts),
    }
    for size, lines in contexts.items():
        arms[f"ctx-{size}"] = (
            plain_titles,
            [ctx_text(line, c.text) for line, c in zip(lines, chunks, strict=True)],
        )
    return arms


def contrast(
    rows: dict[str, dict[str, np.ndarray]],
    a: str,
    b: str,
    clusters: list[str],
    resamples: int,
    metrics: tuple[str, ...] = METRICS,
) -> dict[str, tuple[float, float, float]]:
    """metric -> (mean, lower, upper) of arm a minus arm b, page-clustered."""
    return {
        m: cluster_bootstrap(rows[a][m] - rows[b][m], clusters, resamples, level=CONTRAST_LEVEL)
        for m in metrics
    }


def regressions(vs_base: dict[str, tuple[float, float, float]]) -> list[str]:
    """Metrics whose point estimate against base is below -0.01."""
    return [m for m, (mean, _lo, _hi) in vs_base.items() if mean < REGRESSION]


def contextual_verdict(
    vs_crumbs: dict[str, tuple[float, float, float]],
    vs_base: dict[str, tuple[float, float, float]],
) -> tuple[int, str]:
    """The spec's rule, in order: 1, then 2, else 3."""
    if vs_crumbs["chunk_mrr"][1] > 0 and vs_crumbs["page_ndcg10"][0] >= 0:
        return 1, (
            "1: beats breadcrumbs (chunk MRR bound above zero, nDCG@10 not lower); "
            "worth a feature spec, the user decides"
        )
    if vs_base["chunk_mrr"][1] > 0:
        return 2, (
            "2: beats plain embedding but not breadcrumbs; "
            "not worth the generation cost, no feature"
        )
    return 3, "3: no gain over plain embedding that survives the interval"


def score_arms(arms, chunks, questions, query_vectors, model, embed_fn, cache_dir):
    """Per-arm per-question metrics, and each arm's embedding seconds."""
    page_names = list(dict.fromkeys(c.page for c in chunks))
    page_index = {name: i for i, name in enumerate(page_names)}
    starts = [i for i, c in enumerate(chunks) if i == 0 or chunks[i - 1].page != c.page]
    bounds = np.array([*starts, len(chunks)])
    rows, seconds = {}, {}
    for arm, (titles, texts) in arms.items():
        vectors, seconds[arm] = embed_inputs(titles, texts, arm, model, embed_fn, cache_dir)
        scores = query_vectors @ vectors.T
        per_query = [
            query_metrics(scores[i], bounds, page_index[q.page], sorted(q.gold))
            for i, q in enumerate(questions)
        ]
        rows[arm] = {m: np.array([r[m] for r in per_query]) for m in METRICS}
    return rows, seconds


def _cell(ci: tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def format_report(selection, n_chunks, rows, seconds, empties, vs_crumbs, vs_base) -> str:
    lines = [
        "# Contextual embedding: vector-stage check",
        "",
        f"- Questions scored: {len(selection.questions)} (strict labelling; "
        f"{selection.ambiguous} excluded as ambiguous); chunks: {n_chunks}",
        f"- Contrast intervals are {CONTRAST_LEVEL:.1%} (page-clustered bootstrap)",
        "",
        "| Arm | " + " | ".join(METRICS) + " | embed s |",
        "| --- |" + " --- |" * (len(METRICS) + 1),
    ]
    for arm, metrics in rows.items():
        cells = " | ".join(f"{metrics[m].mean():.4f}" for m in METRICS)
        lines.append(f"| {arm} | {cells} | {seconds[arm]:.0f} |")
    for size in empties:
        lines += [
            "",
            f"## {size}: ctx-{size} minus crumbs, and minus base (empty contexts: {empties[size]})",
            "",
            "| Metric | vs crumbs | vs base |",
            "| --- | --- | --- |",
        ]
        for m in METRICS:
            lines.append(f"| {m} | {_cell(vs_crumbs[size][m])} | {_cell(vs_base[size][m])} |")
        flagged = regressions(vs_base[size])
        if flagged:
            lines += ["", f"Regression flag against base (point estimate below -0.01): {flagged}"]
        _code, text = contextual_verdict(vs_crumbs[size], vs_base[size])
        lines += ["", f"## Verdict ({size}) {text}"]
    return "\n".join(lines) + "\n"


def score_run(
    pages: dict[str, str],
    source: list[dict],
    *,
    model: str,
    embed_fn,
    cache_dir: Path | None,
    caches: dict[str, ContextCache],
    resamples: int = 10_000,
    strict: bool = True,
) -> str:
    """Embed the arms, score every question and return the report. Fails before
    any embedding if a requested generator has not finished."""
    chunks = build_chunks(pages)
    selection = select_questions(source, pages, chunks)
    if strict:
        check_exclusions(selection, 0.15)
    questions = selection.questions
    if not questions:
        raise RuntimeError("no eligible questions")
    contexts = {
        size: cached_contexts(chunks, pages, GENERATORS[size], cache)
        for size, cache in caches.items()
    }
    arms = arm_inputs(chunks, contexts)
    query_vectors = embed_queries([q.query for q in questions], model, embed_fn, cache_dir)
    rows, seconds = score_arms(arms, chunks, questions, query_vectors, model, embed_fn, cache_dir)
    clusters = [q.page for q in questions]
    vs_crumbs = {s: contrast(rows, f"ctx-{s}", "crumbs", clusters, resamples) for s in contexts}
    vs_base = {s: contrast(rows, f"ctx-{s}", "base", clusters, resamples) for s in contexts}
    empties = {size: sum(1 for line in lines if not line) for size, lines in contexts.items()}
    return format_report(selection, len(chunks), rows, seconds, empties, vs_crumbs, vs_base)


def _paths(data_dir: Path, smoke: bool) -> Path:
    return data_dir / ("contextual-smoke" if smoke else "contextual")


def build_parser() -> argparse.ArgumentParser:
    """The common options are accepted after the subcommand (that is how the
    plan's commands are written)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    common.add_argument("--max-pages", type=int, default=None, help="Smoke run on N pages")
    parser = argparse.ArgumentParser(description="Contextual embedding experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser(
        "generate", parents=[common], help="Write a context line for every chunk"
    )
    gen.add_argument("--generator", choices=sorted(GENERATORS), required=True)
    gen.add_argument("--limit-chunks", type=int, default=None, help="First N chunks (timing)")
    score = commands.add_parser("score", parents=[common], help="Embed the arms and apply the rule")
    score.add_argument(
        "--generators", nargs="+", choices=sorted(GENERATORS), default=list(GENERATORS)
    )
    score.add_argument("--resamples", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    work = _paths(args.data_dir, args.max_pages is not None)
    work.mkdir(parents=True, exist_ok=True)
    pages, source, _subset = load_dataset(args.data_dir, args.max_pages, "train")

    if args.command == "generate":
        log = Logger(work / f"generate-{args.generator}.log")
        log.say(f"--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        chunks = build_chunks(pages)
        log.say(f"{len(chunks)} chunks over {len(pages)} pages; generator {args.generator}")
        cache = ContextCache(work / f"contexts-{args.generator}.jsonl")
        generator_id = GENERATORS[args.generator]
        generate_fn = load_generator(generator_id)
        generate_contexts(
            chunks, pages, generator_id, generate_fn, cache, log, limit=args.limit_chunks
        )
        return

    caches = {size: ContextCache(work / f"contexts-{size}.jsonl") for size in args.generators}
    report = score_run(
        pages,
        source,
        model=DEFAULT_EMBED_MODEL,
        embed_fn=embed,
        cache_dir=work / "embeddings",
        caches=caches,
        resamples=args.resamples,
        strict=args.max_pages is None,
    )
    (work / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {work / 'report.md'}")


if __name__ == "__main__":
    main()
