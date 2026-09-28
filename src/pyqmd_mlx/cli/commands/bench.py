"""`pyqmd bench` command."""

import json

import typer

from pyqmd_mlx.bench._fixture import load_fixture
from pyqmd_mlx.bench.runner import (
    BACKENDS,
    SAMPLED_BACKENDS,
    all_zero,
    check_collection_ready,
    result_to_dict,
    run_benchmark,
)
from pyqmd_mlx.cli._db import get_store
from pyqmd_mlx.cli._errors import run_or_exit

app = typer.Typer(help="Run search-quality benchmarks against a fixture file.")


@app.callback()
def _bench_callback() -> None:
    """Run search-quality benchmarks against a fixture file.

    A no-op callback: without it, Typer/Click collapses a Typer() app that
    has only one @app.command() registered into a single top-level command,
    which breaks `runner.invoke(app, ["bench", ...])`-style invocation (the
    literal "bench" argument would otherwise be consumed as a positional
    argument rather than treated as a subcommand name). This module is
    expected to stay single-command, so this callback is not expected to be
    removed later.
    """


def _format_single_table(result) -> str:
    lines = []
    header = f"{'Query':<25} {'Backend':<8} {'Recall@k':>9} {'MRR':>6} {'nDCG@k':>7} {'ms':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for qr in result.results:
        for name in BACKENDS:
            score = qr.backends[name]
            lines.append(
                f"{qr.id:<25} {name:<8} {score.recall_at_k:>9.2f} {score.mrr:>6.2f} "
                f"{score.ndcg_at_k:>7.2f} {score.latency_ms:>7.0f}ms"
            )
        lines.append("")

    lines.append("Summary:")
    lines.append("-" * 70)
    for name in BACKENDS:
        if name not in result.summary:
            continue
        s = result.summary[name]
        lines.append(
            f"  {name:<8} Recall@k={s['avg_recall_at_k']:.3f} MRR={s['avg_mrr']:.3f} "
            f"nDCG@k={s['avg_ndcg_at_k']:.3f} Avg={s['avg_latency_ms']:.0f}ms"
        )
    return "\n".join(lines)


_METRIC_FIELDS = ("recall_at_k", "mrr", "ndcg_at_k")
_CELL = 14  # "0.80 0.60–1.00"


def _range_cell(draws, metric: str) -> str:
    values = [getattr(d, metric) for d in draws]
    mean = sum(values) / len(values)
    return f"{mean:.2f} {min(values):.2f}–{max(values):.2f}"


def _format_sampled_table(result) -> str:
    lines = [
        f"Samples: {result.samples} (hybrid/full: mean min–max; ms from sample 0)",
        "",
    ]
    # 'ms' is 9 wide to match the rows' "{:>7.0f}ms" (the single-sample
    # table uses 8 and ends one column short; left as is there).
    header = (
        f"{'Query':<25} {'Backend':<8} {'Recall@k':<{_CELL}} {'MRR':<{_CELL}} "
        f"{'nDCG@k':<{_CELL}} {'ms':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for qr in result.results:
        for name in BACKENDS:
            score = qr.backends[name]
            if name in SAMPLED_BACKENDS:
                cells = [_range_cell(qr.samples[name], m) for m in _METRIC_FIELDS]
            else:
                cells = [f"{getattr(score, m):.2f}" for m in _METRIC_FIELDS]
            lines.append(
                f"{qr.id:<25} {name:<8} {cells[0]:<{_CELL}} {cells[1]:<{_CELL}} "
                f"{cells[2]:<{_CELL}} {score.latency_ms:>7.0f}ms"
            )
        lines.append("")

    lines.append("Summary:")
    lines.append("-" * 70)
    for name in BACKENDS:
        if name not in result.summary:
            continue
        s = result.summary[name]
        parts = []
        for label, metric in (("Recall@k", "recall_at_k"), ("MRR", "mrr"), ("nDCG@k", "ndcg_at_k")):
            part = f"{label}={s[f'avg_{metric}']:.3f}"
            if f"min_{metric}" in s:
                part += f" ({s[f'min_{metric}']:.3f}–{s[f'max_{metric}']:.3f})"
            parts.append(part)
        lines.append(f"  {name:<8} {' '.join(parts)} Avg={s['avg_latency_ms']:.0f}ms")
    return "\n".join(lines)


def _format_table(result) -> str:
    if result.samples > 1:
        return _format_sampled_table(result)
    return _format_single_table(result)


def _check_samples(samples: int) -> None:
    if samples < 1:
        raise ValueError("--samples must be at least 1")


@app.command("bench")
def bench(
    fixture_path: str = typer.Argument(..., metavar="FIXTURE"),
    collection: str = typer.Option(None, "-c", "--collection"),
    json_output: bool = typer.Option(False, "--json"),
    samples: int = typer.Option(
        1,
        "--samples",
        metavar="N",
        help="Run hybrid/full N times with different expansion seeds; report mean and min–max.",
    ),
) -> None:
    """Run search-quality benchmarks against a fixture of queries and
    expected files, across all 4 retrieval backends (bm25/vector/hybrid/
    full-reranked)."""
    run_or_exit(lambda: _check_samples(samples))
    fixture = run_or_exit(lambda: load_fixture(fixture_path))
    target_collection = collection or fixture.collection

    store = get_store()
    run_or_exit(lambda: check_collection_ready(store, target_collection))

    def on_sample(i: int, n: int) -> None:
        typer.echo(f"bench: sample {i}/{n}…", err=True)

    result = run_or_exit(
        lambda: run_benchmark(
            store, fixture, target_collection, samples=samples, on_sample=on_sample
        )
    )
    result.fixture = fixture_path

    if all_zero(result.summary):
        hint = f"pyqmd ls {target_collection}" if target_collection else "pyqmd ls"
        typer.echo(
            "All benchmark scores were 0.00 — the collection is likely unindexed "
            f"or the fixture's expected files are missing. Check with '{hint}'.",
            err=True,
        )

    if json_output:
        typer.echo(json.dumps(result_to_dict(result), indent=2))
        return

    typer.echo(_format_table(result))
