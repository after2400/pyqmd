"""Manual, on-demand performance-comparison benchmark: pyqmd vs. the
reference Node qmd CLI, over the active dataset profile (SciFact by
default). Measures indexing/embedding throughput, cold-start latency,
per-query CLI latency (search/vsearch/query), and peak RSS.

This is a companion to test_structural.py/test_quality.py, which already
cover output/results *parity* -- this script covers *performance* only,
and asserts nothing. Never wired into `just test-fast`/`test-parity`:
timing numbers are environment-noisy and shouldn't gate CI. Run via
`just bench-scifact`, or directly:

Usage: uv run python -m parity.benchmark --qmd-repo-root ../qmd \
    [--dataset-config path/to/profile.yaml] [--out path/to/report.md]

Every invocation of both CLIs is timed with `/usr/bin/time -l` (see
_timing.py) rather than in-process Python timing, so wall-clock and peak
RSS come from the same measurement and both sides pay identical
subprocess/OS overhead. There is no persistent-server phase: the MCP
surface exposes only query/get/multi_get/status (no search/vsearch), so
it can't isolate all three CLI query types the same way -- see the
Cold-start section for process-startup overhead measured on its own.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from parity._node_cli import get_node_commit, node_cli_cmd, node_cli_env
from parity._pyqmd_cli import PYQMD_REPO_ROOT, pyqmd_cli_cmd, pyqmd_cli_env, pyqmd_invocation_kind
from parity._queries import load_queries
from parity._timing import TimedResult, run_timed
from parity.dataset_profile import DatasetProfile, load_dataset_profile, resolve_active_profile_path
from parity.scenarios._profile_fixtures import known_docs, verbatim_sentence

DEFAULT_OUT = Path(__file__).resolve().parent / "BENCHMARK_RESULTS.md"
DEFAULT_COLD_START_RUNS = 5
QUERY_COMMANDS = ["search", "vsearch", "query"]


def _node_run(qmd_repo_root: Path, args: list[str], index_path: Path) -> TimedResult:
    result = run_timed(node_cli_cmd(args), cwd=qmd_repo_root, env=node_cli_env(index_path))
    if result.exit_code != 0:
        raise RuntimeError(
            f"Node `qmd {' '.join(args)}` failed (exit {result.exit_code}):\n{result.stderr}"
        )
    return result


def _pyqmd_run(args: list[str], db_path: Path) -> TimedResult:
    result = run_timed(pyqmd_cli_cmd(args), cwd=PYQMD_REPO_ROOT, env=pyqmd_cli_env(db_path))
    if result.exit_code != 0:
        raise RuntimeError(
            f"pyqmd `{' '.join(args)}` failed (exit {result.exit_code}):\n{result.stderr}"
        )
    return result


def run_indexing_phase(profile: DatasetProfile, qmd_repo_root: Path, workdir: Path) -> dict:
    node_index = workdir / "node" / "index.sqlite"
    pyqmd_index = workdir / "pyqmd" / "index.sqlite"
    node_index.parent.mkdir(parents=True)
    pyqmd_index.parent.mkdir(parents=True)

    add_args = ["collection", "add", str(profile.corpus_dir), "--name", profile.name]
    node_add = _node_run(qmd_repo_root, add_args, node_index)
    node_embed = _node_run(qmd_repo_root, ["embed"], node_index)
    pyqmd_add = _pyqmd_run(add_args, pyqmd_index)
    pyqmd_embed = _pyqmd_run(["embed"], pyqmd_index)

    return {
        "node": {"add": node_add, "embed": node_embed, "index_path": node_index},
        "pyqmd": {"add": pyqmd_add, "embed": pyqmd_embed, "index_path": pyqmd_index},
    }


def run_cold_start_phase(
    profile: DatasetProfile,
    qmd_repo_root: Path,
    node_index: Path,
    pyqmd_index: Path,
    runs: int,
) -> dict:
    """Times `runs` fresh-process invocations of the same single query
    against an already-embedded index, to isolate process-launch +
    model-load latency from the bulk embedding cost measured in the
    indexing phase."""
    doc_a, _ = known_docs(profile)
    query_text = verbatim_sentence(profile, doc_a)
    query_args = ["query", query_text, "--format", "json"]

    node_runs = [_node_run(qmd_repo_root, query_args, node_index) for _ in range(runs)]
    pyqmd_runs = [_pyqmd_run(query_args, pyqmd_index) for _ in range(runs)]
    return {"node": node_runs, "pyqmd": pyqmd_runs, "query": query_text}


def run_query_latency_phase(
    profile: DatasetProfile, qmd_repo_root: Path, node_index: Path, pyqmd_index: Path
) -> dict:
    """Times every query in the active profile's query set, once per CLI
    query command, on both sides. Each call is a fresh process -- see the
    module docstring for why this is the only fair way to cover
    search/vsearch/query, not just query (which the MCP scenarios of
    test_structural.py already restrict themselves to for that reason)."""
    queries = load_queries(profile)
    results: dict[str, dict[str, list[TimedResult]]] = {
        "node": {cmd: [] for cmd in QUERY_COMMANDS},
        "pyqmd": {cmd: [] for cmd in QUERY_COMMANDS},
    }
    for entry in queries:
        for cmd in QUERY_COMMANDS:
            args = [cmd, entry["query"], "--format", "json"]
            results["node"][cmd].append(_node_run(qmd_repo_root, args, node_index))
            results["pyqmd"][cmd].append(_pyqmd_run(args, pyqmd_index))
    return results


def _stats(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    k = (len(s) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    p95 = s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "p95": p95,
        "min": s[0],
        "max": s[-1],
    }


def _sec(x: float) -> str:
    return f"{x:.3f}s"


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def render_report(
    profile: DatasetProfile,
    node_commit: str,
    indexing: dict,
    cold_start: dict,
    query_latency: dict,
    cold_start_runs: int,
) -> str:
    lines = [
        f"# pyqmd vs Node qmd -- performance benchmark ({profile.name})",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} against Node commit `{node_commit}`.",
        f"pyqmd invoked via {pyqmd_invocation_kind()}.",
        "",
        "All timings are wall-clock from `/usr/bin/time -l` around one full CLI "
        "process invocation (includes process startup) unless noted otherwise. "
        "This is a performance comparison only -- it asserts nothing; see "
        "test_structural.py/test_quality.py for output/results parity.",
        "",
        "## Indexing / embedding throughput",
        "",
        "| Step | Node qmd | pyqmd | Speedup (Node/pyqmd) |",
        "| --- | --- | --- | --- |",
    ]
    for step in ("add", "embed"):
        node_r, pyqmd_r = indexing["node"][step], indexing["pyqmd"][step]
        speedup = (
            node_r.real_seconds / pyqmd_r.real_seconds if pyqmd_r.real_seconds else float("inf")
        )
        lines.append(
            f"| `{step}` | {_sec(node_r.real_seconds)} | {_sec(pyqmd_r.real_seconds)} | {speedup:.2f}x |"
        )
    lines += [
        "",
        "| Step | Node peak RSS | pyqmd peak RSS |",
        "| --- | --- | --- |",
    ]
    for step in ("add", "embed"):
        lines.append(
            f"| `{step}` | {_mb(indexing['node'][step].peak_rss_bytes)} | {_mb(indexing['pyqmd'][step].peak_rss_bytes)} |"
        )

    node_cold = [r.real_seconds for r in cold_start["node"]]
    pyqmd_cold = [r.real_seconds for r in cold_start["pyqmd"]]
    lines += [
        "",
        "## Cold-start latency",
        "",
        f'{cold_start_runs} fresh-process runs of `query "{cold_start["query"]}"` '
        "against a pre-built index (isolates process launch + model load from bulk embedding cost).",
        "",
        "| | Node qmd | pyqmd |",
        "| --- | --- | --- |",
        f"| Median | {_sec(statistics.median(node_cold))} | {_sec(statistics.median(pyqmd_cold))} |",
        f"| Min | {_sec(min(node_cold))} | {_sec(min(pyqmd_cold))} |",
        f"| Max | {_sec(max(node_cold))} | {_sec(max(pyqmd_cold))} |",
        "",
        "## Per-query CLI latency",
        "",
        f"{len(query_latency['node']['search'])} queries from the `{profile.name}` profile, "
        "one fresh CLI process per query (includes process startup -- compare against the "
        "Cold-start section above to gauge how much of this is per-query work vs. launch overhead).",
        "",
        "| Command | Node mean | Node median | Node p95 | pyqmd mean | pyqmd median | pyqmd p95 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cmd in QUERY_COMMANDS:
        node_s = _stats([r.real_seconds for r in query_latency["node"][cmd]])
        pyqmd_s = _stats([r.real_seconds for r in query_latency["pyqmd"][cmd]])
        lines.append(
            f"| `{cmd}` | {_sec(node_s['mean'])} | {_sec(node_s['median'])} | {_sec(node_s['p95'])} "
            f"| {_sec(pyqmd_s['mean'])} | {_sec(pyqmd_s['median'])} | {_sec(pyqmd_s['p95'])} |"
        )

    node_query_peak = max(
        r.peak_rss_bytes for cmd in QUERY_COMMANDS for r in query_latency["node"][cmd]
    )
    pyqmd_query_peak = max(
        r.peak_rss_bytes for cmd in QUERY_COMMANDS for r in query_latency["pyqmd"][cmd]
    )
    lines += [
        "",
        "## Peak memory (RSS)",
        "",
        "| Phase | Node | pyqmd |",
        "| --- | --- | --- |",
        f"| Indexing (`embed`) | {_mb(indexing['node']['embed'].peak_rss_bytes)} | {_mb(indexing['pyqmd']['embed'].peak_rss_bytes)} |",
        f"| Query serving (max across search/vsearch/query) | {_mb(node_query_peak)} | {_mb(pyqmd_query_peak)} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--qmd-repo-root", required=True, help="Path to the qmd (Node) repo checkout"
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Path to a parity dataset profile YAML (default: built-in scifact)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="Where to write the Markdown report"
    )
    parser.add_argument("--cold-start-runs", type=int, default=DEFAULT_COLD_START_RUNS)
    args = parser.parse_args()

    qmd_repo_root = Path(args.qmd_repo_root).resolve()
    profile = load_dataset_profile(resolve_active_profile_path(args.dataset_config))
    node_commit = get_node_commit(qmd_repo_root)

    with tempfile.TemporaryDirectory(prefix="pyqmd-benchmark-") as tmp:
        workdir = Path(tmp)

        print(f"[1/3] Indexing {profile.corpus_dir} on both sides...")
        indexing = run_indexing_phase(profile, qmd_repo_root, workdir)
        node_index = indexing["node"]["index_path"]
        pyqmd_index = indexing["pyqmd"]["index_path"]

        print(f"[2/3] Measuring cold-start latency ({args.cold_start_runs} runs each side)...")
        cold_start = run_cold_start_phase(
            profile, qmd_repo_root, node_index, pyqmd_index, args.cold_start_runs
        )

        print("[3/3] Measuring per-query CLI latency across search/vsearch/query...")
        query_latency = run_query_latency_phase(profile, qmd_repo_root, node_index, pyqmd_index)

    report = render_report(
        profile, node_commit, indexing, cold_start, query_latency, args.cold_start_runs
    )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print()
    print(report)
    print()
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
