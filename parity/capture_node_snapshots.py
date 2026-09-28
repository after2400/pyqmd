"""Manual, on-demand script that captures golden output from a frozen
Node qmd checkout for the active dataset profile. NEVER run automatically
-- always a deliberate, human-initiated step. See the design spec's
"Capture script" section.

Usage: uv run python -m parity.capture_node_snapshots \\
    --qmd-repo-root <path-to-qmd-checkout> [--dataset-config path/to/profile.yaml] \\
    [--phase all|structural|cli-flow|mcp|quality] [--quality-runs N]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from parity._mcp_client import capture_mcp_scenarios_async
from parity._node_cli import get_node_commit, run_node_cli
from parity._quality_baseline import compute_metrics, summarize_runs
from parity._queries import load_queries
from parity._text_normalize import replace_paths
from parity.dataset_profile import DatasetProfile, load_dataset_profile, resolve_active_profile_path
from parity.scenarios.cli_flow_scenarios import build_cli_flow_scenarios
from parity.scenarios.cli_scenarios import build_cli_scenarios
from parity.scenarios.mcp_scenarios import build_mcp_scenarios


def _fresh_isolated_index(output_dir: Path, phase_name: str) -> Path:
    """A brand-new, empty index.sqlite + Node collection registry dir for
    one capture phase, nested under its own subdirectory (rather than
    directly in output_dir) so its derived QMD_CONFIG_DIR
    (index_path.parent / "config") is distinct from every other phase's --
    otherwise their Node collection registries collide even though the
    SQLite DBs themselves have different filenames. See run_node_cli in
    _node_cli.py.

    Removes any leftover directory from a previous capture run first: a
    capture is meant to be safely re-runnable, but `collection add` against
    a leftover index.sqlite that already has this profile's collection
    registered fails with "Collection '<name>' already exists" -- hit for
    real when re-running the capture twice in a row to pick up a code fix,
    without ever building or shipping a genuinely broken snapshot."""
    index_path = output_dir / f"_{phase_name}_capture" / "index.sqlite"
    shutil.rmtree(index_path.parent, ignore_errors=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return index_path


def capture_structural_snapshots(
    profile: DatasetProfile, qmd_repo_root: Path, output_dir: Path
) -> None:
    index_path = _fresh_isolated_index(output_dir, "structural")

    add_result = run_node_cli(
        qmd_repo_root,
        ["collection", "add", str(profile.corpus_dir), "--name", profile.name],
        index_path,
    )
    if add_result.exit_code != 0:
        raise RuntimeError(f"Node `collection add` failed: {add_result.stderr}")

    embed_result = run_node_cli(qmd_repo_root, ["embed"], index_path)
    if embed_result.exit_code != 0:
        raise RuntimeError(f"Node `embed` failed: {embed_result.stderr}")

    # Wipe stale output from a previous capture under a different scenario
    # set -- otherwise a renamed/removed scenario's old file lingers
    # forever, silently unused by any current test.
    cli_dir = output_dir / "cli"
    if cli_dir.is_dir():
        for stale in cli_dir.glob("*.json"):
            stale.unlink()
    cli_dir.mkdir(parents=True, exist_ok=True)

    cli_raw_dir = output_dir / "cli_raw"
    if cli_raw_dir.is_dir():
        for stale in cli_raw_dir.glob("*.json"):
            stale.unlink()
    cli_raw_dir.mkdir(parents=True, exist_ok=True)

    for scenario in build_cli_scenarios(profile):
        result = run_node_cli(qmd_repo_root, scenario.args, index_path)
        extracted = scenario.extract(result.stdout, result.exit_code)
        (cli_dir / f"{scenario.name}.json").write_text(json.dumps(extracted, indent=2))
        # Raw output alongside the extracted summary -- a future
        # extract-function fix (like the ones landed in this fix round) can
        # reprocess these directly instead of forcing a full multi-hour
        # re-capture just to pick up the new extraction logic.
        (cli_raw_dir / f"{scenario.name}.json").write_text(
            json.dumps(
                {
                    "args": scenario.args,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                },
                indent=2,
            )
        )


def capture_cli_flow_snapshots(qmd_repo_root: Path, output_dir: Path) -> None:
    """One fresh, isolated Node index per flow (mirroring
    capture_structural_snapshots's per-phase isolation), running every
    step of that flow in order. Writes two files per flow:
    node_ref/<profile>/cli_flow/<flow>.json ({step_name: extracted}) and
    node_ref/<profile>/cli_flow_raw/<flow>.json (stdout/stderr/exit code
    per step, for test_cli_flow_step_text_matches_node).

    Run-specific paths are replaced with placeholders *before* writing --
    <CORPUS_n> (the flow's temp corpora), <INDEX_DIR>, and <CWD> -- so the
    committed raw files never contain a real, personal path. <CWD> is
    $PWD (falling back to os.getcwd()) because that's what Node's getPwd()
    reads (src/store.ts): the capture's shell cwd, inherited through
    run_node_cli's env, not the Node checkout run_node_cli launches in.
    """
    node_cwd = os.environ.get("PWD") or os.getcwd()
    flow_dir = output_dir / "cli_flow"
    flow_raw_dir = output_dir / "cli_flow_raw"
    for directory in (flow_dir, flow_raw_dir):
        if directory.is_dir():
            for stale in directory.glob("*.json"):
                stale.unlink()
        directory.mkdir(parents=True, exist_ok=True)

    for scenario in build_cli_flow_scenarios():
        index_path = _fresh_isolated_index(output_dir, f"flow_{scenario.name}")
        results: dict[str, object] = {}
        raw_steps: dict[str, dict] = {}
        for step in scenario.steps:
            if step.before:
                step.before()
            args = step.node_args if step.node_args is not None else step.args
            result = run_node_cli(qmd_repo_root, args, index_path)
            results[step.name] = step.extract(result.stdout, result.exit_code)
            raw_steps[step.name] = {
                "args": args,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
        placeholders = {
            **scenario.placeholders(),
            str(index_path.parent): "<INDEX_DIR>",
            node_cwd: "<CWD>",
        }
        scrubbed_steps = {
            name: {
                **raw,
                "args": [replace_paths(arg, placeholders) for arg in raw["args"]],
                "stdout": replace_paths(raw["stdout"], placeholders),
                "stderr": replace_paths(raw["stderr"], placeholders),
            }
            for name, raw in raw_steps.items()
        }
        (flow_dir / f"{scenario.name}.json").write_text(json.dumps(results, indent=2))
        (flow_raw_dir / f"{scenario.name}.json").write_text(
            json.dumps({"steps": scrubbed_steps}, indent=2)
        )


def capture_mcp_snapshots(profile: DatasetProfile, qmd_repo_root: Path, output_dir: Path) -> None:
    """Own isolated Node collection (own embed cycle), matching the other
    two phases -- see the comment in capture_structural_snapshots. Speaks
    the MCP protocol to a real Node `qmd mcp` server via parity._mcp_client
    rather than one-shot CLI invocations, since these are MCP tool calls,
    not CLI commands."""
    index_path = _fresh_isolated_index(output_dir, "mcp")

    add_result = run_node_cli(
        qmd_repo_root,
        ["collection", "add", str(profile.corpus_dir), "--name", profile.name],
        index_path,
    )
    if add_result.exit_code != 0:
        raise RuntimeError(f"Node `collection add` failed: {add_result.stderr}")

    embed_result = run_node_cli(qmd_repo_root, ["embed"], index_path)
    if embed_result.exit_code != 0:
        raise RuntimeError(f"Node `embed` failed: {embed_result.stderr}")

    mcp_dir = output_dir / "mcp"
    if mcp_dir.is_dir():
        for stale in mcp_dir.glob("*.json"):
            stale.unlink()
    mcp_dir.mkdir(parents=True, exist_ok=True)

    mcp_raw_dir = output_dir / "mcp_raw"
    if mcp_raw_dir.is_dir():
        for stale in mcp_raw_dir.glob("*.json"):
            stale.unlink()
    mcp_raw_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_mcp_scenarios(profile)
    extracted, raw = asyncio.run(capture_mcp_scenarios_async(qmd_repo_root, index_path, scenarios))
    for scenario in scenarios:
        (mcp_dir / f"{scenario.name}.json").write_text(
            json.dumps(extracted[scenario.name], indent=2)
        )
        (mcp_raw_dir / f"{scenario.name}.json").write_text(json.dumps(raw[scenario.name], indent=2))


def write_commit_file(qmd_repo_root: Path, output_dir: Path) -> None:
    commit = get_node_commit(qmd_repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMMIT.txt").write_text(
        f"{commit}\ncaptured_at: {datetime.now(UTC).isoformat()}\n"
    )


def _load_qrels(qrels_file: Path) -> dict[str, set[str]]:
    relevance: dict[str, set[str]] = {}
    with qrels_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relevance.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return relevance


def _doc_id_from_file(file_path: str) -> str:
    """'<collection>/<stem>.md' -> '<stem>' -- matches how scripts/
    validate_store_query.py's fixture uses bare doc_id as the filename
    stem, and how capture_structural_snapshots seeds Node's collection
    with the corpus_dir's own filenames."""
    return Path(file_path).stem


# Qrels-mode query passes per capture. Node samples its query expansions
# (temperature 0.7), so its metrics vary run to run; the baseline is the
# mean over this many passes and the test's margin comes from their spread.
# See docs/specs/2026-09-24-qrels-margin-calibration-design.md.
DEFAULT_QUALITY_RUNS = 30


def _run_query_pass(
    qmd_repo_root: Path, index_path: Path, queries: list[dict], has_qrels: bool
) -> dict[str, list[str]]:
    per_query_ranked: dict[str, list[str]] = {}
    for entry in queries:
        result = run_node_cli(
            qmd_repo_root, ["query", entry["query"], "--format", "json"], index_path
        )
        rows = json.loads(result.stdout) if result.exit_code == 0 else []
        if has_qrels:
            per_query_ranked[entry["query_id"]] = [
                _doc_id_from_file(r.get("file", "")) for r in rows
            ]
        else:
            per_query_ranked[entry["query_id"]] = [r.get("file", "") for r in rows]
    return per_query_ranked


def capture_quality_baseline(
    profile: DatasetProfile,
    qmd_repo_root: Path,
    output_dir: Path,
    runs: int = DEFAULT_QUALITY_RUNS,
) -> None:
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    index_path = _fresh_isolated_index(output_dir, "quality")
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    add_result = run_node_cli(
        qmd_repo_root,
        ["collection", "add", str(profile.corpus_dir), "--name", profile.name],
        index_path,
    )
    if add_result.exit_code != 0:
        raise RuntimeError(f"Node `collection add` failed: {add_result.stderr}")
    embed_result = run_node_cli(qmd_repo_root, ["embed"], index_path)
    if embed_result.exit_code != 0:
        raise RuntimeError(f"Node `embed` failed: {embed_result.stderr}")

    queries = load_queries(profile)
    if not profile.has_qrels:
        per_query_ranked = _run_query_pass(qmd_repo_root, index_path, queries, has_qrels=False)
        (quality_dir / "node_query_results.json").write_text(json.dumps(per_query_ranked, indent=2))
        return

    qrels = _load_qrels(profile.qrels_file)
    relevant_sets = [qrels.get(q["query_id"], set()) for q in queries]
    per_run: list[dict[str, float]] = []
    for i in range(runs):
        if i > 0:
            # Node caches query expansions in llm_cache -- without clearing
            # it, every later pass would replay pass 1's sampled expansions
            # and the calibration would measure zero spread.
            cleanup_result = run_node_cli(qmd_repo_root, ["cleanup"], index_path)
            if cleanup_result.exit_code != 0:
                raise RuntimeError(f"Node `cleanup` failed: {cleanup_result.stderr}")
        started = time.monotonic()
        per_query_ranked = _run_query_pass(qmd_repo_root, index_path, queries, has_qrels=True)
        metrics = compute_metrics([per_query_ranked[q["query_id"]] for q in queries], relevant_sets)
        per_run.append(metrics)
        summary = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        print(f"  quality pass {i + 1}/{runs} ({time.monotonic() - started:.0f}s): {summary}")

    baseline = summarize_runs(per_run, num_queries=len(queries))
    (quality_dir / "node_query_baseline.json").write_text(json.dumps(baseline, indent=2))


PHASES = ("all", "structural", "cli-flow", "mcp", "quality")


def _check_commit_matches(qmd_repo_root: Path, output_dir: Path) -> None:
    """A single-phase capture must not mix snapshots from two different
    Node commits in one profile directory -- refuse unless the checkout's
    HEAD matches the commit the existing snapshots were captured from."""
    commit_file = output_dir / "COMMIT.txt"
    if not commit_file.is_file():
        raise SystemExit(
            f"No {commit_file} to check against -- run a full capture (--phase all) first."
        )
    recorded = commit_file.read_text().splitlines()[0].strip()
    actual = get_node_commit(qmd_repo_root)
    if recorded != actual:
        raise SystemExit(
            f"Node checkout is at {actual}, but COMMIT.txt records {recorded}. Check out "
            f"{recorded} for a single-phase capture, or re-pin with --phase all."
        )


def run_capture(
    profile: DatasetProfile,
    qmd_repo_root: Path,
    output_dir: Path,
    phase: str,
    quality_runs: int = DEFAULT_QUALITY_RUNS,
) -> None:
    if phase != "all":
        _check_commit_matches(qmd_repo_root, output_dir)
    if phase in ("all", "structural"):
        print(f"Capturing structural snapshots for profile '{profile.name}'...")
        capture_structural_snapshots(profile, qmd_repo_root, output_dir)
    if phase in ("all", "cli-flow"):
        print("Capturing CLI flow snapshots...")
        capture_cli_flow_snapshots(qmd_repo_root, output_dir)
    if phase in ("all", "mcp"):
        print("Capturing MCP snapshots...")
        capture_mcp_snapshots(profile, qmd_repo_root, output_dir)
    if phase in ("all", "quality"):
        print("Capturing quality baseline...")
        capture_quality_baseline(profile, qmd_repo_root, output_dir, quality_runs)
    if phase == "all":
        write_commit_file(qmd_repo_root, output_dir)
    print(f"Done. Snapshots written to {output_dir}")


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qmd-repo-root", required=True, help="Path to the qmd (Node) repo checkout"
    )
    parser.add_argument(
        "--dataset-config", default=None, help="Path to a dataset profile YAML file"
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Capture only one phase (requires the checkout to match COMMIT.txt). Default: all.",
    )
    parser.add_argument(
        "--quality-runs",
        type=_positive_int,
        default=DEFAULT_QUALITY_RUNS,
        help=(
            "Qrels-mode query passes to average the quality baseline over (the corpus is "
            f"indexed once). Default: {DEFAULT_QUALITY_RUNS}."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    profile_path = resolve_active_profile_path(args.dataset_config)
    profile = load_dataset_profile(profile_path)
    qmd_repo_root = Path(args.qmd_repo_root).resolve()
    output_dir = Path(__file__).resolve().parent / "node_ref" / profile.name

    run_capture(profile, qmd_repo_root, output_dir, args.phase, args.quality_runs)


if __name__ == "__main__":
    main()
