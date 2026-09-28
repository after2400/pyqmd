"""Quality parity tests: does pyqmd's real search quality on the active
profile's corpus stay within the qrels-mode per-metric margin (absolute IR
metrics, allowing for Node's own run-to-run spread -- see
parity/_quality_baseline.py) or the agreement-mode threshold (system-to-system comparison) of Node's
captured baseline? See the design spec's "Quality test methodology"
section for the full rationale behind the two-mode split.
"""

from __future__ import annotations

import csv

import pytest

from parity._quality_baseline import MARGIN_SIGMAS, compute_metrics, metric_margin
from parity._queries import load_queries
from parity._snapshot_io import load_quality_file
from pyqmd_mlx.bench._metrics import spearman_rank_correlation, top_k_overlap
from pyqmd_mlx.cli.commands.search import DEFAULT_SEARCH_LIMIT

OVERLAP_THRESHOLD = 0.7  # starting point -- no second real dataset exists yet to tune this against


def _load_qrels(qrels_file):
    relevance: dict[str, set[str]] = {}
    with qrels_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relevance.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return relevance


def _doc_id_from_file(display_path: str) -> str:
    from pathlib import Path

    return Path(display_path).stem


@pytest.mark.parity
def test_pyqmd_meets_qrels_mode_quality_bar(active_profile, indexed_pyqmd_store):
    if not active_profile.has_qrels:
        pytest.skip(
            f"profile '{active_profile.name}' has no qrels_file -- see agreement-mode test instead"
        )

    queries = load_queries(active_profile)
    qrels = _load_qrels(active_profile.qrels_file)

    ranked_lists = []
    relevant_sets = []
    for entry in queries:
        # Node's baseline was captured with `-n` omitted -> its real CLI
        # default (DEFAULT_SEARCH_LIMIT, 20) -- querying at a shallower
        # depth here made MRR (unlike nDCG/Recall, which apply their own
        # @10 cutoff regardless of retrieval depth) compare a 20-deep list
        # against a 10-deep one (a 2026-09-13 parity-suite review finding).
        results = indexed_pyqmd_store.query(
            entry["query"], limit=DEFAULT_SEARCH_LIMIT, collection=active_profile.name
        )
        ranked_lists.append([_doc_id_from_file(r.file) for r in results])
        relevant_sets.append(qrels.get(entry["query_id"], set()))

    pyqmd_metrics = compute_metrics(ranked_lists, relevant_sets)
    node_baseline = load_quality_file(active_profile.name, "node_query_baseline.json")
    calibration = node_baseline.get("calibration")

    failures = []
    for metric_name, pyqmd_value in pyqmd_metrics.items():
        node_value = node_baseline[metric_name]
        margin = metric_margin(node_baseline, metric_name)
        if calibration is None:
            basis = "fallback margin -- baseline has no calibration block"
        else:
            sigma = calibration["stddev"][metric_name]
            basis = (
                f"max({MARGIN_SIGMAS:g} x sigma={sigma:.4f}, 1/{calibration['num_queries']}) "
                f"over {calibration['runs']} Node runs"
            )
        line = (
            f"{metric_name}: pyqmd={pyqmd_value:.4f} node_mean={node_value:.4f} "
            f"margin={margin:.4f} [{basis}]"
        )
        print(line)
        if pyqmd_value < node_value - margin:
            failures.append(line)

    assert not failures, "pyqmd fell more than the margin below Node on:\n" + "\n".join(failures)


@pytest.mark.parity
def test_pyqmd_agrees_with_node_when_no_qrels(active_profile, indexed_pyqmd_store):
    if active_profile.has_qrels:
        pytest.skip(
            f"profile '{active_profile.name}' has qrels_file -- see qrels-mode test instead"
        )

    queries = load_queries(active_profile)
    node_results = load_quality_file(active_profile.name, "node_query_results.json")

    overlaps = []
    correlations = []
    for entry in queries:
        results = indexed_pyqmd_store.query(
            entry["query"], limit=10, collection=active_profile.name
        )
        # Compare full qmd://collection/path strings, not stems: the capture
        # script stores node_query_results.json values in that same
        # already-qmd://-prefixed form (see capture_node_snapshots.py's
        # agreement-mode branch, which stores `r.get("file", "")` verbatim,
        # unlike its qrels-mode sibling branch which stems). Stemming only
        # pyqmd's side here made the two sides' ids incomparable -- the
        # intersection was always empty, so this test could never pass for
        # any profile (a 2026-09-13 parity-suite review finding).
        pyqmd_ranked = [r.file for r in results]
        node_ranked = node_results.get(entry["query_id"], [])

        overlaps.append(top_k_overlap(pyqmd_ranked, node_ranked, k=10))
        correlation = spearman_rank_correlation(pyqmd_ranked, node_ranked)
        if correlation is not None:
            correlations.append(correlation)

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(f"Mean top-10 overlap: {mean_overlap:.3f} over {len(queries)} queries")
    if correlations:
        print(
            f"Mean Spearman correlation (over queries with >=2 shared docs): {sum(correlations) / len(correlations):.3f}"
        )

    assert mean_overlap >= OVERLAP_THRESHOLD, (
        f"pyqmd and Node agree on only {mean_overlap:.3f} of top-10 results on average "
        f"(threshold: {OVERLAP_THRESHOLD}) -- see printed per-query detail above"
    )
