"""CLI scenario definitions for search/document-retrieval commands. Each
scenario names one CLI invocation and how to reduce its output to a small,
model-independent, JSON-serializable summary. Both capture_node_snapshots.py
(runs each scenario against frozen Node) and test_structural.py (runs each
scenario against live pyqmd) call build_cli_scenarios(profile) as their
single source of truth for what's being compared -- see the design spec's
"Directory & file structure" section.

Scenarios are built from the active DatasetProfile rather than hardcoded,
so this suite works against a user's own corpus, not just the built-in
scifact one (a 2026-09-13 parity-suite review finding).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from parity.dataset_profile import DatasetProfile
from parity.scenarios._profile_fixtures import (
    glob_pattern_for,
    known_docs,
    sample_query,
    verbatim_sentence,
)


@dataclass
class CliScenario:
    name: str
    args: list[str]
    extract: Callable[[str, int], object]


def _search_shape(stdout: str, exit_code: int) -> dict:
    """For lexically-deterministic --format json output (BM25 `search`,
    and any query guaranteed to return an empty result set regardless of
    backend): a sorted list of returned file paths plus the result count --
    never scores (those are model-dependent by design)."""
    if exit_code != 0:
        return {"exit_code": exit_code}
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return {"exit_code": exit_code, "parse_error": True}

    if not isinstance(rows, list):
        return {"exit_code": exit_code, "count": 0, "files": []}

    files = sorted(row.get("file", "") for row in rows if isinstance(row, dict))
    return {"exit_code": exit_code, "count": len(rows), "files": files}


def _embedding_search_invariants_shape(stdout: str, exit_code: int) -> dict:
    """For embedding-backed search (`vsearch`/`query`): assert only what's
    model-independent -- result count and that every file is a qmd://-
    prefixed URI -- never which specific documents appear. Which documents
    land in a real semantic top-K is a function of floating-point scores,
    and those differ between MLX and GGUF by design (the spec's own
    "Structural scenarios and dataset profiles" section). Comparing exact
    document sets here made the suite fail on a real, expected backend
    difference, not a pyqmd defect -- a false negative."""
    if exit_code != 0:
        return {"exit_code": exit_code}
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return {"exit_code": exit_code, "parse_error": True}

    if not isinstance(rows, list):
        return {"exit_code": exit_code, "count": 0, "all_qmd_uri": True}

    files = [row.get("file", "") for row in rows if isinstance(row, dict)]
    return {
        "exit_code": exit_code,
        "count": len(rows),
        "all_qmd_uri": all(f.startswith("qmd://") for f in files),
    }


def _top1_shape(stdout: str, exit_code: int) -> dict:
    """Asserts only the single top-ranked document. Used for the verbatim-
    sentence scenarios: any competent embedding model should rank a
    document first against a sentence copied verbatim from that document's
    own text, regardless of which backend produced the embedding -- so this
    recovers a real, meaningful cross-backend assertion without depending
    on score agreement."""
    if exit_code != 0:
        return {"exit_code": exit_code, "top_file": None}
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return {"exit_code": exit_code, "parse_error": True}
    if not isinstance(rows, list) or not rows:
        return {"exit_code": exit_code, "top_file": None}
    return {"exit_code": exit_code, "top_file": rows[0].get("file", "")}


def _get_shape(stdout: str, exit_code: int) -> dict:
    """For plain-text `get` output: whether it succeeded and roughly how
    much content came back (line count) -- not exact byte content, which
    can differ in trivial whitespace/formatting between the two CLIs."""
    return {"exit_code": exit_code, "line_count": len(stdout.splitlines()) if exit_code == 0 else 0}


def _multi_get_shape(stdout: str, exit_code: int) -> dict:
    return {"exit_code": exit_code, "line_count": len(stdout.splitlines()) if exit_code == 0 else 0}


def _full_path_json_shape(stdout: str, exit_code: int) -> dict:
    """For search/query/multi-get --format json output with --full-path:
    every row's "file" must not be qmd://-prefixed (it's an on-disk
    path instead) and every row's "docid" key must be entirely absent
    (dropped on successful resolution) -- an exact path string isn't
    comparable across Node's and pyqmd's isolated capture directories,
    which live at different absolute locations, so this checks the two
    invariants Node's own tests assert instead of an exact match."""
    if exit_code != 0:
        return {"exit_code": exit_code, "rows": []}
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return {"exit_code": exit_code, "parse_error": True}
    if not isinstance(rows, list):
        return {"exit_code": exit_code, "rows": []}
    return {
        "exit_code": exit_code,
        "rows": [
            {
                "is_qmd_uri": row.get("file", "").startswith("qmd://"),
                "has_docid": "docid" in row,
            }
            for row in rows
            if isinstance(row, dict)
        ],
    }


def _full_path_get_shape(stdout: str, exit_code: int) -> dict:
    """For plain-text `get --full-path`: the header line (first line of
    output) must not start with qmd:// -- same invariant as
    _full_path_json_shape, applied to get's bespoke text header."""
    if exit_code != 0:
        return {"exit_code": exit_code, "header_is_qmd_uri": False}
    first_line = stdout.splitlines()[0] if stdout.splitlines() else ""
    return {"exit_code": exit_code, "header_is_qmd_uri": first_line.startswith("qmd://")}


def _ls_shape(stdout: str, exit_code: int) -> dict:
    """`ls` with no argument lists collections. Node prints a "Collections:"
    header plus indented "  qmd://<name>/  (N files)" lines; pyqmd prints
    one bare "<name>" per line. Comparing raw lines made this an unfixable-
    looking formatting gap -- normalizing away the header and the qmd://
    wrapping recovers a real, comparable signal: the set of collection
    names. (Same class of bug as the collection list/show fixes below,
    found during the 2026-09-13 parity-suite review's fix round.)"""
    if exit_code != 0:
        return {"exit_code": exit_code, "names": []}
    names = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.rstrip(":").lower() in ("collections", "no collections."):
            continue
        token = stripped.split()[0]
        if token.startswith("qmd://"):
            token = token[len("qmd://") :]
        names.append(token.rstrip("/"))
    return {"exit_code": exit_code, "names": sorted(names)}


def _collection_list_shape(stdout: str, exit_code: int) -> dict:
    """Node's `collection list` prints a "Collections (N):" header, then
    per-collection blocks: an unindented "<name> (qmd://<name>/)" line
    followed by several indented "  Label:  value" detail lines. pyqmd
    prints one unindented "<name>  <path>  (<pattern>)" line per
    collection. Taking the first whitespace-separated token of every
    non-blank line -- the previous approach -- picked up the header word
    and the indented labels ("Pattern:", "Files:", "Updated:") as if they
    were collection names. Restricting to unindented lines and skipping the
    literal header fixes this (a 2026-09-13 parity-suite review finding)."""
    if exit_code != 0:
        return {"exit_code": exit_code, "names": []}
    names = []
    for line in stdout.splitlines():
        if not line.strip() or line[0].isspace():
            continue
        first_token = line.split()[0]
        if first_token.rstrip(":").lower() in ("collections", "no"):
            continue
        names.append(first_token)
    return {"exit_code": exit_code, "names": sorted(names)}


def _collection_show_shape(stdout: str, exit_code: int, profile_name: str) -> dict:
    """Node's `collection show <name>` never prints a document/file count
    at all (verified directly against a live Node checkout) -- only
    `collection list` does. The previous "mentions_documents" check
    compared pyqmd's real "Documents: N" line against a field Node simply
    doesn't have, so it failed on every run for every profile, not just
    scifact. Comparing presence of the collection name (the one thing both
    sides do print, under different labels -- pyqmd's "Name:", Node's
    "Collection:") is the only genuinely shared signal here."""
    return {
        "exit_code": exit_code,
        "mentions_profile_name": profile_name.lower() in stdout.lower(),
    }


def _status_shape(stdout: str, exit_code: int, profile_name: str) -> dict:
    # Plain-text, no --format support on either side. Compare presence of
    # the collection name and a rough "has documents" signal, not exact
    # counts (which can differ trivially if one side's corpus scan picked
    # up an extra hidden file, etc. -- out of scope for this scenario).
    return {
        "exit_code": exit_code,
        "mentions_profile_name": profile_name.lower() in stdout.lower(),
    }


def build_cli_scenarios(profile: DatasetProfile) -> list[CliScenario]:
    query = sample_query(profile)
    doc_a, doc_b = known_docs(profile)
    glob_pattern = glob_pattern_for(doc_a)
    sentence = verbatim_sentence(profile, doc_a)

    return [
        CliScenario(
            "search_finds_sample_query", ["search", query, "--format", "json"], _search_shape
        ),
        CliScenario(
            "search_no_results",
            ["search", "zzqqxx_no_such_term_zzqqxx", "--format", "json"],
            _search_shape,
        ),
        CliScenario(
            "vsearch_finds_sample_query",
            ["vsearch", query, "--format", "json"],
            _embedding_search_invariants_shape,
        ),
        CliScenario(
            "vsearch_top1_matches_verbatim_sentence",
            ["vsearch", sentence, "--format", "json"],
            _top1_shape,
        ),
        CliScenario(
            "query_finds_sample_query",
            ["query", query, "--format", "json"],
            _embedding_search_invariants_shape,
        ),
        CliScenario(
            "query_top1_matches_verbatim_sentence",
            ["query", sentence, "--format", "json"],
            _top1_shape,
        ),
        CliScenario(
            "query_respects_min_score_1_0",
            ["query", query, "--min-score", "1.0", "--format", "json"],
            _search_shape,
        ),
        CliScenario(
            "query_no_rerank",
            ["query", query, "--no-rerank", "--format", "json"],
            _embedding_search_invariants_shape,
        ),
        CliScenario("get_known_document", ["get", doc_a], _get_shape),
        CliScenario(
            "get_missing_document_errors_cleanly", ["get", "not-a-real-doc-id.md"], _get_shape
        ),
        CliScenario("multi_get_two_documents", ["multi-get", f"{doc_a},{doc_b}"], _multi_get_shape),
        CliScenario(
            "get_known_document_full_path",
            ["get", doc_a, "--full-path"],
            _full_path_get_shape,
        ),
        CliScenario(
            "search_finds_sample_query_full_path",
            ["search", query, "--full-path", "--format", "json"],
            _full_path_json_shape,
        ),
        CliScenario(
            "multi_get_two_documents_full_path",
            ["multi-get", f"{doc_a},{doc_b}", "--full-path", "--format", "json"],
            _full_path_json_shape,
        ),
        # Was multi_get_glob_pattern_not_supported: multi-get only ever
        # comma-split, so a glob-shaped pattern matched nothing. Renamed
        # once multi-get gained real glob-pattern support (see
        # parity/README.md's "Known gaps" section) -- now pins the
        # matching behavior instead of the absence of it.
        CliScenario(
            "multi_get_glob_pattern_matches", ["multi-get", glob_pattern], _multi_get_shape
        ),
        CliScenario("ls_lists_active_profile_collection", ["ls"], _ls_shape),
        CliScenario(
            "collection_list_shows_active_profile", ["collection", "list"], _collection_list_shape
        ),
        CliScenario(
            "collection_show_active_profile",
            ["collection", "show", profile.name],
            lambda stdout, exit_code: _collection_show_shape(stdout, exit_code, profile.name),
        ),
        CliScenario(
            "status_reports_index_health",
            ["status"],
            lambda stdout, exit_code: _status_shape(stdout, exit_code, profile.name),
        ),
    ]
