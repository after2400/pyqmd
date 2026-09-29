import csv
import json

import pytest

from parity.capture_node_snapshots import capture_structural_snapshots, write_commit_file
from parity.dataset_profile import DatasetProfile


def _fake_profile(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("# Doc 1\nbiomaterials content")
    queries_file = tmp_path / "queries.json"
    queries_file.write_text("[]")
    return DatasetProfile(
        name="fake", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=None
    )


def test_fresh_isolated_index_removes_leftover_state_from_a_prior_run(tmp_path):
    # Regression test: re-running the capture script twice in a row (e.g.
    # to pick up an extract-function fix) used to fail with Node's real
    # "Collection 'scifact' already exists" error, because the isolated
    # index.sqlite and its Node collection registry (config/*.yml) from the
    # first run were left in place for the second run's `collection add`
    # to collide with.
    from parity.capture_node_snapshots import _fresh_isolated_index

    output_dir = tmp_path / "output"
    stale_dir = output_dir / "_structural_capture"
    stale_dir.mkdir(parents=True)
    (stale_dir / "index.sqlite").write_text("leftover db from a prior run")
    (stale_dir / "config").mkdir()
    (stale_dir / "config" / "scifact.yml").write_text("leftover collection registry")

    index_path = _fresh_isolated_index(output_dir, "structural")

    assert index_path == stale_dir / "index.sqlite"
    assert not index_path.exists()
    assert not (stale_dir / "config").exists()
    assert stale_dir.is_dir()


def test_capture_structural_snapshots_writes_one_file_per_scenario(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        from parity._node_cli import NodeCliResult

        return NodeCliResult(exit_code=0, stdout="[]", stderr="")

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)

    profile = _fake_profile(tmp_path)
    output_dir = tmp_path / "output"

    capture_structural_snapshots(profile, tmp_path, output_dir)

    from parity.scenarios.cli_scenarios import build_cli_scenarios

    for scenario in build_cli_scenarios(profile):
        snapshot_path = output_dir / "cli" / f"{scenario.name}.json"
        assert snapshot_path.is_file(), f"missing snapshot for {scenario.name}"
        data = json.loads(snapshot_path.read_text())
        assert isinstance(data, dict)

        raw_path = output_dir / "cli_raw" / f"{scenario.name}.json"
        assert raw_path.is_file(), f"missing raw output for {scenario.name}"
        raw = json.loads(raw_path.read_text())
        assert raw["args"] == scenario.args
        assert raw["exit_code"] == 0


def test_write_commit_file_records_sha_and_date(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "a" * 40)

    output_dir = tmp_path / "output"
    write_commit_file(tmp_path, output_dir)

    content = (output_dir / "COMMIT.txt").read_text()
    assert "a" * 40 in content


def _fake_qrels_profile(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "17388232.md").write_text("# Doc\nbiomaterials content")

    queries_file = tmp_path / "queries.json"
    queries_file.write_text(
        json.dumps([{"query_id": "1", "query": "biomaterials", "candidate_docs": [], "qrels": {}}])
    )

    qrels_file = tmp_path / "qrels.tsv"
    with qrels_file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerow(["1", "17388232", "1"])

    return DatasetProfile(
        name="fakeqrels", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=qrels_file
    )


def _fake_no_qrels_profile(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("# Doc\nbiomaterials content")
    queries_file = tmp_path / "queries.yaml"
    queries_file.write_text("- biomaterials\n")

    return DatasetProfile(
        name="fakenoqrels", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=None
    )


def test_load_queries_normalizes_scifact_fixture_shape(tmp_path):
    from parity._queries import load_queries

    profile = _fake_qrels_profile(tmp_path)
    queries = load_queries(profile)

    assert queries == [{"query_id": "1", "query": "biomaterials"}]


def test_load_queries_normalizes_plain_list_shape(tmp_path):
    from parity._queries import load_queries

    profile = _fake_no_qrels_profile(tmp_path)
    queries = load_queries(profile)

    assert queries == [{"query_id": "0", "query": "biomaterials"}]


def test_load_queries_normalizes_plain_list_shape_even_as_json(tmp_path):
    from parity._queries import load_queries

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("# Doc\nbiomaterials content")
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(json.dumps(["biomaterials"]))

    profile = DatasetProfile(
        name="fakejsonlist", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=None
    )
    queries = load_queries(profile)

    assert queries == [{"query_id": "0", "query": "biomaterials"}]


def test_capture_quality_baseline_qrels_mode_writes_metrics(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module
    from parity._node_cli import NodeCliResult

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        return NodeCliResult(
            exit_code=0,
            stdout=json.dumps([{"file": "scifact/17388232.md", "docid": "#abc123"}]),
            stderr="",
        )

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)

    profile = _fake_qrels_profile(tmp_path)
    output_dir = tmp_path / "output"
    capture_module.capture_quality_baseline(profile, tmp_path, output_dir)

    baseline = json.loads((output_dir / "quality" / "node_query_baseline.json").read_text())
    assert set(baseline.keys()) >= {"mrr", "ndcg_at_10", "recall_at_10"}
    assert not (output_dir / "quality" / "node_query_results.json").exists()


def test_structural_and_quality_phases_use_distinct_config_dirs(tmp_path, monkeypatch):
    """Regression test: capture_structural_snapshots and capture_quality_baseline
    used to derive index_path files that shared the same parent directory
    (output_dir), so run_node_cli's QMD_CONFIG_DIR (index_path.parent / "config")
    was identical for both phases -- quality capture's `collection add` would
    then fail against structural capture's already-registered collection.
    This asserts the two phases' index_path.parent values differ, which
    guarantees their derived QMD_CONFIG_DIRs differ too."""
    import parity.capture_node_snapshots as capture_module
    from parity._node_cli import NodeCliResult

    seen_index_paths: list = []

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        seen_index_paths.append(index_path)
        return NodeCliResult(exit_code=0, stdout="[]", stderr="")

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)

    output_dir = tmp_path / "output"

    structural_input = tmp_path / "structural_input"
    structural_input.mkdir()
    structural_profile = _fake_profile(structural_input)
    capture_module.capture_structural_snapshots(structural_profile, tmp_path, output_dir)
    structural_index_paths = list(seen_index_paths)
    seen_index_paths.clear()

    quality_input = tmp_path / "quality_input"
    quality_input.mkdir()
    quality_profile = _fake_no_qrels_profile(quality_input)
    capture_module.capture_quality_baseline(quality_profile, tmp_path, output_dir)
    quality_index_paths = list(seen_index_paths)

    assert structural_index_paths, "structural phase never invoked run_node_cli"
    assert quality_index_paths, "quality phase never invoked run_node_cli"

    structural_parents = {p.parent for p in structural_index_paths}
    quality_parents = {p.parent for p in quality_index_paths}

    assert structural_parents.isdisjoint(quality_parents), (
        f"structural and quality phases share an index_path parent dir "
        f"(and thus the same QMD_CONFIG_DIR): {structural_parents & quality_parents}"
    )


def test_capture_quality_baseline_agreement_mode_writes_raw_results(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module
    from parity._node_cli import NodeCliResult

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        return NodeCliResult(
            exit_code=0, stdout=json.dumps([{"file": "fakenoqrels/doc1.md"}]), stderr=""
        )

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)

    profile = _fake_no_qrels_profile(tmp_path)
    output_dir = tmp_path / "output"
    capture_module.capture_quality_baseline(profile, tmp_path, output_dir)

    results = json.loads((output_dir / "quality" / "node_query_results.json").read_text())
    assert results == {"0": ["fakenoqrels/doc1.md"]}
    assert not (output_dir / "quality" / "node_query_baseline.json").exists()


def test_capture_mcp_snapshots_writes_one_file_per_scenario(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module
    from parity._node_cli import NodeCliResult
    from parity.scenarios.mcp_scenarios import build_mcp_scenarios

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        return NodeCliResult(exit_code=0, stdout="", stderr="")

    async def fake_capture_mcp_scenarios_async(qmd_repo_root, index_path, scenarios):
        extracted = {s.name: {"fake": True} for s in scenarios}
        raw = {s.name: {"content": [], "isError": False} for s in scenarios}
        return extracted, raw

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)
    monkeypatch.setattr(
        capture_module, "capture_mcp_scenarios_async", fake_capture_mcp_scenarios_async
    )

    profile = _fake_profile(tmp_path)
    output_dir = tmp_path / "output"
    capture_module.capture_mcp_snapshots(profile, tmp_path, output_dir)

    for scenario in build_mcp_scenarios(profile):
        extracted_path = output_dir / "mcp" / f"{scenario.name}.json"
        raw_path = output_dir / "mcp_raw" / f"{scenario.name}.json"
        assert extracted_path.is_file(), f"missing snapshot for {scenario.name}"
        assert raw_path.is_file(), f"missing raw output for {scenario.name}"
        assert json.loads(extracted_path.read_text()) == {"fake": True}


def test_capture_mcp_snapshots_uses_its_own_isolated_index(tmp_path, monkeypatch):
    """Regression guard, same reasoning as
    test_structural_and_quality_phases_use_distinct_config_dirs: the MCP
    phase must not share an index_path parent (and thus a QMD_CONFIG_DIR)
    with the structural or quality phases."""
    import parity.capture_node_snapshots as capture_module
    from parity._node_cli import NodeCliResult

    seen_index_paths: list = []

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        seen_index_paths.append(index_path)
        return NodeCliResult(exit_code=0, stdout="", stderr="")

    async def fake_capture_mcp_scenarios_async(qmd_repo_root, index_path, scenarios):
        seen_index_paths.append(index_path)
        return {s.name: {} for s in scenarios}, {s.name: {} for s in scenarios}

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)
    monkeypatch.setattr(
        capture_module, "capture_mcp_scenarios_async", fake_capture_mcp_scenarios_async
    )

    profile = _fake_profile(tmp_path)
    output_dir = tmp_path / "output"
    capture_module.capture_mcp_snapshots(profile, tmp_path, output_dir)

    mcp_parents = {p.parent for p in seen_index_paths}
    assert len(mcp_parents) == 1
    assert mcp_parents != {output_dir / "_structural_capture"}
    assert mcp_parents != {output_dir / "_quality_capture"}


def _fake_flow_node_cli(qmd_repo_root, args, index_path, extra_env=None):
    from parity._node_cli import NodeCliResult

    return NodeCliResult(exit_code=0, stdout=f"out {' '.join(args)}\n", stderr="warn\n")


_PHASE_FUNCTIONS = (
    "capture_structural_snapshots",
    "capture_cli_flow_snapshots",
    "capture_mcp_snapshots",
    "capture_quality_baseline",
)


def test_capture_cli_flow_snapshots_writes_raw_step_output(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module
    from parity.scenarios.cli_flow_scenarios import build_cli_flow_scenarios

    monkeypatch.setattr(capture_module, "run_node_cli", _fake_flow_node_cli)
    output_dir = tmp_path / "output"

    capture_module.capture_cli_flow_snapshots(tmp_path, output_dir)

    for scenario in build_cli_flow_scenarios():
        raw = json.loads((output_dir / "cli_flow_raw" / f"{scenario.name}.json").read_text())
        assert list(raw) == ["steps"]
        assert list(raw["steps"]) == [step.name for step in scenario.steps]
        first = raw["steps"][scenario.steps[0].name]
        assert first["stderr"] == "warn\n"
        assert first["exit_code"] == 0
        assert first["stdout"].startswith("out ")


def _echo_paths_node_cli(qmd_repo_root, args, index_path, extra_env=None):
    """Echoes every path a real Node run could print: its args (corpus
    dirs), its index dir, its $PWD-derived cwd, and a /private-prefixed
    form of each -- the scrub must leave none of them behind."""
    import os

    from parity._node_cli import NodeCliResult

    cwd = os.environ.get("PWD") or os.getcwd()
    paths = [*args, str(index_path.parent), f"{cwd}/not-a-real-path"]
    paths += ["/private" + p for p in paths if p.startswith(("/var/", "/tmp/"))]
    return NodeCliResult(exit_code=0, stdout="\n".join(paths), stderr=" ".join(paths))


def test_capture_cli_flow_snapshots_scrubs_run_specific_paths(tmp_path, monkeypatch):
    import os

    import parity.capture_node_snapshots as capture_module
    from parity.scenarios.cli_flow_scenarios import build_cli_flow_scenarios

    monkeypatch.setattr(capture_module, "run_node_cli", _echo_paths_node_cli)
    monkeypatch.setenv("PWD", str(tmp_path / "shell-cwd"))
    output_dir = tmp_path / "output"
    scenarios = build_cli_flow_scenarios()
    monkeypatch.setattr(capture_module, "build_cli_flow_scenarios", lambda: scenarios)

    capture_module.capture_cli_flow_snapshots(tmp_path / "qmd", output_dir)

    real_paths = [str(d) for s in scenarios for d in s.corpus_dirs]
    real_paths += [str(output_dir), str(tmp_path / "shell-cwd"), os.getcwd()]
    for scenario in scenarios:
        text = (output_dir / "cli_flow_raw" / f"{scenario.name}.json").read_text()
        for path in real_paths:
            assert path not in text, (scenario.name, path)
        assert "<CORPUS_1>" in text
        assert "<INDEX_DIR>" in text
        assert "<CWD>/not-a-real-path" in text


def test_capture_cli_flow_snapshots_raw_args_use_node_args_override(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "run_node_cli", _fake_flow_node_cli)
    output_dir = tmp_path / "output"

    capture_module.capture_cli_flow_snapshots(tmp_path, output_dir)

    raw = json.loads((output_dir / "cli_flow_raw" / "collection_lifecycle.json").read_text())
    assert raw["steps"]["remove_flow_b"]["args"] == ["collection", "remove", "flow-b"]


def test_capture_cli_flow_snapshots_wipes_stale_raw_files(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "run_node_cli", _fake_flow_node_cli)
    output_dir = tmp_path / "output"
    stale = output_dir / "cli_flow_raw" / "renamed_away.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    capture_module.capture_cli_flow_snapshots(tmp_path, output_dir)

    assert not stale.exists()


def test_run_capture_single_phase_refuses_on_commit_mismatch(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "COMMIT.txt").write_text("a" * 40 + "\ncaptured_at: x\n")
    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "b" * 40)
    called = []
    monkeypatch.setattr(capture_module, "capture_cli_flow_snapshots", lambda *a: called.append(a))

    with pytest.raises(SystemExit, match="COMMIT.txt"):
        capture_module.run_capture(_fake_profile(tmp_path), tmp_path, output_dir, "cli-flow")
    assert called == []


def test_run_capture_single_phase_refuses_without_existing_commit_file(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "a" * 40)

    with pytest.raises(SystemExit, match="--phase all"):
        capture_module.run_capture(_fake_profile(tmp_path), tmp_path, tmp_path / "out", "cli-flow")


def test_run_capture_single_phase_runs_only_that_phase_and_keeps_commit_file(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    commit_text = "a" * 40 + "\ncaptured_at: original\n"
    (output_dir / "COMMIT.txt").write_text(commit_text)
    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "a" * 40)
    calls = []
    for phase_fn in _PHASE_FUNCTIONS:
        monkeypatch.setattr(capture_module, phase_fn, lambda *a, _n=phase_fn: calls.append(_n))

    capture_module.run_capture(_fake_profile(tmp_path), tmp_path, output_dir, "cli-flow")

    assert calls == ["capture_cli_flow_snapshots"]
    assert (output_dir / "COMMIT.txt").read_text() == commit_text


def test_run_capture_all_runs_every_phase_and_writes_commit_file(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "c" * 40)
    calls = []
    for phase_fn in _PHASE_FUNCTIONS:
        monkeypatch.setattr(capture_module, phase_fn, lambda *a, _n=phase_fn: calls.append(_n))
    output_dir = tmp_path / "output"

    capture_module.run_capture(_fake_profile(tmp_path), tmp_path, output_dir, "all")

    assert calls == list(_PHASE_FUNCTIONS)
    assert (output_dir / "COMMIT.txt").read_text().startswith("c" * 40)


def _scripted_node_cli(query_outputs, calls, cleanup_exit_code=0):
    """A fake run_node_cli that records each call's subcommand and answers
    the Nth `query` call with query_outputs[N] (a list of result rows)."""
    from parity._node_cli import NodeCliResult

    def fake(qmd_repo_root, args, index_path, extra_env=None):
        calls.append(args[0])
        if args[0] == "query":
            rows = query_outputs[calls.count("query") - 1]
            return NodeCliResult(exit_code=0, stdout=json.dumps(rows), stderr="")
        if args[0] == "cleanup":
            return NodeCliResult(exit_code=cleanup_exit_code, stdout="", stderr="boom")
        return NodeCliResult(exit_code=0, stdout="", stderr="")

    return fake


def test_capture_quality_baseline_clears_llm_cache_between_query_passes(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    calls: list[str] = []
    rows = [{"file": "fakeqrels/17388232.md"}]
    monkeypatch.setattr(capture_module, "run_node_cli", _scripted_node_cli([rows] * 3, calls))

    capture_module.capture_quality_baseline(
        _fake_qrels_profile(tmp_path), tmp_path, tmp_path / "output", runs=3
    )

    assert calls == ["collection", "embed", "query", "cleanup", "query", "cleanup", "query"]


def test_capture_quality_baseline_writes_mean_stddev_and_every_pass(tmp_path, monkeypatch):
    import math

    import parity.capture_node_snapshots as capture_module

    hit_first = [{"file": "fakeqrels/17388232.md"}]
    hit_second = [{"file": "fakeqrels/other.md"}, {"file": "fakeqrels/17388232.md"}]
    miss = []
    calls: list[str] = []
    monkeypatch.setattr(
        capture_module, "run_node_cli", _scripted_node_cli([hit_first, hit_second, miss], calls)
    )
    output_dir = tmp_path / "output"

    capture_module.capture_quality_baseline(
        _fake_qrels_profile(tmp_path), tmp_path, output_dir, runs=3
    )

    baseline = json.loads((output_dir / "quality" / "node_query_baseline.json").read_text())
    ndcg_second = 1 / math.log2(3)
    assert baseline["mrr"] == pytest.approx(0.5)
    assert baseline["ndcg_at_10"] == pytest.approx((1 + ndcg_second) / 3)
    assert baseline["recall_at_10"] == pytest.approx(2 / 3)
    cal = baseline["calibration"]
    assert cal["runs"] == 3
    assert cal["num_queries"] == 1
    assert cal["stddev"]["mrr"] == pytest.approx(0.5)  # stdev([1, 0.5, 0])
    assert [run["mrr"] for run in cal["per_run"]] == pytest.approx([1.0, 0.5, 0.0])


def test_capture_quality_baseline_single_run_writes_no_calibration_and_no_cleanup(
    tmp_path, monkeypatch
):
    import parity.capture_node_snapshots as capture_module

    calls: list[str] = []
    rows = [{"file": "fakeqrels/17388232.md"}]
    monkeypatch.setattr(capture_module, "run_node_cli", _scripted_node_cli([rows], calls))
    output_dir = tmp_path / "output"

    capture_module.capture_quality_baseline(
        _fake_qrels_profile(tmp_path), tmp_path, output_dir, runs=1
    )

    baseline = json.loads((output_dir / "quality" / "node_query_baseline.json").read_text())
    assert baseline == {"mrr": 1.0, "ndcg_at_10": 1.0, "recall_at_10": 1.0}
    assert "cleanup" not in calls


def test_capture_quality_baseline_raises_when_cleanup_fails(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    calls: list[str] = []
    rows = [{"file": "fakeqrels/17388232.md"}]
    monkeypatch.setattr(
        capture_module,
        "run_node_cli",
        _scripted_node_cli([rows] * 2, calls, cleanup_exit_code=1),
    )

    with pytest.raises(RuntimeError, match="cleanup"):
        capture_module.capture_quality_baseline(
            _fake_qrels_profile(tmp_path), tmp_path, tmp_path / "output", runs=2
        )


def test_capture_quality_baseline_agreement_mode_stays_a_single_pass(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    calls: list[str] = []
    rows = [{"file": "fakenoqrels/doc1.md"}]
    monkeypatch.setattr(capture_module, "run_node_cli", _scripted_node_cli([rows], calls))

    capture_module.capture_quality_baseline(
        _fake_no_qrels_profile(tmp_path), tmp_path, tmp_path / "output", runs=5
    )

    assert calls == ["collection", "embed", "query"]


def test_capture_quality_baseline_rejects_zero_runs(tmp_path):
    import parity.capture_node_snapshots as capture_module

    with pytest.raises(ValueError, match="runs"):
        capture_module.capture_quality_baseline(
            _fake_qrels_profile(tmp_path), tmp_path, tmp_path / "output", runs=0
        )


def test_run_capture_passes_quality_runs_to_the_quality_phase(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    monkeypatch.setattr(capture_module, "get_node_commit", lambda root: "c" * 40)
    seen = []
    for phase_fn in _PHASE_FUNCTIONS:
        monkeypatch.setattr(capture_module, phase_fn, lambda *a, _n=phase_fn: seen.append((_n, a)))

    capture_module.run_capture(_fake_profile(tmp_path), tmp_path, tmp_path / "out", "all", 7)

    quality_args = dict(seen)["capture_quality_baseline"]
    assert quality_args[-1] == 7


def test_quality_runs_flag_defaults_to_30():
    from parity.capture_node_snapshots import DEFAULT_QUALITY_RUNS, build_arg_parser

    args = build_arg_parser().parse_args(["--qmd-repo-root", "x"])

    assert DEFAULT_QUALITY_RUNS == 30
    assert args.quality_runs == 30


def test_quality_runs_flag_accepts_a_positive_count():
    from parity.capture_node_snapshots import build_arg_parser

    args = build_arg_parser().parse_args(["--qmd-repo-root", "x", "--quality-runs", "5"])

    assert args.quality_runs == 5


@pytest.mark.parametrize("bad", ["0", "-3", "many"])
def test_quality_runs_flag_rejects_non_positive_or_non_integer_counts(bad):
    from parity.capture_node_snapshots import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--qmd-repo-root", "x", "--quality-runs", bad])


def test_load_queries_keeps_an_intent_when_present(tmp_path):
    from parity._queries import load_queries

    queries_file = tmp_path / "queries.json"
    queries_file.write_text(
        json.dumps(
            [
                {"query_id": "dev-0", "query": "q0", "intent": "scenario 0"},
                {"query_id": "dev-1", "query": "q1"},
            ]
        )
    )
    profile = DatasetProfile(
        name="fake", corpus_dir=tmp_path, queries_file=queries_file, qrels_file=None
    )

    assert load_queries(profile) == [
        {"query_id": "dev-0", "query": "q0", "intent": "scenario 0"},
        {"query_id": "dev-1", "query": "q1"},
    ]


def test_quality_capture_passes_intent_and_saves_pass_one_rankings(tmp_path, monkeypatch):
    import parity.capture_node_snapshots as capture_module

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.md").write_text("# A\nalpha")
    queries_file = tmp_path / "queries.json"
    queries_file.write_text(
        json.dumps([{"query_id": "dev-0", "query": "alpha?", "intent": "I need alpha"}])
    )
    qrels_file = tmp_path / "qrels.tsv"
    qrels_file.write_text("query-id\tcorpus-id\tscore\ndev-0\ta\t1\n")
    profile = DatasetProfile(
        name="fake", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=qrels_file
    )
    query_calls = []

    def fake_run_node_cli(qmd_repo_root, args, index_path, extra_env=None):
        from parity._node_cli import NodeCliResult

        if args[0] == "query":
            query_calls.append(args)
            return NodeCliResult(0, json.dumps([{"file": "qmd://fake/a.md"}]), "")
        return NodeCliResult(0, "", "")

    monkeypatch.setattr(capture_module, "run_node_cli", fake_run_node_cli)

    capture_module.capture_quality_baseline(profile, tmp_path, tmp_path / "out", runs=2)

    assert query_calls[0] == ["query", "alpha?", "--format", "json", "--intent", "I need alpha"]
    ranked = json.loads((tmp_path / "out" / "quality" / "node_query_ranked.json").read_text())
    assert ranked == {"dev-0": ["a"]}
