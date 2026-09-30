"""Node-independent regression tests for the CliFlowScenario definitions
in parity/scenarios/cli_flow_scenarios.py -- runs each flow's real steps
against a real (fast, fake-embed) pyqmd Store via the actual Typer CLI
apps, and checks the flow's own narrative holds (add -> list shows it,
remove -> list doesn't). No Node/bun involved; this is the TDD vehicle
for the flow definitions themselves, independent of parity/test_structural.py's
separate snapshot-comparison tests (which need real Node captures and
are added in Task 7)."""

from __future__ import annotations

import pytest

from parity.scenarios.cli_flow_scenarios import (
    FlowStepResult,
    build_cli_flow_scenarios,
    run_pyqmd_flow,
    run_pyqmd_flow_detailed,
)
from pyqmd_mlx.store import Store


def _fake_embed(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def flow_scenarios():
    return {s.name: s for s in build_cli_flow_scenarios()}


def _fresh_store(tmp_path):
    return Store(str(tmp_path / "index.sqlite"), embed_fn=_fake_embed)


def test_collection_lifecycle_flow(flow_scenarios, tmp_path, monkeypatch):
    scenario = flow_scenarios["collection_lifecycle"]
    store = _fresh_store(tmp_path)
    results = run_pyqmd_flow(scenario, store, monkeypatch)

    assert results["add_flow_a"] == {
        "exit_code": 0,
        "indexed": 4,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
    }
    assert results["add_flow_a_duplicate"]["exit_code"] != 0
    assert results["add_flow_dummy"] == {
        "exit_code": 0,
        "indexed": 4,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
    }
    assert results["list_after_adds"] == {"exit_code": 0, "names": ["flow-a", "flow-dummy"]}
    assert results["set_update_cmd"] == {"exit_code": 0}
    assert results["rename_collision"]["exit_code"] != 0
    assert results["rename_real"] == {"exit_code": 0}
    assert results["show_flow_b"] == {"exit_code": 0, "mentions_profile_name": True}
    assert results["exclude_flow_b"] == {"exit_code": 0}
    assert results["list_shows_excluded_tag"] == {
        "exit_code": 0,
        "names": ["flow-b", "flow-dummy"],
    }
    assert results["include_flow_b"] == {"exit_code": 0}
    assert results["remove_flow_b"] == {"exit_code": 0}
    assert results["remove_flow_b_again"]["exit_code"] != 0
    assert results["list_final"] == {"exit_code": 0, "names": ["flow-dummy"]}


def test_context_lifecycle_flow(flow_scenarios, tmp_path, monkeypatch):
    scenario = flow_scenarios["context_lifecycle"]
    store = _fresh_store(tmp_path)
    results = run_pyqmd_flow(scenario, store, monkeypatch)

    assert results["add_collection"] == {
        "exit_code": 0,
        "indexed": 4,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
    }
    assert results["context_add_root"] == {"exit_code": 0}
    assert results["context_add_sub"] == {"exit_code": 0}
    assert results["context_list_after_adds"] == {
        "exit_code": 0,
        "entries": [
            {"collection": "flow-ctx", "path": "/ (root)", "context": "root context"},
            {"collection": "flow-ctx", "path": "sub.md", "context": "file context"},
        ],
    }
    assert results["context_add_bad_path"]["exit_code"] != 0
    assert results["context_remove_root"] == {"exit_code": 0}
    assert results["context_remove_root_again"]["exit_code"] != 0
    assert results["context_list_final"] == {
        "exit_code": 0,
        "entries": [{"collection": "flow-ctx", "path": "sub.md", "context": "file context"}],
    }


def test_update_lifecycle_flow(flow_scenarios, tmp_path, monkeypatch):
    scenario = flow_scenarios["update_lifecycle"]
    store = _fresh_store(tmp_path)
    results = run_pyqmd_flow(scenario, store, monkeypatch)

    assert results["add_collection"] == {
        "exit_code": 0,
        "indexed": 4,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
    }
    assert results["update_after_mutation"] == {
        "exit_code": 0,
        "indexed": 1,
        "updated": 1,
        "unchanged": 2,
        "removed": 1,
    }


def test_embed_lifecycle_flow(flow_scenarios, tmp_path, monkeypatch):
    scenario = flow_scenarios["embed_lifecycle"]
    store = _fresh_store(tmp_path)
    results = run_pyqmd_flow(scenario, store, monkeypatch)

    assert results["add_collection"] == {
        "exit_code": 0,
        "indexed": 4,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
    }
    assert results["embed_first"] == {"exit_code": 0, "chunks_embedded": 4, "docs_processed": 4}
    assert results["embed_noop"] == {
        "exit_code": 0,
        "chunks_embedded": None,
        "docs_processed": None,
    }
    assert results["embed_force"] == {"exit_code": 0, "chunks_embedded": 4, "docs_processed": 4}
    assert results["embed_bad_collection"]["exit_code"] != 0


def test_ast_chunking_embed_scenario_produces_multiple_chunks(
    flow_scenarios, tmp_path, monkeypatch
):
    scenarios = build_cli_flow_scenarios()
    scenario = next(s for s in scenarios if s.name == "ast_chunking_embed")

    store = Store(str(tmp_path / "index.sqlite"), embed_fn=_fake_embed)
    results = run_pyqmd_flow(scenario, store, monkeypatch)

    embed_result = results["embed_auto"]
    assert embed_result["exit_code"] == 0
    assert embed_result["chunks_embedded"] is not None
    assert embed_result["chunks_embedded"] > 1


def test_every_flow_records_its_corpus_dirs(flow_scenarios):
    expected_counts = {
        "collection_lifecycle": 2,
        "context_lifecycle": 1,
        "update_lifecycle": 1,
        "embed_lifecycle": 1,
        "ast_chunking_embed": 1,
    }
    for name, count in expected_counts.items():
        scenario = flow_scenarios[name]
        assert len(scenario.corpus_dirs) == count, name
        assert all(d.is_dir() for d in scenario.corpus_dirs), name


def test_placeholders_number_corpus_dirs_from_one(flow_scenarios):
    scenario = flow_scenarios["collection_lifecycle"]
    a, b = scenario.corpus_dirs
    assert scenario.placeholders() == {str(a): "<CORPUS_1>", str(b): "<CORPUS_2>"}


def test_corpus_dirs_appear_in_their_steps_args(flow_scenarios):
    scenario = flow_scenarios["collection_lifecycle"]
    all_args = [arg for step in scenario.steps for arg in step.args]
    for corpus in scenario.corpus_dirs:
        assert str(corpus) in all_args


def test_step_text_fields_default_to_no_declared_differences(flow_scenarios):
    step = flow_scenarios["context_lifecycle"].steps[0]
    assert step.text_subs == []
    assert step.text_skip_reason is None


def test_run_pyqmd_flow_detailed_returns_raw_streams(flow_scenarios, tmp_path, monkeypatch):
    scenario = flow_scenarios["collection_lifecycle"]
    results = run_pyqmd_flow_detailed(scenario, _fresh_store(tmp_path), monkeypatch)

    assert list(results) == [step.name for step in scenario.steps]
    assert all(isinstance(r, FlowStepResult) for r in results.values())
    duplicate = results["add_flow_a_duplicate"]
    assert duplicate.exit_code != 0
    assert duplicate.stderr.strip() != ""
    assert results["add_flow_a"].extracted["indexed"] == 4
