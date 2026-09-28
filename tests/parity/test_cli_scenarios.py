import json

import pytest

from parity.dataset_profile import load_dataset_profile, resolve_active_profile_path
from parity.scenarios.cli_scenarios import (
    _collection_list_shape,
    _collection_show_shape,
    _ls_shape,
    build_cli_scenarios,
)


@pytest.fixture(scope="module")
def scifact_scenarios():
    profile = load_dataset_profile(resolve_active_profile_path(None))
    return build_cli_scenarios(profile)


@pytest.mark.requires_scifact_corpus
def test_cli_scenarios_have_unique_names(scifact_scenarios):
    names = [s.name for s in scifact_scenarios]
    assert len(names) == len(set(names))


@pytest.mark.requires_scifact_corpus
def test_cli_scenarios_is_not_empty(scifact_scenarios):
    assert len(scifact_scenarios) > 0


@pytest.mark.requires_scifact_corpus
def test_every_scenario_args_starts_with_a_command_name(scifact_scenarios):
    known_commands = {
        "search",
        "vsearch",
        "query",
        "get",
        "multi-get",
        "ls",
        "collection",
        "embed",
        "status",
    }
    for scenario in scifact_scenarios:
        assert scenario.args[0] in known_commands, (
            f"{scenario.name}: unknown command {scenario.args[0]!r}"
        )


@pytest.mark.requires_scifact_corpus
def test_search_and_query_scenarios_are_present(scifact_scenarios):
    names = {s.name for s in scifact_scenarios}
    assert any("search" in n for n in names)
    assert any("query" in n for n in names)
    assert any("get" in n for n in names)


@pytest.mark.requires_scifact_corpus
def test_extract_functions_are_callable_and_return_json_serializable_output(scifact_scenarios):
    for scenario in scifact_scenarios:
        result = scenario.extract('{"not": "real output, just checking extract runs"}', 0)
        # Must not raise, and must be JSON-round-trippable (a real snapshot
        # file has to be able to store whatever extract() returns).
        json.dumps(result)


@pytest.mark.requires_scifact_corpus
def test_collection_and_status_scenarios_are_present(scifact_scenarios):
    names = {s.name for s in scifact_scenarios}
    assert any("collection" in n for n in names)
    assert any("status" in n for n in names)


def test_scenarios_use_the_active_profile_not_a_hardcoded_dataset():
    # Regression guard (2026-09-13 parity-suite review): every scenario's args must be built
    # from the profile passed to build_cli_scenarios(), not literal scifact
    # strings baked into the module -- swap in a fake profile and confirm
    # its own query/filenames/collection name show up instead.
    import tempfile
    from pathlib import Path

    from parity.dataset_profile import DatasetProfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "aaa.md").write_text(
            "# Title\nA distinctive unique sentence about zylophones."
        )
        (corpus_dir / "bbb.md").write_text("# Title\nAnother document entirely.")
        queries_file = tmp_path / "queries.yaml"
        queries_file.write_text("- zylophone research topic\n")

        fake_profile = DatasetProfile(
            name="totallyfake", corpus_dir=corpus_dir, queries_file=queries_file, qrels_file=None
        )
        scenarios = build_cli_scenarios(fake_profile)

    all_args = [arg for s in scenarios for arg in s.args]
    assert "zylophone research topic" in all_args
    assert "aaa.md" in all_args
    assert "totallyfake" in all_args
    assert not any("scifact" in arg for arg in all_args)
    assert not any("biomaterials" in arg for arg in all_args)


def test_collection_list_shape_ignores_node_header_and_indented_detail_lines():
    node_output = (
        "Collections (1):\n\nprobetest (qmd://probetest/)\n"
        "  Pattern:  **/*.md\n  Files:    2\n  Updated:  0s ago\n"
    )
    assert _collection_list_shape(node_output, 0) == {"exit_code": 0, "names": ["probetest"]}


def test_collection_list_shape_matches_pyqmd_single_line_format():
    pyqmd_output = "probetest  /some/path  (**/*.md)\n"
    assert _collection_list_shape(pyqmd_output, 0) == {"exit_code": 0, "names": ["probetest"]}


def test_collection_show_shape_does_not_require_a_document_count_field():
    # Node's `collection show` never prints a document/file count at all --
    # only presence of the collection name is a genuinely shared signal.
    node_output = (
        "Collection: probetest\n  Path:     /x\n  Pattern:  **/*.md\n  Include:  yes (default)\n"
    )
    pyqmd_output = "Name: probetest\nPath: /x\nPattern: **/*.md\nDocuments: 2\n"
    assert _collection_show_shape(node_output, 0, "probetest") == _collection_show_shape(
        pyqmd_output, 0, "probetest"
    )


def test_ls_shape_normalizes_node_header_and_qmd_uri_wrapping():
    node_output = "Collections:\n\n  qmd://probetest/  (2 files)\n"
    pyqmd_output = "probetest\n"
    assert (
        _ls_shape(node_output, 0)
        == _ls_shape(pyqmd_output, 0)
        == {
            "exit_code": 0,
            "names": ["probetest"],
        }
    )
