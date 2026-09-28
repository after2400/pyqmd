import json

import pytest

from parity.dataset_profile import load_dataset_profile, resolve_active_profile_path
from parity.scenarios.mcp_scenarios import build_mcp_scenarios


@pytest.fixture(scope="module")
def scifact_scenarios():
    profile = load_dataset_profile(resolve_active_profile_path(None))
    return build_mcp_scenarios(profile)


@pytest.mark.requires_scifact_corpus
def test_mcp_scenarios_have_unique_names(scifact_scenarios):
    names = [s.name for s in scifact_scenarios]
    assert len(names) == len(set(names))


@pytest.mark.requires_scifact_corpus
def test_mcp_scenarios_cover_all_four_tools(scifact_scenarios):
    tools = {s.tool for s in scifact_scenarios}
    assert tools == {"query", "get", "multi_get", "status"}


@pytest.mark.requires_scifact_corpus
def test_extract_functions_are_callable(scifact_scenarios):
    for scenario in scifact_scenarios:
        result = scenario.extract({"results": []})
        json.dumps(result)  # must be JSON-serializable


def test_scenarios_use_the_active_profile_not_a_hardcoded_dataset():
    # Regression guard (2026-09-13 parity-suite review), mirroring the CLI-scenario test of
    # the same name.
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
        scenarios = build_mcp_scenarios(fake_profile)

    all_values = [v for s in scenarios for v in s.arguments.values()]
    assert "zylophone research topic" in all_values
    assert any("aaa.md" in v for v in all_values)
    assert not any("biomaterials" in v for v in all_values)


def test_query_invariants_shape_never_asserts_exact_document_set():
    from parity.scenarios.mcp_scenarios import _query_invariants_shape

    # Node's MCP query tool emits bare paths in `file`, not qmd:// URIs --
    # this shape must not assume otherwise.
    result = _query_invariants_shape({"results": [{"file": "c/a.md"}, {"file": "c/b.md"}]})
    assert result == {"count": 2}


def test_query_top1_shape_returns_only_the_first_result():
    from parity.scenarios.mcp_scenarios import _query_top1_shape

    result = _query_top1_shape({"results": [{"file": "qmd://c/a.md"}, {"file": "qmd://c/b.md"}]})
    assert result == {"top_file": "qmd://c/a.md"}
