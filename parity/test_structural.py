"""Structural parity tests: does live pyqmd produce the same
model-independent output shape as a captured, frozen Node qmd snapshot,
for the active dataset profile? Run with:

    uv run pytest parity/test_structural.py [--dataset-config PATH]

Requires captured snapshots (see capture_node_snapshots.py) -- these tests
fail with a clear FileNotFoundError if none exist yet for the active
profile, which correctly signals the missing prerequisite.
"""

from __future__ import annotations

import asyncio
import difflib
import os

import pytest
from typer.testing import CliRunner

from parity._mcp_client import normalize_call_tool_result
from parity._snapshot_io import load_flow_raw, load_snapshot
from parity._text_normalize import normalize
from parity.dataset_profile import load_dataset_profile, resolve_active_profile_path
from parity.scenarios.cli_flow_scenarios import (
    build_cli_flow_scenarios,
    run_pyqmd_flow,
    run_pyqmd_flow_detailed,
)
from parity.scenarios.cli_scenarios import build_cli_scenarios
from parity.scenarios.mcp_scenarios import build_mcp_scenarios
from pyqmd_mlx.cli.commands import collection as collection_cmd
from pyqmd_mlx.cli.commands import documents as documents_cmd
from pyqmd_mlx.cli.commands import embed as embed_cmd
from pyqmd_mlx.cli.commands import status as status_cmd
from pyqmd_mlx.cli.commands.search import app as search_app
from pyqmd_mlx.mcp.server import build_server
from pyqmd_mlx.store import Store


def _fake_embed_for_flows(texts, model, kind="query", title=None):
    return [[1.0, 0.0] for _ in texts]


runner = CliRunner()

_APP_FOR_COMMAND = {
    "search": search_app,
    "vsearch": search_app,
    "query": search_app,
    "get": documents_cmd.app,
    "multi-get": documents_cmd.app,
    "ls": documents_cmd.app,
    "collection": collection_cmd.app,
    "embed": embed_cmd.app,
    "status": status_cmd.app,
}


def _module_for_command(command: str):
    return {
        "search": "pyqmd_mlx.cli.commands.search",
        "vsearch": "pyqmd_mlx.cli.commands.search",
        "query": "pyqmd_mlx.cli.commands.search",
        "get": "pyqmd_mlx.cli.commands.documents",
        "multi-get": "pyqmd_mlx.cli.commands.documents",
        "ls": "pyqmd_mlx.cli.commands.documents",
        "collection": "pyqmd_mlx.cli.commands.collection",
        "embed": "pyqmd_mlx.cli.commands.embed",
        "status": "pyqmd_mlx.cli.commands.status",
    }[command]


def _invoke_args(command: str, args: list[str]) -> list[str]:
    """`collection` is mounted into the top-level app with
    `name="collection"` (see pyqmd_mlx/cli/app.py), so the "collection" prefix in
    scenario.args is added by the PARENT app at mount time -- it is not a
    command registered on collection_cmd.app itself (whose commands are
    named "add"/"list"/"show"/"remove"/"rename"). Invoking
    collection_cmd.app directly with the unstripped args would make Typer
    look for a literal "collection" command inside that sub-app and fail to
    resolve it. Every other entry in _APP_FOR_COMMAND is either mounted flat
    (no name=) or is a command registered directly on its app, so no
    stripping is needed for them."""
    return args[1:] if command == "collection" else args


# Known, documented gaps -- see parity/README.md's "Known gaps" section for
# the full explanation of each. xfail(strict=True) so `just test-parity`
# shows a clean, all-green suite with the real gaps called out explicitly,
# and so the moment either gap actually gets fixed, the resulting XPASS
# fails the suite and forces removal of the marker here.
_CLI_KNOWN_GAPS = {}


def pytest_generate_tests(metafunc):
    """CLI/MCP scenarios depend on the active DatasetProfile (--dataset-
    config), which is only resolvable from pytest's Config -- not available
    at plain module-import time, which is when a static
    @pytest.mark.parametrize list would need to exist. Resolving the
    profile here and building scenarios from it keeps CLI_SCENARIOS/
    MCP_SCENARIOS from ever hardcoding one dataset."""
    if "scenario" not in metafunc.fixturenames and "flow_scenario" not in metafunc.fixturenames:
        return
    profile_path = resolve_active_profile_path(metafunc.config.getoption("--dataset-config"))
    profile = load_dataset_profile(profile_path)
    if metafunc.function.__name__ == "test_cli_scenario_matches_snapshot":
        params = [
            pytest.param(
                s,
                marks=[pytest.mark.xfail(reason=_CLI_KNOWN_GAPS[s.name], strict=True)]
                if s.name in _CLI_KNOWN_GAPS
                else [],
                id=s.name,
            )
            for s in build_cli_scenarios(profile)
        ]
        metafunc.parametrize("scenario", params)
    elif metafunc.function.__name__ == "test_mcp_scenario_matches_snapshot":
        metafunc.parametrize("scenario", build_mcp_scenarios(profile), ids=lambda s: s.name)
    elif metafunc.function.__name__ == "test_cli_flow_scenario_matches_snapshot":
        metafunc.parametrize("flow_scenario", build_cli_flow_scenarios(), ids=lambda s: s.name)
    elif metafunc.function.__name__ == "test_cli_flow_step_text_matches_node":
        # Built separately from the structural test's scenarios on purpose:
        # each build_cli_flow_scenarios() call makes fresh corpora, and
        # update_lifecycle mutates its corpus on disk -- sharing scenario
        # objects between the two tests would run those steps twice.
        scenarios = build_cli_flow_scenarios()
        metafunc.parametrize(
            ("flow_scenario", "step_name"),
            [(s, step.name) for s in scenarios for step in s.steps],
            ids=[f"{s.name}.{step.name}" for s in scenarios for step in s.steps],
        )


@pytest.mark.parity
def test_cli_scenario_matches_snapshot(scenario, active_profile, indexed_pyqmd_store, monkeypatch):
    command = scenario.args[0]
    app = _APP_FOR_COMMAND[command]
    monkeypatch.setattr(
        f"{_module_for_command(command)}.get_store", lambda db_path=None: indexed_pyqmd_store
    )

    invoke_args = _invoke_args(command, scenario.args)
    result = runner.invoke(app, invoke_args)
    # .output is stdout+stderr interleaved; the captured Node snapshot used
    # stdout only (see capture_node_snapshots.py, which extracts from
    # `result.stdout`) -- comparing against .output would silently corrupt
    # the first scenario that succeeds *and* writes to stderr (e.g. a
    # warning on an otherwise-successful command).
    live = scenario.extract(result.stdout, result.exit_code)
    expected = load_snapshot(active_profile.name, "cli", scenario.name)

    assert live == expected


@pytest.mark.parity
def test_cli_flow_scenario_matches_snapshot(flow_scenario, active_profile, tmp_path, monkeypatch):
    store = Store(str(tmp_path / "index.sqlite"), embed_fn=_fake_embed_for_flows)
    live = run_pyqmd_flow(flow_scenario, store, monkeypatch)
    expected = load_snapshot(active_profile.name, "cli_flow", flow_scenario.name)
    assert live == expected


@pytest.mark.parity
def test_mcp_scenario_matches_snapshot(scenario, active_profile, indexed_pyqmd_store):
    server = build_server(indexed_pyqmd_store)
    result = asyncio.run(server.call_tool(scenario.tool, scenario.arguments))

    live = normalize_call_tool_result(scenario, result)
    expected = load_snapshot(active_profile.name, "mcp", scenario.name)
    assert live == expected


@pytest.fixture(scope="module")
def pyqmd_flow_text_runs(tmp_path_factory):
    """Runs each flow at most once per module (every step's text test
    reads from the cached run), returning (results, placeholders)."""
    cache: dict[str, tuple[dict, dict[str, str]]] = {}

    def run(scenario):
        if scenario.name not in cache:
            store_dir = tmp_path_factory.mktemp(f"flow_text_{scenario.name}")
            store = Store(str(store_dir / "index.sqlite"), embed_fn=_fake_embed_for_flows)
            with pytest.MonkeyPatch.context() as mp:
                results = run_pyqmd_flow_detailed(scenario, store, mp)
            placeholders = {
                **scenario.placeholders(),
                str(store_dir): "<INDEX_DIR>",
                os.getcwd(): "<CWD>",
            }
            cache[scenario.name] = (results, placeholders)
        return cache[scenario.name]

    return run


@pytest.mark.parity
def test_cli_flow_step_text_matches_node(
    flow_scenario, step_name, active_profile, pyqmd_flow_text_runs
):
    raw = load_flow_raw(active_profile.name, flow_scenario.name)
    if raw is None:
        pytest.skip(
            "no raw flow capture; run `uv run python -m parity.capture_node_snapshots "
            "--qmd-repo-root <path> --phase cli-flow`"
        )
    node_step = raw["steps"].get(step_name)
    if node_step is None:
        pytest.skip(f"step {step_name!r} not in the raw capture; re-run with --phase cli-flow")
    step = next(s for s in flow_scenario.steps if s.name == step_name)
    if step.text_skip_reason:
        pytest.skip(step.text_skip_reason)

    results, pyqmd_placeholders = pyqmd_flow_text_runs(flow_scenario)
    live = results[step_name]
    for stream in ("stdout", "stderr"):
        # Node's raw capture already had its paths replaced with the same
        # placeholders at capture time (so no real path is ever committed).
        expected = normalize(node_step[stream], {}, step.text_subs)
        actual = normalize(getattr(live, stream), pyqmd_placeholders, step.text_subs)
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"node {stream}",
                tofile=f"pyqmd {stream}",
                lineterm="",
            )
        )
        assert actual == expected, f"{flow_scenario.name}.{step_name} {stream} differs:\n{diff}"
