"""Unit coverage for the arg-stripping logic in
parity/test_structural.py's `_invoke_args`, in isolation from the
active_profile/indexed_pyqmd_store fixtures (which require a fully embedded
corpus and captured Node snapshots that don't exist yet -- see
parity/test_structural.py's own module docstring).

These tests import and call the real `_invoke_args` function used by
test_cli_scenario_matches_snapshot, rather than duplicating its logic --
that way, if the fix in test_structural.py ever regresses, these tests
fail too.
"""

from __future__ import annotations

from typer.testing import CliRunner

from parity.test_structural import _APP_FOR_COMMAND, _invoke_args, _module_for_command
from pyqmd_mlx.cli.commands import collection as collection_cmd

runner = CliRunner()


def test_invoke_args_strips_collection_prefix():
    assert _invoke_args("collection", ["collection", "list"]) == ["list"]
    assert _invoke_args("collection", ["collection", "show", "scifact"]) == ["show", "scifact"]


def test_invoke_args_leaves_other_commands_unchanged():
    for command, args in [
        ("search", ["search", "biomaterials"]),
        ("get", ["get", "some/path.md"]),
        ("status", ["status"]),
        ("embed", ["embed"]),
    ]:
        assert _invoke_args(command, args) == args


def test_collection_dispatch_needs_stripped_args(monkeypatch):
    """End-to-end sanity check that the stripped args are what actually let
    Typer resolve the real subcommand, not just an isolated unit fact about
    _invoke_args."""
    monkeypatch.setattr(
        f"{_module_for_command('collection')}.get_store", lambda db_path=None: object()
    )

    args = ["collection", "list"]
    command = args[0]
    app = _APP_FOR_COMMAND[command]

    # Unstripped: Typer can't find a "collection" command inside
    # collection_cmd.app and fails with a usage error, not the real error.
    result_unstripped = runner.invoke(app, args)
    assert result_unstripped.exit_code == 2
    assert "No such command" in result_unstripped.output

    # Stripped (the fix): Typer resolves "list" and actually dispatches
    # into it -- it may still fail deeper (e.g. because our fake store
    # doesn't implement list_collections), but that's a different failure
    # than Typer's own command-resolution error above.
    result_stripped = runner.invoke(app, _invoke_args(command, args))
    assert result_stripped.output != result_unstripped.output
    assert "No such command" not in result_stripped.output


def test_show_dispatches_past_typer_resolution_when_stripped(monkeypatch):
    """Sanity check with a second collection subcommand (one that takes a
    positional arg) to make sure the fix isn't specific to zero-arg
    commands like `list`."""
    monkeypatch.setattr(
        f"{_module_for_command('collection')}.get_store", lambda db_path=None: object()
    )

    result = runner.invoke(
        collection_cmd.app, _invoke_args("collection", ["collection", "show", "scifact"])
    )
    assert "No such command" not in result.output
