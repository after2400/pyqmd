import json
import warnings

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app
from pyqmd_mlx.store import Store

runner = CliRunner()


def test_help_lists_all_top_level_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "collection",
        "context",
        "get",
        "multi-get",
        "ls",
        "search",
        "vsearch",
        "query",
        "bench",
        "embed",
        "status",
        "update",
        "cleanup",
        "mcp",
    ):
        assert command in result.output


def test_collection_subcommand_help():
    result = runner.invoke(app, ["collection", "--help"])
    assert result.exit_code == 0
    for sub in ("add", "list", "show", "remove", "rename"):
        assert sub in result.output


@pytest.mark.parametrize("path", [[], ["search"], ["collection", "add"]])
def test_short_h_is_an_alias_for_help(path):
    long_result = runner.invoke(app, [*path, "--help"])
    short_result = runner.invoke(app, [*path, "-h"])
    assert long_result.exit_code == 0
    assert short_result.exit_code == 0
    assert short_result.output == long_result.output


@pytest.mark.requires_expansion_weights
def test_end_to_end_add_embed_query_via_real_app(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text("# Authentication\nHow to configure auth for your app.")

    def _fake_embed(texts, model, kind="query", title=None):
        return [[1.0, 0.0] for _ in texts]

    def _fake_rerank(query, documents, model):
        return [1.0 for _ in documents]

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    for module_name in ("collection", "documents", "search", "embed", "status"):
        monkeypatch.setattr(
            f"pyqmd_mlx.cli.commands.{module_name}.get_store", lambda db_path=None, s=store: s
        )

    add_result = runner.invoke(app, ["collection", "add", str(tmp_path), "--name", "notes"])
    assert add_result.exit_code == 0

    embed_result = runner.invoke(app, ["embed"])
    assert embed_result.exit_code == 0

    query_result = runner.invoke(app, ["query", "authentication", "--format", "json"])
    assert query_result.exit_code == 0
    parsed = json.loads(query_result.stdout)
    assert parsed[0]["title"] == "Authentication"


def test_embed_and_status_do_not_emit_callback_not_supported_warning(tmp_path, monkeypatch):
    """Regression test for the add_typer-with-callback pitfall.

    embed.app and status.app each carry a permanent no-op @app.callback()
    (added in Tasks 13/14 to keep them invocable as single-command Typer
    apps). If app.py mounted them via `app.add_typer(embed.app)` /
    `app.add_typer(status.app)` (flat, no name=), Typer would detect the
    callback and emit `UserWarning: The 'callback' parameter is not
    supported by Typer when using add_typer without a name`, dropping the
    callback -- and this fires on every real invocation, including
    `pyqmd --help` and unrelated commands. app.py must instead register the
    underlying command functions directly via `app.command(...)`.
    """

    def _fake_embed(texts, model, kind="query", title=None):
        return [[1.0, 0.0] for _ in texts]

    def _fake_rerank(query, documents, model):
        return [1.0 for _ in documents]

    store = Store(":memory:", embed_fn=_fake_embed, rerank_fn=_fake_rerank)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.embed.get_store", lambda db_path=None: store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.status.get_store", lambda db_path=None: store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.update.get_store", lambda db_path=None: store)
    monkeypatch.setattr("pyqmd_mlx.cli.commands.cleanup.get_store", lambda db_path=None: store)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embed_result = runner.invoke(app, ["embed"])
        status_result = runner.invoke(app, ["status"])
        update_result = runner.invoke(app, ["update"])
        cleanup_result = runner.invoke(app, ["cleanup"])
        mcp_help_result = runner.invoke(app, ["mcp", "--help"])
        help_result = runner.invoke(app, ["--help"])

    assert embed_result.exit_code == 0
    assert status_result.exit_code == 0
    assert update_result.exit_code == 0
    assert cleanup_result.exit_code == 0
    assert mcp_help_result.exit_code == 0
    assert help_result.exit_code == 0

    callback_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "not supported by Typer" in str(w.message)
    ]
    assert callback_warnings == []

    for result in (
        embed_result,
        status_result,
        update_result,
        cleanup_result,
        mcp_help_result,
        help_result,
    ):
        assert "not supported by Typer" not in (result.output or "")
        stderr = ""
        try:
            stderr = result.stderr
        except ValueError:
            pass
        assert "not supported by Typer" not in stderr
