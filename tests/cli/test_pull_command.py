from typer.testing import CliRunner

from pyqmd_mlx.cli.commands.pull import app

runner = CliRunner()

_MESSAGE = (
    "pyqmd's MLX models download automatically on first use — no separate pull step is needed."
)


def test_pull_prints_explanation_and_exits_zero():
    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 0
    assert _MESSAGE in result.output


def test_pull_accepts_and_ignores_node_flags():
    result = runner.invoke(app, ["pull", "--refresh", "--progress"])

    assert result.exit_code == 0
    assert _MESSAGE in result.output
