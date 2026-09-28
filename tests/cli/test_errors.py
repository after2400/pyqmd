import pytest
import typer
from typer.testing import CliRunner

from pyqmd_mlx.cli._errors import CliError, run_or_exit

app = typer.Typer()


@app.command()
def boom(kind: str) -> None:
    def _fail():
        if kind == "cli":
            raise CliError(
                "Collection 'x' already exists.", "Use a different name with --name <name>"
            )
        raise ValueError("plain failure")

    run_or_exit(_fail)


runner = CliRunner()


def test_cli_error_prints_each_line_to_stderr_without_prefix():
    result = runner.invoke(app, ["cli"])
    assert result.exit_code == 1
    assert result.stderr == (
        "Collection 'x' already exists.\nUse a different name with --name <name>\n"
    )
    assert result.stdout == ""


def test_other_expected_errors_keep_error_prefix():
    result = runner.invoke(app, ["value"])
    assert result.exit_code == 1
    assert result.stderr == "Error: plain failure\n"


def test_cli_error_str_joins_lines():
    assert str(CliError("a", "b")) == "a\nb"


def test_cli_error_requires_at_least_one_line():
    with pytest.raises(ValueError):
        CliError()
