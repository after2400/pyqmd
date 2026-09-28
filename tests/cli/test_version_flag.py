from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app
from pyqmd_mlx.version import __version__

runner = CliRunner()


def test_version_flag_prints_package_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_version_matches_pyproject_metadata():
    from importlib.metadata import version

    assert version("pyqmd-mlx") == __version__
