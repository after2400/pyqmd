import pytest
import typer

from pyqmd_mlx.cli._metadata_filter import parse_cli_metadata_filter
from pyqmd_mlx.store._metadata_filter import MetadataCondition


def test_parse_cli_metadata_filter_returns_none_for_no_input():
    assert parse_cli_metadata_filter(None) is None


def test_parse_cli_metadata_filter_parses_valid_json():
    result = parse_cli_metadata_filter('{"key":"status","operator":"eq","value":"published"}')
    assert result == MetadataCondition(key="status", operator="eq", value="published")


def test_parse_cli_metadata_filter_exits_cleanly_on_invalid_json(capsys):
    with pytest.raises(typer.Exit):
        parse_cli_metadata_filter("not json")
    captured = capsys.readouterr()
    assert "Invalid --filter JSON" in captured.err
    assert "Example:" in captured.err


def test_parse_cli_metadata_filter_exits_cleanly_on_invalid_ast(capsys):
    with pytest.raises(typer.Exit):
        parse_cli_metadata_filter('{"key":"status","operator":"bogus","value":"x"}')
    captured = capsys.readouterr()
    assert "unknown operator" in captured.err
