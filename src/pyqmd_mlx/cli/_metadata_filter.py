"""Shared --filter <json> parsing for the search/vsearch/query CLI commands.
Mirrors the Node reference's parseCliMetadataFilter: invalid JSON or an
invalid filter AST both exit cleanly with an actionable message, never a
raw traceback."""

import json as _json

import typer

from pyqmd_mlx.store._metadata_filter import (
    MetadataFilter,
    MetadataFilterError,
    parse_metadata_filter,
)


def parse_cli_metadata_filter(raw: str | None) -> "MetadataFilter | None":
    if raw is None:
        return None

    try:
        filter_json = _json.loads(raw)
    except ValueError as exc:
        typer.echo(f"Invalid --filter JSON: {exc}", err=True)
        typer.echo(
            'Example: --filter \'{"key":"status","operator":"eq","value":"published"}\'', err=True
        )
        raise typer.Exit(1) from exc

    try:
        return parse_metadata_filter(filter_json)
    except MetadataFilterError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
