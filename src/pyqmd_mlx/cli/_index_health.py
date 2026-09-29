"""Index-health warning for vsearch/query, ported from qmd.ts's
checkIndexHealth (qmd.ts:361-378): pending embeddings (a warning at 10%
or more of active documents, else a tip) and a stale index (a tip when
the newest document is 14 or more days old). Written to stderr, so JSON
and other machine formats on stdout are unaffected."""

import math
from datetime import UTC, datetime

import typer

from pyqmd_mlx.cli._theme import dim, yellow

STALE_AFTER_DAYS = 14


def _days_since(iso_timestamp: str, now: datetime) -> int:
    then = datetime.fromisoformat(iso_timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return math.floor((now - then).total_seconds() / 86400)


def index_health_lines(counts: dict, now: datetime) -> list[str]:
    lines: list[str] = []
    pending = counts["pending_embed"]
    total = counts["active_documents"]
    if pending > 0 and total > 0:
        # Math.round: halves round up (Python's round() rounds to even).
        pct = math.floor((pending / total) * 100 + 0.5)
        if pct >= 10:
            lines.append(
                yellow(
                    f"Warning: {pending} documents ({pct}%) need embeddings. "
                    "Run 'pyqmd embed' for better results."
                )
            )
        else:
            lines.append(
                dim(f"Tip: {pending} documents need embeddings. Run 'pyqmd embed' to index them.")
            )
    latest = counts["most_recent_modified_at"]
    if latest:
        days = _days_since(latest, now)
        if days >= STALE_AFTER_DAYS:
            lines.append(
                dim(f"Tip: Index last updated {days} days ago. Run 'pyqmd update' to refresh.")
            )
    return lines


def check_index_health(store) -> None:
    for line in index_health_lines(store.get_status_counts(), datetime.now(UTC)):
        typer.echo(line, err=True)
