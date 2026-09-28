"""Shared byte-size / timestamp formatting for status/ls output (roadmap
#10). Extracted so status.py and documents.py don't each carry their own
copy."""

import math
from datetime import UTC, datetime, tzinfo


def format_bytes(n: int) -> str:
    """Matches Node's formatBytes exactly: bytes < 1024 print as a bare
    int ('191 B'), not '191.0 B' -- only KB/MB/GB/TB get one decimal
    place (`.toFixed(1)` there)."""
    if n < 1024:
        return f"{n} B"
    size = n / 1024
    for unit in ("KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_time_ago(iso_timestamp: str, *, now: datetime | None = None) -> str:
    """Humanize an ISO-8601 timestamp as a relative time, matching Node's
    formatTimeAgo (src/cli/qmd.ts) exactly: <60s -> 'Ns ago', <60m -> 'Nm
    ago', <24h -> 'Nh ago', else 'Nd ago'."""
    now = now or datetime.now(UTC)
    then = datetime.fromisoformat(iso_timestamp)
    seconds = int((now - then).total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def format_ls_time(
    iso_timestamp: str, *, now: datetime | None = None, local_tz: tzinfo | None = None
) -> str:
    """Classic `ls -l` style timestamp, matching Node's formatLsTime
    (src/cli/qmd.ts) exactly: 'Mon DD HH:MM' within the last ~6 months
    (180 days), else 'Mon DD  YYYY' (two spaces before the year).

    Node's Date.getHours()/getMinutes()/getDate() etc. return the
    system's *local* time, not UTC -- stored timestamps are UTC
    (_now_iso() uses datetime.now(UTC)), so this must convert before
    reading date/time components, or the displayed hour is off by the
    local UTC offset (caught via a real qmd vs pyqmd ls diff: Node showed
    20:58, pyqmd showed 00:58 for the same file). `local_tz` exists only
    so tests can inject a fixed offset instead of depending on whatever
    timezone the test happens to run in; production leaves it None,
    which makes `astimezone()` convert to the system's real local zone.
    """
    now = now or datetime.now(UTC)
    then = datetime.fromisoformat(iso_timestamp).astimezone(local_tz)
    months = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
    month = months[then.month - 1]
    day = f"{then.day:02d}"
    if (now - then).days >= 180:
        return f"{month} {day}  {then.year}"
    return f"{month} {day} {then.hour:02d}:{then.minute:02d}"


def js_round(x: float) -> int:
    """JavaScript's Math.round for non-negative values: half-up, where
    Python's round() is banker's rounding. For porting Node's numbers."""
    return math.floor(x + 0.5)


def format_eta(seconds: float) -> str:
    """Port of Node qmd's formatETA (src/cli/qmd.ts): "12s", "2m 3s",
    "1h 4m"."""
    if seconds < 60:
        return f"{js_round(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {js_round(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
