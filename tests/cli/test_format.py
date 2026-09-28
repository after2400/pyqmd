from datetime import UTC, datetime, timedelta, timezone

import pytest

from pyqmd_mlx.cli._format import format_bytes, format_eta, format_ls_time, format_time_ago


def _iso(dt):
    return dt.isoformat()


def test_format_bytes_adds_space_before_unit():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(int(29.2 * 1024 * 1024)) == "29.2 MB"


def test_format_bytes_under_1024_is_a_plain_integer_not_one_decimal():
    """Matches Node's formatBytes exactly: bytes < 1024 print as a bare
    int ('191 B'), not '191.0 B' -- only KB/MB/GB get .toFixed(1)."""
    assert format_bytes(191) == "191 B"
    assert format_bytes(1023) == "1023 B"


def test_format_bytes_scales_up_through_units():
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024 * 1024) == "1.0 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
    assert format_bytes(1024**4) == "1.0 TB"


def test_format_time_ago_seconds():
    now = datetime.now(UTC)
    ts = _iso(now - timedelta(seconds=30))
    assert format_time_ago(ts, now=now) == "30s ago"


def test_format_time_ago_minutes_boundary():
    now = datetime.now(UTC)
    assert format_time_ago(_iso(now - timedelta(seconds=59)), now=now) == "59s ago"
    assert format_time_ago(_iso(now - timedelta(seconds=60)), now=now) == "1m ago"


def test_format_time_ago_hours_boundary():
    now = datetime.now(UTC)
    assert format_time_ago(_iso(now - timedelta(minutes=59)), now=now) == "59m ago"
    assert format_time_ago(_iso(now - timedelta(minutes=60)), now=now) == "1h ago"


def test_format_time_ago_days_boundary():
    now = datetime.now(UTC)
    assert format_time_ago(_iso(now - timedelta(hours=23)), now=now) == "23h ago"
    assert format_time_ago(_iso(now - timedelta(hours=24)), now=now) == "1d ago"


def test_format_ls_time_recent_uses_month_day_time_format():
    now = datetime(2026, 9, 16, 14, 30, tzinfo=UTC)
    ts = _iso(datetime(2026, 9, 10, 8, 5, tzinfo=UTC))
    assert format_ls_time(ts, now=now, local_tz=UTC) == "Sep 10 08:05"


def test_format_ls_time_older_than_six_months_shows_year():
    now = datetime(2026, 9, 16, 14, 30, tzinfo=UTC)
    ts = _iso(datetime(2025, 1, 1, 0, 0, tzinfo=UTC))
    assert format_ls_time(ts, now=now, local_tz=UTC) == "Jan 01  2025"


def test_format_ls_time_converts_to_local_timezone_not_utc():
    """Regression test: Node's Date.getHours()/getDate() read the
    system's LOCAL time, but stored timestamps are UTC -- a naive port
    that reads hour/day straight off the UTC value is off by the local
    offset (caught via a real qmd vs pyqmd ls diff: Node showed 20:58,
    pyqmd showed 00:58 for the same file's 00:58 UTC mtime, because the
    viewer was in UTC-4)."""
    edt = timezone(timedelta(hours=-4))
    now = datetime(2026, 9, 16, 14, 30, tzinfo=UTC)
    ts = _iso(datetime(2026, 9, 15, 0, 58, tzinfo=UTC))
    assert format_ls_time(ts, now=now, local_tz=edt) == "Sep 14 20:58"


def test_format_ls_time_local_tz_can_shift_the_year_boundary():
    edt = timezone(timedelta(hours=-4))
    now = datetime(2026, 9, 16, 14, 30, tzinfo=UTC)
    ts = _iso(datetime(2025, 1, 1, 2, 0, tzinfo=UTC))  # UTC Jan 1, but 2024-12-31 in EDT
    assert format_ls_time(ts, now=now, local_tz=edt) == "Dec 31  2024"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (0.4, "0s"),
        (0.5, "1s"),
        (12.2, "12s"),
        (59.4, "59s"),
        (60, "1m 0s"),
        (123, "2m 3s"),
        (3599, "59m 59s"),
        (3600, "1h 0m"),
        (3840, "1h 4m"),
    ],
)
def test_format_eta_matches_node_format_eta(seconds, expected):
    assert format_eta(seconds) == expected
