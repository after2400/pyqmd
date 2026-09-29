from datetime import UTC, datetime

import click

from pyqmd_mlx.cli._index_health import index_health_lines

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def _lines(pending, total, latest):
    counts = {
        "pending_embed": pending,
        "active_documents": total,
        "most_recent_modified_at": latest,
    }
    return [click.unstyle(line) for line in index_health_lines(counts, NOW)]


def test_warns_when_ten_percent_or_more_need_embedding():
    assert _lines(1, 10, "2026-09-29T00:00:00+00:00") == [
        "Warning: 1 documents (10%) need embeddings. Run 'pyqmd embed' for better results."
    ]


def test_tips_when_fewer_than_ten_percent_need_embedding():
    assert _lines(1, 11, "2026-09-29T00:00:00+00:00") == [
        "Tip: 1 documents need embeddings. Run 'pyqmd embed' to index them."
    ]


def test_percentage_rounds_half_up_like_javascript():
    # 1/8 = 12.5% -> Math.round gives 13 (Python's round() would give 12).
    assert _lines(1, 8, "2026-09-29T00:00:00+00:00")[0].startswith("Warning: 1 documents (13%)")


def test_tips_when_the_index_is_fourteen_days_stale():
    assert _lines(0, 5, "2026-09-15T12:00:00+00:00") == [
        "Tip: Index last updated 14 days ago. Run 'pyqmd update' to refresh."
    ]


def test_silent_when_current_and_fresh():
    assert _lines(0, 5, "2026-09-16T12:00:01+00:00") == []
    assert _lines(0, 0, None) == []


def test_accepts_a_z_suffixed_timestamp():
    assert _lines(0, 5, "2026-01-01T00:00:00Z")[0].startswith("Tip: Index last updated")


def test_warning_is_yellow_and_tips_are_dim():
    counts = {
        "pending_embed": 5,
        "active_documents": 5,
        "most_recent_modified_at": "2026-01-01T00:00:00Z",
    }
    warning, tip = index_health_lines(counts, NOW)
    assert warning == click.style(click.unstyle(warning), fg="yellow")
    assert tip == click.style(click.unstyle(tip), dim=True)
