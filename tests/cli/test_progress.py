import io

import pytest

from pyqmd_mlx.cli._progress import EmbedProgress, final_bar_line, render_bar


class FakeTerminal(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeClock:
    def __init__(self, *times: float) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


@pytest.mark.parametrize(
    ("percent", "filled"),
    [(0, 0), (1.6, 0), (1.7, 1), (50, 15), (99, 30), (100, 30)],
)
def test_render_bar_matches_node_render_progress_bar(percent, filled):
    bar = render_bar(percent)
    assert len(bar) == 30
    assert bar == "█" * filled + "░" * (30 - filled)


def test_final_bar_line_matches_node_plain_text():
    assert _strip_ansi(final_bar_line()) == "\r" + "█" * 30 + " 100%" + " " * 36


def test_final_bar_line_colors_bar_green_and_percent_bold():
    line = final_bar_line()
    assert "\x1b[32m" + "█" * 30 + "\x1b[0m" in line
    assert "\x1b[1m100%\x1b[0m" in line


def test_non_tty_stream_gets_nothing_at_all():
    stream = FakeTerminal(tty=False)
    with EmbedProgress(total_bytes=1000, stream=stream, clock=FakeClock(0, 1)) as progress:
        progress.update(bytes_processed=500, chunks_embedded=3)
    assert stream.getvalue() == ""


def test_tty_start_hides_cursor_and_sets_indeterminate():
    stream = FakeTerminal(tty=True)
    progress = EmbedProgress(total_bytes=1000, stream=stream, clock=FakeClock(0))
    progress.start()
    assert stream.getvalue() == "\x1b[?25l\x1b]9;4;3\x07"


def test_tty_update_redraws_node_style_line_with_osc_percent():
    stream = FakeTerminal(tty=True)
    progress = EmbedProgress(total_bytes=2048, stream=stream, clock=FakeClock(0, 4.0))
    progress.start()
    stream.truncate(0)
    stream.seek(0)

    progress.update(bytes_processed=1024, chunks_embedded=7)

    out = stream.getvalue()
    assert out.startswith("\x1b]9;4;1;50\x07\r")
    assert _strip_ansi(out.split("\r", 1)[1]) == (
        "█" * 15 + "░" * 15 + "  50% input 7 chunks · 1.0 KB/2.0 KB input · 256 B/s · ETA 4s   "
    )


def test_tty_update_shows_placeholders_before_two_seconds():
    stream = FakeTerminal(tty=True)
    progress = EmbedProgress(total_bytes=100, stream=stream, clock=FakeClock(0, 1.0))
    progress.start()

    progress.update(bytes_processed=0, chunks_embedded=0)

    line = _strip_ansi(stream.getvalue().split("\r")[-1])
    assert "0 chunks · 0 B/100 B input · .../s · ETA ..." in line


def test_tty_update_ignored_when_there_are_no_input_bytes():
    stream = FakeTerminal(tty=True)
    progress = EmbedProgress(total_bytes=0, stream=stream, clock=FakeClock(0, 1))
    progress.start()
    before = stream.getvalue()

    progress.update(bytes_processed=0, chunks_embedded=0)

    assert stream.getvalue() == before


def test_tty_finish_clears_osc_and_shows_cursor_once():
    stream = FakeTerminal(tty=True)
    progress = EmbedProgress(total_bytes=10, stream=stream, clock=FakeClock(0))
    progress.start()
    stream.truncate(0)
    stream.seek(0)

    progress.finish()
    progress.finish()

    assert stream.getvalue() == "\x1b]9;4;0\x07\x1b[?25h"


def test_context_manager_restores_cursor_on_interrupt():
    stream = FakeTerminal(tty=True)
    with pytest.raises(KeyboardInterrupt):
        with EmbedProgress(total_bytes=10, stream=stream, clock=FakeClock(0)):
            raise KeyboardInterrupt
    assert stream.getvalue().endswith("\x1b]9;4;0\x07\x1b[?25h")
