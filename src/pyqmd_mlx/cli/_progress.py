"""`embed`'s progress display, ported from Node qmd's vectorIndex
(src/cli/qmd.ts): a live, in-place bar on stderr plus OSC 9;4 terminal
progress (tab/taskbar indicators in terminals that support it), both only
when stderr is a TTY, and a final 100% bar line that `embed` prints to
stdout unconditionally, as Node does.

Progress is measured in input bytes, not documents or chunks, so one huge
document doesn't make the bar jump. It advances between documents: each
document's chunks are embedded in one blocking MLX call.

Unlike Node, which writes its cursor hide/show codes to stderr even when
it isn't a TTY, EmbedProgress writes nothing at all to a non-TTY stream."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from time import monotonic
from typing import TextIO

from pyqmd_mlx.cli._format import format_bytes, format_eta, js_round
from pyqmd_mlx.cli._theme import bold, cyan, dim, green

BAR_WIDTH = 30


def render_bar(percent: float, width: int = BAR_WIDTH) -> str:
    """Node's renderProgressBar: `width` cells, filled cells rounded half-up."""
    filled = js_round(percent / 100 * width)
    return "█" * filled + "░" * (width - filled)


def final_bar_line() -> str:
    """The 100% line Node prints to stdout once embedding finishes -- a
    leading \\r so it overwrites the live stderr line when both streams are
    the same terminal, and trailing spaces to blank out its leftovers."""
    return f"\r{green(render_bar(100))} {bold('100%')}" + " " * 36


class EmbedProgress:
    """Owns everything embed's progress display writes to the terminal.
    Use as a context manager (or start()/finish()) so the cursor and the
    terminal's progress indicator are restored even on Ctrl-C."""

    def __init__(
        self,
        total_bytes: int,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._total = total_bytes
        self._stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._start_time = 0.0
        self._active = False

    def __enter__(self) -> EmbedProgress:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def start(self) -> None:
        if not self._tty:
            return
        self._start_time = self._clock()
        self._active = True
        self._write("\x1b[?25l\x1b]9;4;3\x07")

    def update(self, bytes_processed: int, chunks_embedded: int) -> None:
        if not self._active or self._total == 0:
            return
        percent = min(100.0, bytes_processed / self._total * 100)
        elapsed = self._clock() - self._start_time
        bytes_per_sec = bytes_processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self._total - bytes_processed)
        eta_sec = remaining / bytes_per_sec if bytes_per_sec > 0 else math.inf

        throughput = f"{format_bytes(int(bytes_per_sec))}/s" if bytes_per_sec > 0 else ".../s"
        eta = format_eta(eta_sec) if elapsed > 2 and math.isfinite(eta_sec) else "..."
        input_str = f"{format_bytes(bytes_processed)}/{format_bytes(self._total)} input"
        details = f"{chunks_embedded:,} chunks · {input_str} · {throughput} · ETA {eta}"
        percent_str = str(js_round(percent)).rjust(3)

        self._write(
            f"\x1b]9;4;1;{js_round(percent)}\x07"
            f"\r{cyan(render_bar(percent))} {bold(f'{percent_str}% input')} {dim(details)}   "
        )

    def finish(self) -> None:
        if not self._active:
            return
        self._active = False
        self._write("\x1b]9;4;0\x07\x1b[?25h")
