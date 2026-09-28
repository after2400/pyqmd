"""macOS-only timing helper: wraps a command with `/usr/bin/time -l` to get
wall-clock time and peak RSS from a single subprocess invocation, without
adding a psutil dependency -- consistent with this project's Apple-
Silicon-only scope (see CLAUDE.md). Used by benchmark.py; unrelated to the
structural/quality parity suite.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Not anchored to line-start: a spinner/progress-bar library (e.g. Node's
# `embed`) can leave ANSI cursor-show/hide escapes (\x1b[?25l\x1b[?25h)
# immediately before time's report with no newline in between, which a
# `^`-anchored pattern never matches even though the number itself is
# intact -- found via a real `embed` run whose stderr looked like
# "[?25l[?25h      135.34 real        32.69 user        10.20 sys".
_REAL_RE = re.compile(r"([\d.]+)\s+real\s+[\d.]+\s+user\s+[\d.]+\s+sys")
_RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


@dataclass
class TimedResult:
    exit_code: int
    stdout: str
    stderr: str
    real_seconds: float
    peak_rss_bytes: int


def run_timed(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> TimedResult:
    """Runs `/usr/bin/time -l <cmd>` and parses wall-clock seconds and peak
    RSS (bytes) out of its report, which `time -l` appends to stderr after
    the child process's own stderr output."""
    result = subprocess.run(
        ["/usr/bin/time", "-l", *cmd],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    real_match = _REAL_RE.search(result.stderr)
    rss_match = _RSS_RE.search(result.stderr)
    if not real_match or not rss_match:
        raise RuntimeError(f"could not parse `time -l` output:\n{result.stderr}")
    return TimedResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        real_seconds=float(real_match.group(1)),
        peak_rss_bytes=int(rss_match.group(1)),
    )
