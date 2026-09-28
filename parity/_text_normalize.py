"""Normalizes captured CLI output so Node qmd's and pyqmd's text can be
compared exactly, hiding only run-specific noise: color/OSC escapes,
in-place "\r" redraws (compared as line breaks), temp-dir paths, durations, relative times, docids, and the command name
in hint text (`qmd …` vs `pyqmd …`). Everything else -- wording, line
order, leading/interior blank lines -- is part of the format and is kept.
Used identically on both sides by parity/test_structural.py's
test_cli_flow_step_text_matches_node. See docs/specs/
2026-09-24-cli-flow-output-text-parity-design.md."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_OSC_RE = re.compile(r"\x1b\][^\x07]*\x07")
_AGO_RE = re.compile(r"\b\d+\s*[a-z]+ ago\b")
_DURATION_RES = (
    re.compile(r"\b\d+h \d+m\b"),
    re.compile(r"\b\d+m \d+s\b"),
    re.compile(r"\b\d+(?:\.\d+)?(?:ms|s)\b"),
)
_DOCID_RE = re.compile(r"#[0-9a-f]{6}\b")
_SUBCOMMANDS = (
    "collection|context|embed|update|status|cleanup|search|vsearch|query|get|"
    "multi-get|ls|mcp|pull|skill|doctor|trust"
)
_CMD_RE = re.compile(rf"\b(?:py)?qmd (?=(?:{_SUBCOMMANDS})\b)")


@dataclass(frozen=True)
class TextSub:
    """A declared, deliberate Node/pyqmd output difference: a regex
    (re.MULTILINE) substitution applied to both sides' normalized text.
    `reason` is required -- every declared difference must say why."""

    pattern: str
    replacement: str
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(f"TextSub({self.pattern!r}) needs a non-empty reason")
        try:
            re.compile(self.pattern, re.MULTILINE)
        except re.error as exc:
            raise ValueError(
                f"TextSub pattern {self.pattern!r} is not a valid regex: {exc}"
            ) from exc


def _path_variants(path: str) -> list[str]:
    """macOS's /var and /tmp are symlinks into /private, so the same temp
    dir can be printed either way depending on whether a side resolved it."""
    variants = [path]
    if path.startswith("/private/"):
        variants.append(path[len("/private") :])
    elif path.startswith(("/var/", "/tmp/")):
        variants.append("/private" + path)
    return variants


def replace_paths(text: str, placeholders: dict[str, str]) -> str:
    """Replaces each run-specific path (and its /private variant) with its
    placeholder, longest first, touching nothing else. Used on its own by
    capture_node_snapshots.py so committed raw captures never contain a
    real (personal) path, and as normalize()'s path step."""
    pairs = [
        (variant, placeholder)
        for path, placeholder in placeholders.items()
        for variant in _path_variants(path)
    ]
    for variant, placeholder in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        text = text.replace(variant, placeholder)
    return text


def normalize(text: str, placeholders: dict[str, str], text_subs: Sequence[TextSub] = ()) -> str:
    text = _OSC_RE.sub("", text)
    text = _ANSI_RE.sub("", text)
    # In-place redraws ("\r<bar> 100%") compare as ordinary line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = replace_paths(text, placeholders)

    # Relative times before durations, so "0s ago" becomes <AGO> rather
    # than "<DURATION> ago".
    text = _AGO_RE.sub("<AGO>", text)
    for duration_re in _DURATION_RES:
        text = duration_re.sub("<DURATION>", text)
    text = _DOCID_RE.sub("#<DOCID>", text)
    text = _CMD_RE.sub("<CMD> ", text)

    # Declared differences run before the whitespace step, so a sub that
    # deletes a whole line doesn't leave a stray trailing newline behind.
    for sub in text_subs:
        text = re.sub(sub.pattern, sub.replacement, text, flags=re.MULTILINE)

    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n")
