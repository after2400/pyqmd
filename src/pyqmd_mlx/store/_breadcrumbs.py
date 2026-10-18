"""Section-heading breadcrumbs for embedding: the title string a markdown
chunk is embedded under becomes "<doc title> > <h2> > <h3> ...", the path of
the section that holds most of the chunk. Validated by the experiment in
docs/specs/2026-10-15-heading-breadcrumbs-experiment-design.md."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

BREADCRUMBS_ENV_VAR = "PYQMD_EMBED_BREADCRUMBS"
MAX_BREADCRUMB_CHARS = 256

# ATX heading; an optional closing run of hashes needs a space before it, so
# "# C#" keeps its "#".
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*\r?$", re.MULTILINE)


# Opening or closing fence line: up to 3 spaces of indent, then 3+ backticks or tildes.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Heading:
    pos: int
    level: int
    text: str


def _fenced_ranges(content: str) -> list[tuple[int, int]]:
    """Character ranges covered by fenced code blocks, CommonMark-style: ```
    or ~~~ opens, a line of the same character at least as long (and with
    nothing after it) closes, and an unclosed fence runs to the end. This is
    independent of the chunker's fence finder, which only knows Node's
    "\n```" pairs; a `#` line is a heading or not by markdown's rules."""
    ranges: list[tuple[int, int]] = []
    open_start = 0
    marker = ""
    for match in _FENCE_RE.finditer(content):
        run, rest = match.group(1), match.group(2)
        if not marker:
            if run[0] == "`" and "`" in rest:
                continue  # inline code, not a fence
            marker, open_start = run, match.start()
        elif run[0] == marker[0] and len(run) >= len(marker) and not rest.strip():
            ranges.append((open_start, match.end()))
            marker = ""
    if marker:
        ranges.append((open_start, len(content)))
    return ranges


def find_headings(content: str) -> list[Heading]:
    """ATX headings in document order, skipping `#` lines inside code fences
    and headings with no text."""
    fenced = _fenced_ranges(content)
    headings = []
    for match in _HEADING_RE.finditer(content):
        text = match.group(2).strip()
        if text and not any(start <= match.start() < end for start, end in fenced):
            headings.append(Heading(match.start(), len(match.group(1)), text))
    return headings


def heading_path(headings: list[Heading], pos: int) -> list[str]:
    """The headings of the section that contains `pos`, outermost first. A
    heading starting at `pos` opens that section, so it is included. A heading
    pops every entry of the same or deeper level. Level-1 headings are the
    page title (already in the title string) and never appear in the path."""
    stack: list[Heading] = []
    for heading in headings:
        if heading.pos > pos:
            break
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        if heading.level > 1:
            stack.append(heading)
    return [heading.text for heading in stack]


def dominant_heading_path(headings: list[Heading], pos: int, length: int) -> list[str]:
    """The heading path of the section that holds the most characters of the
    chunk `[pos, pos + length)`; a tie goes to the earlier section. Not the
    path at `pos`: a chunk starts EMBED_CHUNK_OVERLAP_CHARS before the cut the
    chunker made at a heading, so its start usually lies in the previous
    section."""
    end = pos + length
    starts = [pos, *(h.pos for h in headings if pos < h.pos < end)]
    best_path: list[str] = []
    best_chars = -1
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end
        if stop - start > best_chars:
            best_chars = stop - start
            best_path = heading_path(headings, start)
    return best_path


def breadcrumb_title(doc_title: str, path: list[str], max_chars: int = MAX_BREADCRUMB_CHARS) -> str:
    """`doc_title` followed by ` > ` and each heading of `path`. The added text
    (everything after the doc title) is capped at `max_chars` characters:
    outermost headings are dropped until it fits, and if the innermost heading
    alone does not fit it is truncated and ends with an ellipsis. A leading
    heading equal to `doc_title` is dropped: extract_title takes the first `#`
    or `##` heading, so a note that opens at `##` would otherwise read
    "Overview > Overview"."""
    separator = " > "
    path = path[1:] if path and path[0] == doc_title else path
    kept = list(path)
    while kept and sum(len(separator) + len(h) for h in kept) > max_chars:
        kept.pop(0)
    if not kept and path:
        room = max_chars - len(separator) - 1
        kept = [path[-1][:room] + "…"]
    return doc_title + "".join(separator + h for h in kept)


def breadcrumbs_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Breadcrumbs are opt-in: True only when PYQMD_EMBED_BREADCRUMBS is 1, true
    or on (any case). Off by default, so existing indexes and Node's embedding
    input are untouched unless a user asks for them."""
    value = (os.environ if environ is None else environ).get(BREADCRUMBS_ENV_VAR)
    return value is not None and value.strip().lower() in ("1", "true", "on")
