"""Shared ':line'/':line:count' suffix parsing for document identifiers --
used by the CLI's `get` command and the MCP `get` tool, so both surfaces
parse 'file.md:100' / 'file.md:100:40' identically.
"""

import re


def parse_line_range(
    identifier: str, from_line: int | None, max_lines: int | None
) -> tuple[str, int | None, int | None]:
    """Parse an optional ':line' or ':line:count' suffix off `identifier`.
    Explicit from_line/max_lines arguments always win over values parsed
    from the identifier. Returns (bare_identifier, from_line, max_lines)."""
    range_match = re.search(r":(\d+):(\d+)$", identifier)
    if range_match:
        if from_line is None:
            from_line = int(range_match.group(1))
        if max_lines is None:
            max_lines = int(range_match.group(2))
        identifier = identifier[: -len(range_match.group(0))]
    else:
        colon_match = re.search(r":(\d+)$", identifier)
        if colon_match:
            if from_line is None:
                from_line = int(colon_match.group(1))
            identifier = identifier[: -len(colon_match.group(0))]
    if from_line is not None:
        from_line = max(1, from_line)
    return identifier, from_line, max_lines
