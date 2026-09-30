"""Regex/heading-aware chunking, ported from store.ts's smart-chunking
functions. AST-aware chunking (tree-sitter, roadmap #5) adds
function/class-boundary break points via pyqmd_mlx.store._ast when
chunk_document() is called with chunk_strategy="auto".

Two sizes, as in Node: embedding_chunks() (what gets embedded, 2700-char
chunks) and chunk_document()'s defaults (query-time best-chunk selection,
3600-char chunks).

Differences from Node, deliberate (docs/specs/2026-09-29-embedding-input-
parity-design.md, section 8):
- a cut or an overlap start that lands inside a code fence moves forward
  to the fence's end (Node can cut inside a fence);
- a trailing unmatched ``` is ignored (Node treats it as a fence to end of
  file);
- when the overlap would stall, the next chunk starts halfway through the
  previous one (Node starts it at the previous end, with no overlap).
And one not yet ported: Node re-splits an embed-time chunk that turns out
longer than 900 real tokens (chunkDocumentByTokensWithLlm); pyqmd keeps
the character estimate.
"""

import re
from typing import Literal

from ._types import BreakPoint, CodeFenceRegion

CHUNK_SIZE_TOKENS = 900
CHUNK_OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * 0.15)  # 135 tokens

# Char-based approximation (~4 chars per token), same as store.ts's sync path.
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4  # 3600
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4  # 540
CHUNK_WINDOW_TOKENS = 200
CHUNK_WINDOW_CHARS = CHUNK_WINDOW_TOKENS * 4  # 800

# Embed-time chunking (store.ts's chunkDocumentByTokensWithLlm) estimates 3
# chars per token (prose ~4, code ~2), so embedding chunks are smaller than
# query-time ones.
EMBED_CHARS_PER_TOKEN = 3
EMBED_CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * EMBED_CHARS_PER_TOKEN  # 2700
EMBED_CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * EMBED_CHARS_PER_TOKEN  # 405
EMBED_CHUNK_WINDOW_CHARS = CHUNK_WINDOW_TOKENS * EMBED_CHARS_PER_TOKEN  # 600

VALID_CHUNK_STRATEGIES = ("regex", "auto")


def validate_chunk_strategy(chunk_strategy: str) -> str:
    """Runtime-check a `chunk_strategy` value. The `Literal["regex",
    "auto"]` annotations on chunk_document()/Store.index_content()/
    Store.query() are type-checker hints only -- any non-CLI caller (a
    script, a test, a future MCP tool) bypasses the CLI's Typer enum
    validation, and without this check a typo like "Auto" would silently
    fall back to regex chunking. Raises ValueError on anything else."""
    if chunk_strategy not in VALID_CHUNK_STRATEGIES:
        raise ValueError(f'Invalid chunk_strategy: {chunk_strategy!r} (expected "regex" or "auto")')
    return chunk_strategy


def dedup_break_points(points: list[BreakPoint]) -> list[BreakPoint]:
    """Keep the highest-scoring BreakPoint per position (first wins ties),
    sorted by position. Shared by scan_break_points, _ast.merge_break_points,
    and _ast.get_ast_break_points so the tie-breaking rule lives in one
    place."""
    seen: dict[int, BreakPoint] = {}
    for bp in points:
        existing = seen.get(bp.pos)
        if existing is None or bp.score > existing.score:
            seen[bp.pos] = bp
    return sorted(seen.values(), key=lambda bp: bp.pos)


# (pattern, score, type). Order matters only for readability -- scanning
# collects every match and keeps the highest score at each position.
BREAK_PATTERNS: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"\n#{1}(?!#)"), 100, "h1"),
    (re.compile(r"\n#{2}(?!#)"), 90, "h2"),
    (re.compile(r"\n#{3}(?!#)"), 80, "h3"),
    (re.compile(r"\n#{4}(?!#)"), 70, "h4"),
    (re.compile(r"\n#{5}(?!#)"), 60, "h5"),
    (re.compile(r"\n#{6}(?!#)"), 50, "h6"),
    (re.compile(r"\n```"), 80, "codeblock"),
    (re.compile(r"\n(?:---|\*\*\*|___)\s*\n"), 60, "hr"),
    (re.compile(r"\n\n+"), 20, "blank"),
    (re.compile(r"\n[-*]\s"), 5, "list"),
    (re.compile(r"\n\d+\.\s"), 5, "numlist"),
    (re.compile(r"\n"), 1, "newline"),
]


def scan_break_points(text: str) -> list[BreakPoint]:
    """Scan text for potential break points, keeping the highest score at
    each position when multiple patterns match the same spot."""
    points = [
        BreakPoint(pos=match.start(), score=score, type=kind)
        for pattern, score, kind in BREAK_PATTERNS
        for match in pattern.finditer(text)
    ]
    return dedup_break_points(points)


_FENCE_PATTERN = re.compile(r"\n```")


def find_code_fences(text: str) -> list[CodeFenceRegion]:
    """Find code fence regions (between paired ``` markers) that must never
    be split. A trailing unmatched marker is ignored rather than treated
    as a fence running to end-of-file: a lone ``` (e.g. an unbalanced
    example in a docstring) would otherwise swallow every candidate cut
    point for the rest of the document and collapse the whole file into a
    single chunk. Deliberate divergence from Node's store.ts, which extends
    an unclosed fence to EOF -- the safer default here is to assume a lone
    marker is not a real fence pair."""
    regions: list[CodeFenceRegion] = []
    in_fence = False
    fence_start = 0
    for match in _FENCE_PATTERN.finditer(text):
        if not in_fence:
            fence_start = match.start()
            in_fence = True
        else:
            regions.append(CodeFenceRegion(start=fence_start, end=match.end()))
            in_fence = False
    # A trailing unmatched marker opens no region (see docstring).
    return regions


def is_inside_code_fence(pos: int, fences: list[CodeFenceRegion]) -> bool:
    return any(f.start < pos < f.end for f in fences)


def _extend_past_fence(pos: int, code_fences: list[CodeFenceRegion], limit: int) -> int:
    """If pos lands inside a code fence, push it forward to that fence's end
    (never backward to the fence's start -- clamping backward is what causes
    a cascade of near-empty sliver chunks when a fence is large relative to
    the surrounding text). Allowed to make one chunk exceed max_chars: never
    splitting a fence outranks the size target."""
    for fence in code_fences:
        if fence.start < pos < fence.end:
            return min(fence.end, limit)
    return pos


def find_best_cutoff(
    break_points: list[BreakPoint],
    target_char_pos: int,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = 0.7,
    code_fences: list[CodeFenceRegion] = (),
) -> int:
    """Find the best cut position using scored break points with squared-
    distance decay: gentle early, steep late, so a distant heading can still
    beat a low-quality break point right at the target."""
    window_start = target_char_pos - window_chars
    best_score = -1.0
    best_pos = target_char_pos

    for bp in break_points:
        if bp.pos < window_start:
            continue
        if bp.pos > target_char_pos:
            break  # sorted by position, so nothing further can qualify
        if is_inside_code_fence(bp.pos, code_fences):
            continue

        distance = target_char_pos - bp.pos
        normalized_dist = distance / window_chars if window_chars else 0
        multiplier = 1.0 - (normalized_dist * normalized_dist) * decay_factor
        final_score = bp.score * multiplier

        if final_score > best_score:
            best_score = final_score
            best_pos = bp.pos

    return best_pos


def chunk_document(
    content: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS,
    filepath: str | None = None,
    chunk_strategy: Literal["regex", "auto"] = "regex",
) -> list[tuple[str, int]]:
    """Split content into (text, pos) chunks at scored break points, never
    inside a code fence, with overlap between consecutive chunks.

    Python strings are sequences of Unicode code points, not UTF-16 code
    units, so the UTF-16 surrogate-pair boundary adjustment in store.ts
    (adjustSurrogateBoundary) has no equivalent problem here — slicing a
    Python str can never split an astral-plane character, so that safeguard
    is correctly omitted, not dropped by oversight.

    When chunk_strategy="auto" and filepath is given, AST break points
    (function/class/etc. boundaries for supported code languages) are
    merged in as extra candidates alongside the regex break points --
    see pyqmd_mlx.store._ast. Falls back to regex-only when chunk_strategy is
    "regex" (the default), filepath is None, the file's language is
    unsupported, or AST extraction returns nothing.

    Raises ValueError for any chunk_strategy other than "regex"/"auto" --
    see validate_chunk_strategy.
    """
    validate_chunk_strategy(chunk_strategy)
    if len(content) <= max_chars:
        return [(content, 0)]

    break_points = scan_break_points(content)
    if chunk_strategy == "auto" and filepath is not None:
        from ._ast import get_ast_break_points, merge_break_points

        ast_points = get_ast_break_points(content, filepath)
        if ast_points:
            break_points = merge_break_points(break_points, ast_points)

    code_fences = find_code_fences(content)

    chunks: list[tuple[str, int]] = []
    char_pos = 0

    while char_pos < len(content):
        target_end_pos = min(char_pos + max_chars, len(content))
        end_pos = target_end_pos

        if end_pos < len(content):
            best_cutoff = find_best_cutoff(
                break_points, target_end_pos, window_chars, 0.7, code_fences
            )
            if char_pos < best_cutoff <= target_end_pos:
                end_pos = best_cutoff

        # Apply helper: extend forward past any fence
        end_pos = _extend_past_fence(end_pos, code_fences, len(content))

        if end_pos <= char_pos:
            end_pos = min(char_pos + max_chars, len(content))
            # Apply helper again after recomputing
            end_pos = _extend_past_fence(end_pos, code_fences, len(content))

        chunks.append((content[char_pos:end_pos], char_pos))

        if end_pos >= len(content):
            break

        next_pos = end_pos - overlap_chars
        # Apply helper: extend forward past any fence
        next_pos = _extend_past_fence(next_pos, code_fences, len(content))

        if next_pos <= char_pos:
            # When overlap_chars is very large, still create some overlap
            if end_pos > char_pos + 1:
                next_pos = char_pos + max(1, (end_pos - char_pos) // 2)
            else:
                next_pos = end_pos
        char_pos = next_pos

    return chunks


def embedding_chunks(
    content: str, filepath: str | None, chunk_strategy: Literal["regex", "auto"] = "regex"
) -> list[tuple[str, int]]:
    """The chunks Store.index_content embeds: chunk_document() at Node's
    embed-time size. Every embed-side caller goes through here, so the
    stored positions and the already-embedded check always agree."""
    return chunk_document(
        content,
        max_chars=EMBED_CHUNK_SIZE_CHARS,
        overlap_chars=EMBED_CHUNK_OVERLAP_CHARS,
        window_chars=EMBED_CHUNK_WINDOW_CHARS,
        filepath=filepath,
        chunk_strategy=chunk_strategy,
    )
