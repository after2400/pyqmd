"""AST-aware chunking support via py-tree-sitter, ported from ast.ts.

Language detection, grammar loading/caching, and per-language AST break
point extraction for supported code file types. Every public function
degrades gracefully: unsupported languages, parse failures, or grammar
load failures return an empty list (or a status entry marking the
language unavailable) rather than raising -- callers always fall back to
regex-only chunking. Never prints or logs: pyqmd_mlx/store/ has no
user-facing-output convention (that's pyqmd_mlx/cli/'s job), so a load/parse
failure is only visible through get_ast_status(), consumed by `pyqmd
status`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from os.path import splitext
from typing import Literal

from ._types import BreakPoint

SupportedLanguage = Literal["typescript", "tsx", "javascript", "python", "go", "rust"]

_EXTENSION_MAP: dict[str, SupportedLanguage] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
}


def detect_language(filepath: str) -> SupportedLanguage | None:
    """Detect language from file path extension. Returns None for
    unsupported or unknown extensions (including .md)."""
    _, ext = splitext(filepath)
    return _EXTENSION_MAP.get(ext.lower())


def merge_break_points(
    regex_points: list[BreakPoint], ast_points: list[BreakPoint]
) -> list[BreakPoint]:
    """Merge two BreakPoint lists (regex-derived and AST-derived),
    keeping the higher-scoring entry when both land on the same
    position, sorted by position."""
    from ._chunking import dedup_break_points

    return dedup_break_points([*regex_points, *ast_points])


# =============================================================================
# Grammar Resolution
# =============================================================================


def _load_python() -> object:
    import tree_sitter_python as ts_python

    return ts_python.language()


def _load_typescript() -> object:
    import tree_sitter_typescript as ts_typescript

    return ts_typescript.language_typescript()


def _load_tsx() -> object:
    import tree_sitter_typescript as ts_typescript

    return ts_typescript.language_tsx()


def _load_javascript() -> object:
    import tree_sitter_javascript as ts_javascript

    return ts_javascript.language()


def _load_go() -> object:
    import tree_sitter_go as ts_go

    return ts_go.language()


def _load_rust() -> object:
    import tree_sitter_rust as ts_rust

    return ts_rust.language()


# Order matches Node's GRAMMAR_MAP iteration order (ast.ts) -- purely
# cosmetic (dict iteration order), kept for easy side-by-side comparison.
GRAMMAR_LOADERS: dict[SupportedLanguage, Callable[[], object]] = {
    "typescript": _load_typescript,
    "tsx": _load_tsx,
    "javascript": _load_javascript,
    "python": _load_python,
    "go": _load_go,
    "rust": _load_rust,
}

# Cached compiled Language objects, keyed by language. Populated lazily by
# _load_grammar so a language whose grammar is never needed is never
# imported (each grammar package is a native extension module).
_GRAMMAR_CACHE: dict[SupportedLanguage, object] = {}
# Languages that have already failed to load -- never retried.
_FAILED_LANGUAGES: set[SupportedLanguage] = set()
# Last load error per failed language, for status reporting.
_GRAMMAR_LOAD_ERRORS: dict[SupportedLanguage, str] = {}


def _load_grammar(language: SupportedLanguage) -> object | None:
    """Load and cache a tree_sitter.Language for `language`. Returns None
    (and remembers the failure) if the grammar package is missing or
    raises on load -- never raises itself."""
    if language in _FAILED_LANGUAGES:
        return None
    if language in _GRAMMAR_CACHE:
        return _GRAMMAR_CACHE[language]

    from tree_sitter import Language

    try:
        raw = GRAMMAR_LOADERS[language]()
        grammar = Language(raw)
    except Exception as exc:  # noqa: BLE001 -- must never propagate
        _FAILED_LANGUAGES.add(language)
        _GRAMMAR_LOAD_ERRORS[language] = f"{type(exc).__name__}: {exc}"
        return None

    _GRAMMAR_CACHE[language] = grammar
    return grammar


# =============================================================================
# Byte/char offset conversion
# =============================================================================


def _byte_to_char_offsets(
    content: str, content_bytes: bytes, byte_offsets: set[int]
) -> dict[int, int]:
    """Map each of `byte_offsets` (UTF-8 byte offsets into `content_bytes`)
    to the corresponding character offset into `content`. Every offset
    tree-sitter reports is a token boundary, which is always a valid UTF-8
    character boundary, so slicing content_bytes at these offsets never
    splits a multi-byte character. Pure-ASCII content (len(content) ==
    len(content_bytes)) short-circuits to an identity mapping; otherwise a
    single left-to-right pass decodes each inter-offset slice exactly
    once, giving O(n) total work rather than O(n) per offset."""
    if not byte_offsets:
        return {}
    if len(content) == len(content_bytes):
        return {b: b for b in byte_offsets}

    result: dict[int, int] = {}
    char_count = 0
    prev_byte = 0
    for target in sorted(byte_offsets):
        char_count += len(content_bytes[prev_byte:target].decode("utf-8"))
        result[target] = char_count
        prev_byte = target
    return result


# =============================================================================
# AST Break Point Extraction
# =============================================================================

# Cached per-language tree_sitter.Parser objects, populated lazily next to
# the grammar/query caches. Parser construction is cheap next to parsing
# itself, but reusing one per language avoids churning native objects on
# every query-candidate chunking pass. Single-threaded use only (pyqmd's
# store layer is fully synchronous), so sharing is safe. Tests that
# monkeypatch tree_sitter.Parser must clear this alongside the break-point
# cache -- see _clear_ast_caches.
_PARSER_CACHE: dict[SupportedLanguage, object] = {}

# Memoized get_ast_break_points results, keyed by (sha256(content),
# filepath). Store.query() re-chunks every RRF candidate on every call, so
# without this a repeated query over unchanged files re-runs the full
# native parse each time. Bounded with simple oldest-first eviction; keyed
# by content hash so unchanged files always hit and edited files always
# miss (invalidation falls out of content addressing, the same way
# content_vectors is keyed by content hash).
_AST_BREAK_POINTS_CACHE: dict[tuple[str, str], list[BreakPoint]] = {}
_AST_BREAK_POINTS_CACHE_MAX = 1024


def _clear_ast_caches() -> None:
    """Drop the parser and break-point caches. Test-only seam (lets
    monkeypatched-Parser tests force a fresh parse); never called in
    production code paths."""
    _PARSER_CACHE.clear()
    _AST_BREAK_POINTS_CACHE.clear()


def get_ast_break_points(content: str, filepath: str) -> list[BreakPoint]:
    """Parse `content` (whose language is detected from `filepath`) and
    return break points at AST node boundaries (function/class/etc.
    starts), scored on the same scale as _chunking.py's regex break
    points. Returns [] for unsupported languages, missing/broken
    grammars, or any parse failure -- never raises. Successful results
    are memoized by content hash (see _AST_BREAK_POINTS_CACHE)."""
    from ._ast_queries import SCORE_MAP, get_query
    from ._chunking import dedup_break_points

    language = detect_language(filepath)
    if language is None:
        return []

    grammar = _load_grammar(language)
    if grammar is None:
        return []

    cache_key = (sha256(content.encode("utf-8")).hexdigest(), filepath)
    cached = _AST_BREAK_POINTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from tree_sitter import Parser, QueryCursor

        content_bytes = content.encode("utf-8")
        parser = _PARSER_CACHE.get(language)
        if parser is None:
            parser = Parser(grammar)
            _PARSER_CACHE[language] = parser
        tree = parser.parse(content_bytes)
        query = get_query(language, grammar)
        captures = QueryCursor(query).captures(tree.root_node)

        raw: list[tuple[int, int, str]] = []
        for capture_name, nodes in captures.items():
            score = SCORE_MAP.get(capture_name, 20)
            kind = f"ast:{capture_name}"
            for node in nodes:
                raw.append((node.start_byte, score, kind))

        byte_offsets = {byte_pos for byte_pos, _score, _kind in raw}
        offset_map = _byte_to_char_offsets(content, content_bytes, byte_offsets)

        points = [
            BreakPoint(pos=offset_map[byte_pos], score=score, type=kind)
            for byte_pos, score, kind in raw
        ]
        result = dedup_break_points(points)
        if len(_AST_BREAK_POINTS_CACHE) >= _AST_BREAK_POINTS_CACHE_MAX:
            _AST_BREAK_POINTS_CACHE.pop(next(iter(_AST_BREAK_POINTS_CACHE)))
        _AST_BREAK_POINTS_CACHE[cache_key] = result
        return result
    except Exception:  # noqa: BLE001 -- must never propagate, fall back to regex
        return []


# =============================================================================
# Health / Status
# =============================================================================


@dataclass
class LanguageStatus:
    language: SupportedLanguage
    available: bool
    error: str | None = None


@dataclass
class ASTStatus:
    available: bool
    languages: list[LanguageStatus] = field(default_factory=list)


# One tiny sample per language, each containing at least one AST node the
# language's query captures. get_ast_status() parses these (not just
# compiling the query) so the health check exercises the same
# parse→query→offset-map path get_ast_break_points() uses on real files --
# a bug inside that path (byte/char mapping, native-extension issue on
# specific input) shows up as "unavailable" instead of silently degrading
# to regex-only chunking at query time.
_STATUS_SAMPLES: dict[SupportedLanguage, tuple[str, str]] = {
    "typescript": ("sample.ts", "import x from 'y';\nfunction foo() {}\n"),
    "tsx": ("sample.tsx", "export function App() {\n  return 1;\n}\n"),
    "javascript": ("sample.js", "function foo() {}\n"),
    "python": ("sample.py", "import os\n\ndef foo():\n    pass\n"),
    "go": ("sample.go", 'package main\n\nfunc main() {\n\tprintln("hi")\n}\n'),
    "rust": ("sample.rs", "fn main() {}\n"),
}


def get_ast_status() -> ASTStatus:
    """Check which tree-sitter grammars are available. Attempts to load
    (and compile the query for) every supported language, then parses a
    small canned sample through the real get_ast_break_points() path --
    a language whose sample yields no break points is reported
    unavailable, since real files would silently fall back to regex-only
    chunking for it."""
    from ._ast_queries import get_query

    languages: list[LanguageStatus] = []
    for language in GRAMMAR_LOADERS:
        grammar = _load_grammar(language)
        if grammar is None:
            languages.append(
                LanguageStatus(
                    language=language,
                    available=False,
                    error=_GRAMMAR_LOAD_ERRORS.get(language, "grammar failed to load"),
                )
            )
            continue
        try:
            get_query(language, grammar)
            sample_path, sample_content = _STATUS_SAMPLES[language]
            if not get_ast_break_points(sample_content, sample_path):
                raise RuntimeError("sample parse produced no break points")
        except Exception as exc:  # noqa: BLE001 -- must never propagate
            languages.append(
                LanguageStatus(
                    language=language, available=False, error=f"{type(exc).__name__}: {exc}"
                )
            )
            continue
        languages.append(LanguageStatus(language=language, available=True))

    return ASTStatus(available=any(lang.available for lang in languages), languages=languages)
