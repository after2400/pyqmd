"""Shared plain-data shapes used across pyqmd_mlx.store, ported from store.ts's
type definitions (BreakPoint, CodeFenceRegion, RankedResult, SearchResult,
HybridQueryResult).
"""

from dataclasses import dataclass, field


@dataclass
class BreakPoint:
    pos: int  # character position
    score: int  # base score (higher = better break point)
    type: str  # for debugging: 'h1', 'h2', 'blank', etc.


@dataclass
class CodeFenceRegion:
    start: int  # position of opening ```
    end: int  # position of closing ``` (only paired fences produce regions;
    # a lone unmatched marker is ignored -- see find_code_fences)


@dataclass
class RankedResult:
    file: str
    display_path: str
    title: str
    body: str
    score: float


@dataclass
class RankedListMeta:
    source: str  # "fts" | "vec"
    query_type: str  # "original" | "lex" | "vec" | "hyde"
    query: str


@dataclass
class ExpandedQueryPart:
    type: str  # "lex" | "vec" | "hyde"
    query: str


@dataclass
class SearchResult:
    filepath: str
    display_path: str
    title: str
    hash: str
    docid: str
    collection_name: str
    body: str
    score: float
    source: str  # "fts" | "vec"
    chunk_pos: int | None = None
    context: str | None = None


@dataclass
class HybridQueryResult:
    file: str
    display_path: str
    title: str
    body: str
    best_chunk: str
    best_chunk_pos: int
    score: float
    context: str | None
    docid: str
    metadata: dict = field(default_factory=dict)
