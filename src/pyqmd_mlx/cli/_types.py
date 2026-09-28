"""Plain-data shapes for CLI display, decoupled from pyqmd_mlx.store's internal
dataclasses (SearchResult, HybridQueryResult) so the output formatters
don't need to know which one produced a given row -- each command module
maps its Store result onto these before formatting.
"""

from dataclasses import dataclass, field
from enum import Enum


class OutputFormat(str, Enum):
    """Valid `--format` values for search/query/multi-get commands. A str,
    Enum subclass so Typer validates the CLI value itself (rejecting bad
    input with a clean Typer-native error and listing choices in --help)
    instead of the ValueError from format_search_results/format_documents
    ever reaching the user as a raw traceback."""

    CLI = "cli"
    JSON = "json"
    CSV = "csv"
    MD = "md"
    XML = "xml"
    FILES = "files"


class ChunkStrategy(str, Enum):
    """Valid `--chunk-strategy` values for embed/query commands. A str,
    Enum subclass so Typer validates the CLI value itself, matching the
    OutputFormat pattern above."""

    REGEX = "regex"
    AUTO = "auto"


@dataclass
class DisplayResult:
    score: float
    display_path: str
    title: str
    body: str
    chunk_pos: int | None
    docid: str | None = None
    context: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentEntry:
    display_path: str
    title: str
    body: str
    context: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    docid: str | None = None
