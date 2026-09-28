"""Shared helpers for deriving scenario inputs (a sample query, known
document filenames, a verbatim sentence) from the active DatasetProfile.
Used by both cli_scenarios.py and mcp_scenarios.py so scenario definitions
never hardcode a specific dataset's query terms or filenames (a 2026-09-13 parity-suite review finding).
"""

from __future__ import annotations

import re
from pathlib import Path

from parity._queries import load_queries
from parity.dataset_profile import DatasetProfile

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def sample_query(profile: DatasetProfile) -> str:
    queries = load_queries(profile)
    return queries[0]["query"] if queries else "test"


def known_docs(profile: DatasetProfile) -> tuple[str, str]:
    """The first two markdown filenames in the corpus, sorted for
    determinism. Falls back to repeating the only document if the corpus
    has just one."""
    names = sorted(p.name for p in profile.corpus_dir.glob("*.md"))
    if not names:
        return "missing-document.md", "missing-document.md"
    doc_a = names[0]
    doc_b = names[1] if len(names) > 1 else doc_a
    return doc_a, doc_b


def glob_pattern_for(doc_name: str) -> str:
    return f"{Path(doc_name).stem}*.md"


def verbatim_sentence(profile: DatasetProfile, doc_name: str) -> str:
    """A single real sentence of body text from the named document,
    verbatim -- used as a query that should deterministically rank that
    document first regardless of embedding backend. Skips markdown
    headings; splits body lines into sentences rather than using a whole
    line verbatim, since a corpus line can be an entire unwrapped
    paragraph."""
    doc_path = profile.corpus_dir / doc_name
    for line in doc_path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        for sentence in _SENTENCE_BOUNDARY.split(text):
            if len(sentence.split()) >= 6:
                return sentence
    return doc_path.stem
