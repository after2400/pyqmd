"""Intent-term extraction, ported from store.ts's extractIntentTerms /
INTENT_STOP_WORDS / INTENT_WEIGHT_CHUNK (store.ts:5316-5352). Pure
functions, no DB access -- shared between Store.query()'s chunk-selection
(this module) and the CLI's snippet-extraction display logic
(pyqmd_mlx.cli._snippet, which imports extract_intent_terms from here rather than
duplicating the stop-word list).
"""

import re

INTENT_WEIGHT_CHUNK = 0.5

INTENT_STOP_WORDS = frozenset(
    {
        # 2-char function words
        "am",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "he",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "so",
        "to",
        "up",
        "us",
        "we",
        # 3-char function words
        "all",
        "and",
        "any",
        "are",
        "but",
        "can",
        "did",
        "for",
        "get",
        "has",
        "her",
        "him",
        "his",
        "how",
        "its",
        "let",
        "may",
        "not",
        "our",
        "out",
        "the",
        "too",
        "was",
        "who",
        "why",
        "you",
        # 4+ char common words
        "also",
        "does",
        "find",
        "from",
        "have",
        "into",
        "more",
        "need",
        "show",
        "some",
        "tell",
        "that",
        "them",
        "this",
        "want",
        "what",
        "when",
        "will",
        "with",
        "your",
        # Search-context noise
        "about",
        "looking",
        "notes",
        "search",
        "where",
        "which",
    }
)

# Strips leading/trailing non-word characters. store.ts uses a Unicode-aware
# \p{L}\p{N} regex; Python's \w is a close-enough substitute for natural-
# language intent strings (the only divergence is \w also treating '_' as a
# word character, which doesn't come up in practice here).
_STRIP_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def extract_intent_terms(intent: str) -> list[str]:
    """Extract meaningful terms from an intent string, filtering stop words
    and punctuation. Returns lowercase terms suitable for text matching."""
    terms = []
    for token in intent.lower().split():
        stripped = _STRIP_PUNCT_RE.sub("", token)
        if len(stripped) > 1 and stripped not in INTENT_STOP_WORDS:
            terms.append(stripped)
    return terms
