"""FTS5 query building and CJK normalization, ported from store.ts's
buildFTS5Query / sanitizeFTS5Term / splitFTS5CompoundTerm /
normalizeCjkForFTS / sanitizeFTS5Phrase.
"""

import re
import unicodedata

# Explicit code-point ranges for Han, Hiragana, Katakana, Hangul -- covers
# the practical range of CJK text without the third-party `regex` package's
# \p{Script=...} syntax, which stdlib `re` doesn't support.
_CJK_CHAR_PATTERN = re.compile(
    r"[㐀-䶿一-鿿豈-﫿"  # Han (+ extension A, compat)
    r"぀-ゟ"  # Hiragana
    r"゠-ヿ"  # Katakana
    r"가-힣ᄀ-ᇿ]"  # Hangul syllables + Jamo
)
_CJK_RUN_PATTERN = re.compile(_CJK_CHAR_PATTERN.pattern + "+")

_SANITIZE_KEEP_RE = re.compile(r"[^\w']", re.UNICODE)
_FTS5_SEPARATOR_RUN = re.compile(r"[^\w']+", re.UNICODE)


def sanitize_fts5_term(term: str) -> str:
    """Strip everything except letters/digits/underscore/apostrophe, lowercase."""
    return unicodedata.normalize("NFC", _SANITIZE_KEEP_RE.sub("", term)).lower()


def contains_cjk(text: str) -> bool:
    return _CJK_CHAR_PATTERN.search(text) is not None


def normalize_cjk_for_fts(text: str) -> str:
    """Space out runs of CJK characters so the FTS5 tokenizer treats each
    character as its own token (matches store.ts's normalizeCjkForFTS)."""
    return _CJK_RUN_PATTERN.sub(lambda m: " " + " ".join(m.group()) + " ", text)


def _split_fts5_compound_term(term: str) -> list[str]:
    """Split a term the way the FTS5 tokenizer splits document text, and
    sanitize each part. 'PIO-1384' -> ['pio', '1384']."""
    parts = _FTS5_SEPARATOR_RUN.split(term)
    return [p for p in (sanitize_fts5_term(p) for p in parts) if p]


def _sanitize_fts5_phrase(phrase: str) -> str:
    normalized = normalize_cjk_for_fts(phrase)
    parts: list[str] = []
    for token in normalized.split():
        parts.extend(_split_fts5_compound_term(token))
    return " ".join(parts)


def build_fts5_query(query: str) -> str | None:
    """Parse lex query syntax (quoted phrases, negation, compound terms,
    plain prefix terms) into an FTS5 MATCH expression."""
    positive: list[str] = []
    negative: list[str] = []

    s = query.strip()
    i = 0
    n = len(s)

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break

        negated = s[i] == "-"
        if negated:
            i += 1

        if i < n and s[i] == '"':
            start = i + 1
            i += 1
            while i < n and s[i] != '"':
                i += 1
            phrase = s[start:i].strip()
            i += 1  # skip closing quote
            if phrase:
                sanitized = _sanitize_fts5_phrase(phrase)
                if sanitized:
                    fts_phrase = f'"{sanitized}"'
                    (negative if negated else positive).append(fts_phrase)
        else:
            start = i
            while i < n and not s[i].isspace() and s[i] != '"':
                i += 1
            term = s[start:i]

            if contains_cjk(term):
                sanitized = _sanitize_fts5_phrase(term)
                if sanitized:
                    fts_phrase = f'"{sanitized}"'
                    (negative if negated else positive).append(fts_phrase)
            else:
                parts = _split_fts5_compound_term(term)
                if parts:
                    fts_term = f'"{" ".join(parts)}"' if len(parts) > 1 else f'"{parts[0]}"*'
                    (negative if negated else positive).append(fts_term)

    if not positive:
        return None  # FTS5 NOT is binary -- can't search with only negative terms

    result = " AND ".join(positive)
    for neg in negative:
        result = f"{result} NOT {neg}"
    return result
