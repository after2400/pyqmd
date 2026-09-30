"""Document titles, a mechanical port of store.ts's titleExtractors +
extractTitle (store.ts:2926-2956), quirks included:

- the extension is everything from the path's last "." (lower-cased), so
  "README" has extension "e" and "notes.md/README" has ".md/readme";
- an extractor that returns an empty title falls back to the file name;
- "\\s" crosses line breaks, so "#+TITLE:" followed by a newline captures
  the next line ("#+TITLE:\\n* Heading" is titled "* Heading");
- the file-name fallback strips from the last "." in the whole path, so
  "archive.v2/README" becomes "archive".

JS's ".", and multiline "^"/"$", treat \\r, U+2028 and U+2029 as line
breaks where Python's only treat \\n, so the patterns spell out JS's line
classes instead of using Python's.
"""

import re
from collections.abc import Callable

_JS_BOL = r"(?:\A|(?<=[\n\r  ]))"
_JS_EOL = r"(?=[\n\r  ]|\Z)"
_JS_DOT = r"[^\n\r  ]"

_MD_HEADING_RE = re.compile(rf"{_JS_BOL}##?\s+({_JS_DOT}+){_JS_EOL}")
_MD_H2_RE = re.compile(rf"{_JS_BOL}##\s+({_JS_DOT}+){_JS_EOL}")
_ORG_TITLE_RE = re.compile(rf"{_JS_BOL}#\+TITLE:\s*({_JS_DOT}+){_JS_EOL}", re.IGNORECASE)
_ORG_HEADING_RE = re.compile(rf"{_JS_BOL}\*+\s+({_JS_DOT}+){_JS_EOL}")
_LAST_EXTENSION_RE = re.compile(r"\.[^.]+\Z")


def _md_title(content: str) -> str | None:
    match = _MD_HEADING_RE.search(content)
    if match is None:
        return None
    title = match.group(1).strip()
    if title in ("\U0001f4dd Notes", "Notes"):
        next_match = _MD_H2_RE.search(content)
        if next_match is not None:
            return next_match.group(1).strip()
    return title


def _org_title(content: str) -> str | None:
    title_prop = _ORG_TITLE_RE.search(content)
    if title_prop is not None:
        return title_prop.group(1).strip()
    heading = _ORG_HEADING_RE.search(content)
    if heading is not None:
        return heading.group(1).strip()
    return None


_EXTRACTORS: dict[str, Callable[[str], str | None]] = {".md": _md_title, ".org": _org_title}


def extract_title(content: str, filename: str) -> str:
    """Title for a document at collection-relative path `filename`."""
    # rfind is -1 without a dot, so this is the last character, as JS's
    # slice(-1) is.
    extension = filename[filename.rfind(".") :].lower()
    extractor = _EXTRACTORS.get(extension)
    if extractor is not None:
        title = extractor(content)
        if title:
            return title
    return _LAST_EXTENSION_RE.sub("", filename, count=1).split("/")[-1] or filename
