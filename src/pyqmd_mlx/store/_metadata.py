"""Frontmatter metadata extraction. Documents opt into metadata through a
namespaced Markdown frontmatter block:

    ---
    qmd:
      metadata:
        topics:
          - typescript
          - programming
        status: published
    ---

Ported from the Node reference's `src/metadata.ts`. Pure functions -- no
`self`, no SQL -- matching this codebase's existing `_rrf.py`/`_fts_query.py`
pattern. The raw document is never modified: frontmatter stays part of the
stored, indexed, chunked, and embedded content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MetadataScalar = str | int | float | bool
MetadataScalarArray = list[str] | list[int] | list[float] | list[bool]
MetadataValue = MetadataScalar | MetadataScalarArray
DocumentMetadata = dict


@dataclass
class MetadataExtractionResult:
    """Result of extracting metadata from one document. `error` is set when
    the document opted into `qmd.metadata` but the value was invalid -- the
    document still indexes normally, but is excluded from filtered search
    until the metadata is corrected and re-extracted."""

    metadata: DocumentMetadata
    extraction_version: int
    error: str | None = None


# Bump when extraction/normalization semantics change so existing rows are
# re-extracted on the next `collection add`/future `update`.
METADATA_EXTRACTION_VERSION = 1

# Defensive limits for metadata from untrusted repositories.
METADATA_LIMITS: dict[str, int] = {
    "max_frontmatter_bytes": 64 * 1024,
    "max_keys": 64,
    "max_key_bytes": 128,
    "max_string_length": 1024,
    "max_array_length": 128,
    "max_yaml_alias_count": 100,
    "max_error_length": 200,
}

_FRONTMATTER_FILE_EXTENSIONS = {".md", ".markdown", ".mdx"}

_OPEN_RE = re.compile(r"^---[ \t]*\r?\n")
_CLOSE_RE = re.compile(r"^(?:---|\.\.\.)[ \t]*(?:\r?\n|$)", re.MULTILINE)


def extract_document_metadata(content: str, path: str) -> MetadataExtractionResult:
    """Extract `qmd.metadata` from a document's leading YAML frontmatter.
    Never raises: a document without frontmatter, without the `qmd`
    namespace, or with a non-frontmatter extension yields empty metadata
    with no error. Invalid frontmatter or invalid metadata yields empty
    metadata plus a bounded extraction error."""

    def success(metadata: DocumentMetadata) -> MetadataExtractionResult:
        return MetadataExtractionResult(
            metadata=metadata, extraction_version=METADATA_EXTRACTION_VERSION
        )

    def failure(message: str) -> MetadataExtractionResult:
        return MetadataExtractionResult(
            metadata={},
            extraction_version=METADATA_EXTRACTION_VERSION,
            error=_truncate_error_message(message),
        )

    if not _has_frontmatter_file_extension(path):
        return success({})

    frontmatter_yaml = _get_frontmatter_yaml(content)
    if frontmatter_yaml is None:
        return success({})

    if len(frontmatter_yaml.encode("utf-8")) > METADATA_LIMITS["max_frontmatter_bytes"]:
        return failure(f"frontmatter exceeds {METADATA_LIMITS['max_frontmatter_bytes']} bytes")

    import yaml

    try:
        frontmatter = _parse_frontmatter_yaml(frontmatter_yaml)
    except yaml.YAMLError as exc:
        return failure(f"invalid frontmatter YAML: {exc}")

    if not isinstance(frontmatter, dict):
        return success({})

    if "qmd" not in frontmatter:
        return success({})
    qmd_namespace = frontmatter["qmd"]
    if not isinstance(qmd_namespace, dict):
        return failure("frontmatter 'qmd' must be a mapping")

    if "metadata" not in qmd_namespace:
        return success({})
    raw_metadata = qmd_namespace["metadata"]
    if not isinstance(raw_metadata, dict):
        return failure("frontmatter 'qmd.metadata' must be a mapping")

    try:
        return success(_normalize_metadata(raw_metadata))
    except ValueError as exc:
        return failure(str(exc))


def _has_frontmatter_file_extension(path: str) -> bool:
    dot_index = path.rfind(".")
    if dot_index < 0:
        return False
    return path[dot_index:].lower() in _FRONTMATTER_FILE_EXTENSIONS


def _get_frontmatter_yaml(content: str) -> str | None:
    """Slice the YAML between a leading `---` line and a closing `---`/`...`
    line. Tolerates a UTF-8 BOM and CRLF line endings. Returns None when the
    document has no complete leading frontmatter block."""
    body = content[1:] if content.startswith("﻿") else content

    open_match = _OPEN_RE.match(body)
    if not open_match:
        return None

    yaml_start = open_match.end()
    close_match = _CLOSE_RE.search(body, yaml_start)
    if not close_match:
        return None

    return body[yaml_start : close_match.start()]


def _parse_frontmatter_yaml(frontmatter_yaml: str):
    from ._yaml_alias_guard import safe_load_with_alias_limit

    return safe_load_with_alias_limit(frontmatter_yaml, METADATA_LIMITS["max_yaml_alias_count"])


def _normalize_metadata(raw_metadata: dict) -> DocumentMetadata:
    keys = list(raw_metadata.keys())
    if len(keys) > METADATA_LIMITS["max_keys"]:
        raise ValueError(f"metadata has {len(keys)} keys (max {METADATA_LIMITS['max_keys']})")

    result: DocumentMetadata = {}
    for key in keys:
        _validate_metadata_key(key)
        result[key] = _normalize_metadata_value(key, raw_metadata[key])
    return result


def _validate_metadata_key(key) -> None:
    # PyYAML's SafeLoader resolves some unquoted mapping keys to non-str
    # types (the "Norway problem": yes/no/true/false -> bool, bare numerals
    # -> int) -- guard here so that reaches a bounded failure() instead of
    # an uncaught TypeError from len()/encode()/re.search() below.
    if not isinstance(key, str):
        raise ValueError(f"metadata keys must be strings, got {type(key).__name__}")
    if len(key) == 0:
        raise ValueError("metadata keys must be non-empty strings")
    if len(key.encode("utf-8")) > METADATA_LIMITS["max_key_bytes"]:
        raise ValueError(f"metadata key exceeds {METADATA_LIMITS['max_key_bytes']} bytes")
    if re.search(r"[\x00-\x1f\x7f]", key):
        raise ValueError("metadata keys must not contain control characters")


def _normalize_metadata_value(key: str, raw_value) -> MetadataValue:
    if isinstance(raw_value, list):
        return _normalize_metadata_array(key, raw_value)
    return _normalize_metadata_scalar(key, raw_value)


def _normalize_metadata_scalar(key: str, raw_value) -> MetadataScalar:
    # bool must be checked before int/float -- bool is an int subclass in Python.
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        if len(raw_value) > METADATA_LIMITS["max_string_length"]:
            raise ValueError(
                f'metadata key "{key}": string exceeds {METADATA_LIMITS["max_string_length"]} characters'
            )
        return raw_value
    if isinstance(raw_value, (int, float)):
        if isinstance(raw_value, float) and not _is_finite(raw_value):
            raise ValueError(f'metadata key "{key}": numbers must be finite')
        return raw_value
    if raw_value is None:
        raise ValueError(f'metadata key "{key}": null is not supported -- omit the key instead')
    raise ValueError(
        f'metadata key "{key}": unsupported value type -- use strings, numbers, booleans, '
        "or flat arrays of one of those"
    )


def _normalize_metadata_array(key: str, raw_values: list) -> MetadataScalarArray:
    if len(raw_values) == 0:
        raise ValueError(
            f'metadata key "{key}": empty arrays are not supported -- omit the key instead'
        )
    if len(raw_values) > METADATA_LIMITS["max_array_length"]:
        raise ValueError(
            f'metadata key "{key}": array exceeds {METADATA_LIMITS["max_array_length"]} values'
        )

    scalars = []
    for raw_value in raw_values:
        if isinstance(raw_value, list):
            raise ValueError(f'metadata key "{key}": nested arrays are not supported')
        scalars.append(_normalize_metadata_scalar(key, raw_value))

    if all(isinstance(s, str) for s in scalars):
        return list(dict.fromkeys(scalars))
    if all(isinstance(s, bool) for s in scalars):
        return list(dict.fromkeys(scalars))
    if all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scalars):
        return list(dict.fromkeys(scalars))
    raise ValueError(f'metadata key "{key}": mixed-type arrays are not supported')


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def _truncate_error_message(message: str) -> str:
    single_line = " ".join(message.split())
    max_len = METADATA_LIMITS["max_error_length"]
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3] + "..."
