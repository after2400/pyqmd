from pyqmd_mlx.store._metadata import METADATA_LIMITS, DocumentMetadata, extract_document_metadata


def test_extract_returns_empty_metadata_for_content_with_no_frontmatter():
    result = extract_document_metadata("# Just a heading\nbody text", "notes/a.md")
    assert result.metadata == {}
    assert result.error is None
    assert result.extraction_version == 1


def test_extract_returns_empty_metadata_for_non_frontmatter_extension():
    result = extract_document_metadata(
        "---\nqmd:\n  metadata:\n    status: x\n---\nbody", "notes/a.txt"
    )
    assert result.metadata == {}
    assert result.error is None


def test_extract_returns_empty_metadata_when_frontmatter_has_no_qmd_namespace():
    content = "---\ntitle: Something\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is None


def test_extract_returns_empty_metadata_when_qmd_namespace_has_no_metadata_key():
    content = "---\nqmd:\n  other: 1\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is None


def test_extract_returns_scalar_metadata():
    content = "---\nqmd:\n  metadata:\n    status: published\n    priority: 3\n    active: true\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"status": "published", "priority": 3, "active": True}
    assert result.error is None


def test_extract_returns_array_metadata():
    content = (
        "---\nqmd:\n  metadata:\n    topics:\n      - typescript\n      - programming\n---\nbody"
    )
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"topics": ["typescript", "programming"]}


def test_extract_handles_utf8_bom_and_crlf():
    content = "﻿---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"status": "ok"}


def test_extract_handles_dotdotdot_close_marker():
    content = "---\nqmd:\n  metadata:\n    status: ok\n...\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"status": "ok"}


def test_extract_returns_none_error_for_no_closing_marker():
    content = "---\nqmd:\n  metadata:\n    status: ok\nno closing marker at all"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is None


def test_document_metadata_is_a_plain_dict_type_alias():
    assert DocumentMetadata == dict  # noqa: E721


def test_extract_fails_on_oversized_frontmatter():
    huge_value = "x" * (METADATA_LIMITS["max_frontmatter_bytes"] + 1)
    content = f"---\nqmd:\n  metadata:\n    filler: {huge_value}\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is not None
    assert "exceeds" in result.error


def test_extract_fails_on_invalid_yaml():
    content = "---\nqmd: [unterminated\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert "invalid frontmatter YAML" in result.error


def test_extract_fails_when_qmd_namespace_is_not_a_mapping():
    content = "---\nqmd: not-a-mapping\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.error == "frontmatter 'qmd' must be a mapping"


def test_extract_fails_when_metadata_is_not_a_mapping():
    content = "---\nqmd:\n  metadata: not-a-mapping\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.error == "frontmatter 'qmd.metadata' must be a mapping"


def test_extract_fails_on_too_many_keys():
    keys = "\n".join(f"    k{i}: {i}" for i in range(METADATA_LIMITS["max_keys"] + 1))
    content = f"---\nqmd:\n  metadata:\n{keys}\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert "max" in result.error


def test_extract_fails_on_oversized_key():
    long_key = "k" * (METADATA_LIMITS["max_key_bytes"] + 1)
    content = f"---\nqmd:\n  metadata:\n    {long_key}: 1\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "exceeds" in result.error


def test_extract_fails_on_control_character_in_key():
    content = '---\nqmd:\n  metadata:\n    "a\\tb": 1\n---\nbody'
    result = extract_document_metadata(content, "notes/a.md")
    assert "control characters" in result.error


def test_extract_fails_on_oversized_string_value():
    long_value = "x" * (METADATA_LIMITS["max_string_length"] + 1)
    content = f'---\nqmd:\n  metadata:\n    note: "{long_value}"\n---\nbody'
    result = extract_document_metadata(content, "notes/a.md")
    assert "exceeds" in result.error


def test_extract_fails_on_null_value():
    content = "---\nqmd:\n  metadata:\n    status: null\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "null is not supported" in result.error


def test_extract_fails_on_unsupported_value_type():
    content = "---\nqmd:\n  metadata:\n    nested:\n      a: 1\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "unsupported value type" in result.error


def test_extract_fails_on_empty_array():
    content = "---\nqmd:\n  metadata:\n    tags: []\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "empty arrays" in result.error


def test_extract_fails_on_oversized_array():
    items = "\n".join(f"      - v{i}" for i in range(METADATA_LIMITS["max_array_length"] + 1))
    content = f"---\nqmd:\n  metadata:\n    tags:\n{items}\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "exceeds" in result.error


def test_extract_fails_on_nested_array():
    content = "---\nqmd:\n  metadata:\n    tags:\n      - [1, 2]\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "nested arrays" in result.error


def test_extract_fails_on_mixed_type_array():
    content = "---\nqmd:\n  metadata:\n    tags:\n      - one\n      - 2\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert "mixed-type arrays" in result.error


def test_extract_array_dedups_while_preserving_order():
    content = "---\nqmd:\n  metadata:\n    tags:\n      - b\n      - a\n      - b\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"tags": ["b", "a"]}


def test_extract_underscore_proto_key_roundtrips_like_any_other_key():
    content = '---\nqmd:\n  metadata:\n    "__proto__": "x"\n---\nbody'
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {"__proto__": "x"}


def test_extract_truncates_long_error_messages():
    content = "---\nqmd: [unterminated " + ("x" * 500) + "\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.error is not None
    assert len(result.error) <= METADATA_LIMITS["max_error_length"]


def test_extract_fails_on_non_string_key_from_yaml_norway_problem():
    # PyYAML's SafeLoader resolves unquoted "yes" as a bool key (the classic
    # YAML "Norway problem"), not a string -- extract_document_metadata must
    # still return a clean failure() rather than raising.
    content = "---\nqmd:\n  metadata:\n    yes: 1\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is not None
    assert "must be strings" in result.error


def test_extract_fails_on_yaml_alias_bomb():
    # Classic exponential alias-expansion pattern, well under
    # max_frontmatter_bytes in raw size -- must be caught by the alias-count
    # limit specifically, not the byte-size limit.
    lines = ['a0: &a0 ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]']
    for i in range(1, 20):
        lines.append(
            f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}, *a{i - 1}, *a{i - 1}, *a{i - 1}, "
            f"*a{i - 1}, *a{i - 1}, *a{i - 1}, *a{i - 1}, *a{i - 1}]"
        )
    bomb_yaml = "\n".join(lines)
    content = f"---\nqmd:\n  metadata:\n    filler: ok\n{bomb_yaml}\n---\nbody"
    result = extract_document_metadata(content, "notes/a.md")
    assert result.metadata == {}
    assert result.error is not None
