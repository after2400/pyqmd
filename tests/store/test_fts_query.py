from pyqmd_mlx.store._fts_query import (
    build_fts5_query,
    contains_cjk,
    normalize_cjk_for_fts,
    sanitize_fts5_term,
)


def test_sanitize_fts5_term_lowercases_and_strips_punctuation():
    assert sanitize_fts5_term("Hello!!!") == "hello"
    assert sanitize_fts5_term("multi_word") == "multi_word"  # underscore kept
    assert sanitize_fts5_term("O'Brien") == "o'brien"  # apostrophe kept


def test_contains_cjk_detects_han_hiragana_katakana_hangul():
    assert contains_cjk("日本語") is True
    assert contains_cjk("ひらがな") is True
    assert contains_cjk("カタカナ") is True
    assert contains_cjk("한글") is True
    assert contains_cjk("plain english") is False


def test_normalize_cjk_for_fts_spaces_out_cjk_runs():
    result = normalize_cjk_for_fts("hello日本語world")
    assert result == "hello 日 本 語 world"


def test_build_fts5_query_plain_term_gets_prefix_match():
    assert build_fts5_query("hello") == '"hello"*'


def test_build_fts5_query_multiple_terms_joined_with_and():
    assert build_fts5_query("hello world") == '"hello"* AND "world"*'


def test_build_fts5_query_quoted_phrase_is_exact_no_prefix():
    assert build_fts5_query('"exact phrase"') == '"exact phrase"'


def test_build_fts5_query_negation():
    assert build_fts5_query("performance -sports") == '"performance"* NOT "sports"*'


def test_build_fts5_query_hyphenated_compound_becomes_phrase():
    assert build_fts5_query("multi-agent") == '"multi agent"'


def test_build_fts5_query_only_negative_terms_returns_none():
    assert build_fts5_query("-sports") is None


def test_build_fts5_query_empty_query_returns_none():
    assert build_fts5_query("   ") is None


def test_build_fts5_query_compound_term_with_slashes():
    assert build_fts5_query("src/lib/i18n.ts") == '"src lib i18n ts"'
