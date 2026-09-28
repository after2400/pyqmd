from pyqmd_mlx.store._intent import INTENT_WEIGHT_CHUNK, extract_intent_terms


def test_extract_intent_terms_filters_stop_words():
    terms = extract_intent_terms("I want to find the authentication setup")
    assert "authentication" in terms
    assert "setup" in terms
    assert "want" not in terms
    assert "the" not in terms


def test_extract_intent_terms_strips_punctuation():
    terms = extract_intent_terms("performance, latency!")
    assert "performance" in terms
    assert "latency" in terms


def test_extract_intent_terms_keeps_short_domain_terms():
    terms = extract_intent_terms("API SQL setup")
    assert "api" in terms
    assert "sql" in terms


def test_extract_intent_terms_empty_string_returns_empty_list():
    assert extract_intent_terms("") == []


def test_intent_weight_chunk_is_half_of_query_term_weight():
    assert INTENT_WEIGHT_CHUNK == 0.5
