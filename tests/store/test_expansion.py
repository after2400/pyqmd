from pyqmd_mlx.store._expansion import parse_expanded_lines, postprocess_expansion


def _pairs(parts):
    return [(p.type, p.query) for p in parts]


def test_parse_keeps_typed_lines_in_order():
    parts = parse_expanded_lines(
        ["hyde: a passage", "lex: keyword one", "lex: keyword two", "vec: semantic rephrase"]
    )
    assert _pairs(parts) == [
        ("hyde", "a passage"),
        ("lex", "keyword one"),
        ("lex", "keyword two"),
        ("vec", "semantic rephrase"),
    ]


def test_parse_skips_untyped_lines():
    parts = parse_expanded_lines(["not a typed line", "lex: real one"])
    assert _pairs(parts) == [("lex", "real one")]


def test_keeps_only_parts_that_mention_a_query_term():
    lines = [
        "lex: auth config setup",
        "vec: how to set up login",
        "hyde: Auth is configured via env vars.",
    ]
    assert _pairs(postprocess_expansion("auth config", lines)) == [
        ("lex", "auth config setup"),
        ("hyde", "Auth is configured via env vars."),
    ]


def test_term_match_is_case_insensitive_substring():
    # Node's hasQueryTerm uses substring containment: "auth" matches inside
    # "AUTHENTICATION".
    parts = postprocess_expansion("Auth", ["lex: AUTHENTICATION flow"])
    assert _pairs(parts) == [("lex", "AUTHENTICATION flow")]


def test_query_punctuation_is_not_a_term():
    # "c++" -> terms ["c"]; the "+" characters are stripped like Node's
    # replace(/[^a-z0-9\s]/g, " ").
    parts = postprocess_expansion("c++", ["lex: c programming", "vec: rust language"])
    assert _pairs(parts) == [("lex", "c programming")]


def test_query_without_terms_keeps_every_part():
    parts = postprocess_expansion("???", ["lex: question marks", "vec: punctuation only"])
    assert _pairs(parts) == [("lex", "question marks"), ("vec", "punctuation only")]


def test_prose_output_falls_back_to_hyde_only():
    # Fallback is hyde/lex/vec; lex and vec equal the original query, which
    # pyqmd already searches as-is, so only hyde survives -- as in Node.
    parts = postprocess_expansion(
        "how long to cook noodles", ["Here is a guide.", "Boil the water first."]
    )
    assert _pairs(parts) == [("hyde", "Information about how long to cook noodles")]


def test_all_parts_filtered_out_falls_back():
    parts = postprocess_expansion("zebra", ["lex: horse", "vec: striped animal"])
    assert _pairs(parts) == [("hyde", "Information about zebra")]


def test_drops_parts_identical_to_the_query():
    parts = postprocess_expansion("auth config", ["lex: auth config", "vec: auth config options"])
    assert _pairs(parts) == [("vec", "auth config options")]


def test_drops_empty_parts():
    parts = postprocess_expansion("auth", ["lex:", "vec: auth flow"])
    assert _pairs(parts) == [("vec", "auth flow")]
