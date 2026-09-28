"""Node's expansion grammar as a byte-level DFA, and the per-state vocab
tables built from it. Pure Python: no MLX, so this runs on CI's Ubuntu leg."""

import types

import pytest

from pyqmd_mlx.llm._expand_grammar import (
    _UNICODE_TO_BYTE,
    COLON,
    CONTENT,
    DEAD,
    N_STATES,
    SPACE,
    START0,
    START1,
    TYPE,
    build_tables,
    vocab_token_bytes,
    walk,
)


@pytest.mark.parametrize(
    "text, state",
    [
        (b"lex: a\n", START1),
        (b"vec: x y\nhyde: z\n", START1),
        (b"lex: a", CONTENT),  # a valid prefix, but not complete
        (b"hyde:", COLON),
        (b"lex: ", SPACE),
    ],
)
def test_walk_follows_the_grammar(text, state):
    assert walk(START0, text) == state


@pytest.mark.parametrize(
    "text",
    [b"lex:a", b"lex: \n", b"LEX: a", b"\n", b"foo", b"lex: a\n\n", b"<think>", b"hyd: a"],
)
def test_walk_rejects_what_the_grammar_forbids(text):
    assert walk(START0, text) == DEAD


# A tiny vocab. Index = token id. None = a special added token.
TOKENS = [
    b"lex",  # 0
    b"vec",  # 1
    b"hy",  # 2
    b"de",  # 3
    b":",  # 4
    b" foo",  # 5
    b" ",  # 6
    b"\n",  # 7
    b".\n",  # 8
    b".\n\n",  # 9
    b" foo\nlex",  # 10
    b"l",  # 11
    b"<think>",  # 12: a non-special added token, as literal text
    None,  # 13: EOS (special)
    None,  # 14: another special token
]
ID = {t: i for i, t in enumerate(TOKENS) if t is not None}
EOS = 13
OTHER_SPECIAL = 14


def _allowed(tables, state):
    return {i for i, s in enumerate(tables.next_state[state]) if s != DEAD}


def test_tables_match_walk_for_every_state_and_token():
    tables = build_tables(TOKENS, [EOS])
    assert len(tables.next_state) == N_STATES
    assert tables.vocab_size == len(TOKENS)
    for state in range(N_STATES):
        for tid, data in enumerate(TOKENS):
            if data is not None:
                assert tables.next_state[state][tid] == walk(state, data), (state, data)


def test_line_start_allows_only_type_word_prefixes():
    tables = build_tables(TOKENS, [EOS])
    assert _allowed(tables, START0) == {ID[b"lex"], ID[b"vec"], ID[b"hy"], ID[b"l"]}
    assert _allowed(tables, START1) == {ID[b"lex"], ID[b"vec"], ID[b"hy"], ID[b"l"], EOS}


def test_tokens_spanning_states_end_in_the_right_state():
    tables = build_tables(TOKENS, [EOS])
    assert tables.next_state[START0][ID[b"lex"]] == TYPE
    assert tables.next_state[TYPE][ID[b":"]] == COLON
    assert tables.next_state[COLON][ID[b" foo"]] == CONTENT
    assert tables.next_state[CONTENT][ID[b" foo\nlex"]] == TYPE
    assert tables.next_state[CONTENT][ID[b".\n"]] == START1
    assert tables.next_state[CONTENT][ID[b".\n\n"]] == DEAD
    assert tables.next_state[SPACE][ID[b"\n"]] == DEAD


def test_eos_is_allowed_only_after_a_complete_line():
    tables = build_tables(TOKENS, [EOS])
    assert [s for s in range(N_STATES) if tables.next_state[s][EOS] != DEAD] == [START1]


def test_special_tokens_are_always_dead():
    tables = build_tables(TOKENS, [EOS])
    assert all(tables.next_state[s][OTHER_SPECIAL] == DEAD for s in range(N_STATES))


def test_non_special_added_tokens_are_content_text():
    tables = build_tables(TOKENS, [EOS])
    think = ID[b"<think>"]
    assert tables.next_state[CONTENT][think] == CONTENT
    assert tables.next_state[SPACE][think] == CONTENT
    assert tables.next_state[START0][think] == DEAD
    assert tables.next_state[START1][think] == DEAD


def test_eos_ids_outside_the_vocab_are_ignored():
    tables = build_tables(TOKENS, [EOS, 999])
    assert tables.vocab_size == len(TOKENS)


def test_byte_alphabet_covers_all_256_bytes():
    assert len(_UNICODE_TO_BYTE) == 256
    assert sorted(_UNICODE_TO_BYTE.values()) == list(range(256))
    assert _UNICODE_TO_BYTE["Ġ"] == ord(" ")
    assert _UNICODE_TO_BYTE["Ċ"] == ord("\n")
    assert _UNICODE_TO_BYTE["a"] == ord("a")


def _fake_tokenizer(vocab, added):
    """Stands in for an HF tokenizer (or mlx_lm's TokenizerWrapper, which
    forwards these attributes to one)."""
    return types.SimpleNamespace(
        get_vocab=lambda: dict(vocab),
        added_tokens_decoder={
            tid: types.SimpleNamespace(content=content, special=special)
            for tid, (content, special) in added.items()
        },
    )


def test_vocab_token_bytes_maps_each_kind_of_token():
    tokenizer = _fake_tokenizer(
        {"lex": 0, "Ġfoo": 1, "Ċ": 2, "<|im_end|>": 3, "<think>": 4},
        {3: ("<|im_end|>", True), 4: ("<think>", False)},
    )
    assert vocab_token_bytes(tokenizer) == [b"lex", b" foo", b"\n", None, b"<think>"]


def test_vocab_token_bytes_includes_added_tokens_missing_from_the_vocab():
    tokenizer = _fake_tokenizer({"lex": 0}, {1: ("</think>", False), 2: ("<|x|>", True)})
    assert vocab_token_bytes(tokenizer) == [b"lex", b"</think>", None]


def test_vocab_token_bytes_rejects_a_non_byte_level_token():
    tokenizer = _fake_tokenizer({"lex": 0, "☃": 1}, {})
    with pytest.raises(ValueError, match="isn't byte-level BPE"):
        vocab_token_bytes(tokenizer)
