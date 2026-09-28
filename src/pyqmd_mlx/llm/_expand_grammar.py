"""Node's query-expansion output grammar, compiled by hand into a byte-level
DFA, plus per-state vocab tables for constraining decoding.

Node (qmd/src/llm.ts, expandQuery) decodes expansions under this GBNF:

    root ::= line+
    line ::= type ": " content "\\n"
    type ::= "lex" | "vec" | "hyde"
    content ::= [^\\n]+

Pure Python, no MLX: llm/expand.py turns the tables into logit masks. See
docs/specs/2026-09-25-expansion-output-grammar-design.md.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

(
    START0,  # line start, no complete line yet
    START1,  # line start after at least one complete line; EOS allowed here only
    L,
    LE,
    V,
    VE,
    H,
    HY,
    HYD,  # partial type words
    TYPE,  # a full type word; needs ":"
    COLON,  # needs " "
    SPACE,  # needs one content byte
    CONTENT,  # inside content
) = range(13)
N_STATES = 13
DEAD = 255

_NL = 0x0A
_LINE_START = {ord("l"): L, ord("v"): V, ord("h"): H}
# state -> byte -> state, for every state before content starts. Working on
# bytes keeps partial-UTF-8 tokens correct: 0x0A never occurs inside a
# multi-byte sequence, so such a token's bytes are ordinary content bytes.
_FIXED: dict[int, dict[int, int]] = {
    START0: _LINE_START,
    START1: _LINE_START,
    L: {ord("e"): LE},
    LE: {ord("x"): TYPE},
    V: {ord("e"): VE},
    VE: {ord("c"): TYPE},
    H: {ord("y"): HY},
    HY: {ord("d"): HYD},
    HYD: {ord("e"): TYPE},
    TYPE: {ord(":"): COLON},
    COLON: {ord(" "): SPACE},
}


def step(state: int, byte: int) -> int:
    """The state after one byte, or DEAD."""
    if state == SPACE or state == CONTENT:
        if byte == _NL:
            return START1 if state == CONTENT else DEAD
        return CONTENT
    return _FIXED[state].get(byte, DEAD)


def walk(state: int, data: bytes) -> int:
    """The state after all of data, or DEAD as soon as a byte is dead."""
    for byte in data:
        state = step(state, byte)
        if state == DEAD:
            return DEAD
    return state


@dataclass
class GrammarTables:
    """next_state[state][token_id]: the state after that token, or DEAD if
    the grammar forbids it there. EOS ids map START1 -> START1 and are DEAD
    elsewhere. mask_cache is scratch space for llm/expand.py's MLX masks."""

    next_state: list[bytearray]
    mask_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def vocab_size(self) -> int:
        return len(self.next_state[0])


def build_tables(token_bytes: Sequence[bytes | None], eos_ids: Iterable[int]) -> GrammarTables:
    """Per-state next-state rows over a vocab. token_bytes[i] is token i's
    raw bytes, or None for a token the grammar never allows (a special
    token)."""
    n = len(token_bytes)
    rows = [bytearray([DEAD]) * n for _ in range(N_STATES)]
    for tid, data in enumerate(token_bytes):
        if not data:
            continue
        has_newline = _NL in data
        for state in range(N_STATES):
            if state == SPACE or state == CONTENT:
                # Most tokens have no newline, so they just stay in content.
                rows[state][tid] = walk(state, data) if has_newline else CONTENT
            elif data[0] in _FIXED[state]:
                rows[state][tid] = walk(state, data)
    for tid in eos_ids:
        if 0 <= tid < n:
            rows[START1][tid] = START1
    return GrammarTables(next_state=rows)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level BPE alphabet: each byte's stand-in character in
    vocab token strings. Qwen's tokenizer uses the same table."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    mapping = {b: chr(b) for b in printable}
    extra = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + extra)
            extra += 1
    return mapping


_UNICODE_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}


def vocab_token_bytes(tokenizer) -> list[bytes | None]:
    """Each token id's raw bytes, for a byte-level BPE tokenizer (an HF
    tokenizer, or mlx_lm's TokenizerWrapper, which forwards to one).

    Read from the vocab rather than decode(), which can clean up spaces and
    turns partial UTF-8 into U+FFFD. Added tokens flagged special come out
    None. Other added tokens (Qwen's <think>, </think>, ...) are literal
    text, as llama.cpp's grammar treats them. Raises ValueError on a regular
    token outside the byte-level alphabet."""
    vocab = tokenizer.get_vocab()
    added = tokenizer.added_tokens_decoder
    out: list[bytes | None] = [None] * (max([*vocab.values(), *added]) + 1)
    for text, tid in vocab.items():
        if tid in added:
            continue
        try:
            out[tid] = bytes(_UNICODE_TO_BYTE[c] for c in text)
        except KeyError:
            raise ValueError(
                f"token {tid} ({text!r}) isn't byte-level BPE; the expansion "
                "grammar needs a byte-level BPE tokenizer"
            ) from None
    for tid, token in added.items():
        if not token.special:
            out[tid] = token.content.encode("utf-8")
    return out
