"""_keyed_sampler: seeded sampling that never touches MLX's global RNG.

Needs real MLX (not model weights), so it skips where mlx can't import."""

import pytest

# On Linux the mlx wheel installs but its import fails with a plain
# ImportError (no libmlx.so), which importorskip doesn't catch by default.
mx = pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm.sample_utils", exc_type=ImportError)

from pyqmd_mlx.llm._expand_grammar import build_tables  # noqa: E402
from pyqmd_mlx.llm.expand import (  # noqa: E402
    _TOP_K,
    _grammar_processor,
    _keyed_sampler,
)


def _logprob_steps(n: int = 20, vocab: int = 100) -> list:
    """A fixed sequence of (1, vocab) log-probability rows, as generate_step
    hands the sampler."""
    steps = []
    for i in range(n):
        logits = mx.random.normal((1, vocab), key=mx.random.key(i))
        steps.append(logits - mx.logsumexp(logits, keepdims=True))
    return steps


def _draw(sampler, steps) -> list[int]:
    return [sampler(step).item() for step in steps]


def test_same_seed_gives_same_tokens():
    steps = _logprob_steps()
    assert _draw(_keyed_sampler(7), steps) == _draw(_keyed_sampler(7), steps)


def test_different_seeds_diverge():
    steps = _logprob_steps()
    assert _draw(_keyed_sampler(7), steps) != _draw(_keyed_sampler(8), steps)


def test_sampled_tokens_stay_within_top_k():
    steps = _logprob_steps()
    tokens = _draw(_keyed_sampler(3), steps)
    for step, token in zip(steps, tokens):
        top = set(mx.argsort(-step[0])[:_TOP_K].tolist())
        assert token in top


def test_sampler_does_not_touch_global_rng():
    steps = _logprob_steps()
    mx.random.seed(0)
    expected = mx.random.uniform(shape=(4,)).tolist()

    mx.random.seed(0)
    _draw(_keyed_sampler(3), steps)
    assert mx.random.uniform(shape=(4,)).tolist() == expected


# A tiny vocab for the grammar processor. Index = token id; 5 is EOS.
_GRAMMAR_TOKENS = [b"lex", b":", b" foo", b"\n", b"zzz", None]
_EOS = 5
_WIDTH = 8  # wider than the vocab, like Qwen3's padded logits


def _allowed(row) -> set[int]:
    return {i for i, v in enumerate(row.tolist()) if v != float("-inf")}


def test_grammar_processor_tracks_state_across_calls():
    proc = _grammar_processor(build_tables(_GRAMMAR_TOKENS, [_EOS]))
    logits = mx.zeros((1, _WIDTH))
    history = [100, 101, 102]  # the prompt's tail; no token generated yet

    assert _allowed(proc(mx.array(history), logits)[0]) == {0}  # line start: "lex"
    history.append(0)
    assert _allowed(proc(mx.array(history), logits)[0]) == {1}  # ":"
    history.append(1)
    assert _allowed(proc(mx.array(history), logits)[0]) == {2}  # " foo" (needs the space)
    history.append(2)
    assert _allowed(proc(mx.array(history), logits)[0]) == {0, 1, 2, 3, 4}  # content
    history.append(3)
    assert _allowed(proc(mx.array(history), logits)[0]) == {0, _EOS}  # line start after a line


def test_grammar_processor_leaves_allowed_logits_unchanged():
    proc = _grammar_processor(build_tables(_GRAMMAR_TOKENS, [_EOS]))
    logits = mx.arange(_WIDTH, dtype=mx.float32)[None]

    out = proc(mx.array([100]), logits)[0].tolist()

    assert out[0] == 0.0
    assert all(v == float("-inf") for v in out[1:])


def test_sampler_never_picks_a_masked_token():
    vocab = 100
    for allowed in ({3, 17, 42}, {64}):
        mask = mx.array([i in allowed for i in range(vocab)])
        sampler = _keyed_sampler(5)
        for step in _logprob_steps(n=50, vocab=vocab):
            masked = mx.where(mask, step, float("-inf"))
            masked = masked - mx.logsumexp(masked, keepdims=True)
            assert sampler(masked).item() in allowed
