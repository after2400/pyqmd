"""rerank()'s scoring path, with a tiny fake model instead of real weights.

Needs real MLX (not model weights), so it skips where mlx can't import."""

import importlib
import math
from types import SimpleNamespace

import pytest

# On Linux the mlx wheel installs but its import fails with a plain
# ImportError (no libmlx.so), which importorskip doesn't catch by default.
mx = pytest.importorskip("mlx.core", exc_type=ImportError)

# pyqmd_mlx.llm re-exports the rerank *function* under the module's name.
rerank_module = importlib.import_module("pyqmd_mlx.llm.rerank")

YES, NO = 1, 2
VOCAB = 4
MODEL_ID = "fake/reranker"


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {"yes": YES, "no": NO}[token]

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class FakeHead:
    """Output layer: the "yes" logit is the hidden value, the others 0."""

    def __init__(self):
        self.seen_shapes = []

    def __call__(self, hidden):
        self.seen_shapes.append(tuple(hidden.shape))
        logits = mx.zeros((*hidden.shape[:2], VOCAB))
        return logits + (mx.arange(VOCAB) == YES) * hidden[..., :1]


class FakeBody:
    """Transformer body: position t's hidden state is t, width 3."""

    def __init__(self, head):
        self.embed_tokens = SimpleNamespace(as_linear=head)

    def __call__(self, inputs):
        batch, length = inputs.shape
        return mx.broadcast_to(
            mx.arange(length, dtype=mx.float32)[None, :, None], (batch, length, 3)
        )


class FakeModel:
    def __init__(self, tie_word_embeddings):
        self.head = FakeHead()
        self.args = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.model = FakeBody(self.head)
        if not tie_word_embeddings:
            self.lm_head = self.head
            self.model.embed_tokens = SimpleNamespace(as_linear=None)

    def __call__(self, inputs):
        raise AssertionError("rerank() must not compute logits for every position")


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_rerank_applies_output_layer_to_last_position_only(monkeypatch, tie_word_embeddings):
    model = FakeModel(tie_word_embeddings)
    monkeypatch.setitem(rerank_module._rerank_cache, MODEL_ID, (model, FakeTokenizer()))

    scores = rerank_module.rerank("q", ["one two", "one two three four"], model=MODEL_ID)

    assert model.head.seen_shapes == [(1, 1, 3), (1, 1, 3)]
    # Scored from the last position's yes/no logits: yes = last index, no = 0.
    lengths = [
        len(rerank_module.build_rerank_prompt("q", d).split())
        for d in ["one two", "one two three four"]
    ]
    assert scores == [1.0 / (1.0 + math.exp(-(n - 1))) for n in lengths]
