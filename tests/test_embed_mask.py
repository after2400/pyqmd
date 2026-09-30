"""embed()'s padding mask for gemma3_text models.

Needs real MLX (not model weights), so it skips where mlx can't import."""

from types import SimpleNamespace

import pytest

# On Linux the mlx wheel installs but its import fails with a plain
# ImportError (no libmlx.so), which importorskip doesn't catch by default.
mx = pytest.importorskip("mlx.core", exc_type=ImportError)

from pyqmd_mlx.llm.embed import _activation_dtype, _additive_padding_mask  # noqa: E402

NEG_INF = float("-inf")


def test_padding_keys_are_masked_for_every_query():
    mask = _additive_padding_mask(mx.array([[1, 1, 0], [1, 1, 1]]), mx.float16)
    assert mask.shape == (2, 1, 3, 3)
    assert mask.dtype == mx.float16
    assert mask[0, 0].tolist() == [[0.0, 0.0, NEG_INF]] * 3
    assert mask[1, 0].tolist() == [[0.0, 0.0, 0.0]] * 3


def test_a_quantized_embedding_runs_in_its_scales_dtype():
    # The packed uint32 weight would turn -inf into 0 (mlx-embeddings
    # 0.1.0's bug); the scales carry the dtype activations use.
    embed_tokens = SimpleNamespace(
        weight=mx.zeros((1,), dtype=mx.uint32), scales=mx.zeros((1,), dtype=mx.float16)
    )
    assert _activation_dtype(embed_tokens) == mx.float16


def test_an_unquantized_embedding_runs_in_its_weight_dtype():
    embed_tokens = SimpleNamespace(weight=mx.zeros((1,), dtype=mx.bfloat16))
    assert _activation_dtype(embed_tokens) == mx.bfloat16
