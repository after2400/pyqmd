"""The cross-encoder rerank backend, with fakes instead of real weights.

Needs real MLX (not model weights), so it skips where mlx can't import."""

import importlib
import json
from types import SimpleNamespace

import pytest

# On Linux the mlx wheel installs but its import fails with a plain
# ImportError (no libmlx.so), which importorskip doesn't catch by default.
mx = pytest.importorskip("mlx.core", exc_type=ImportError)
nn = pytest.importorskip("mlx.nn", exc_type=ImportError)

from pyqmd_mlx.llm import _constants  # noqa: E402

cross_encoder = importlib.import_module("pyqmd_mlx.llm._cross_encoder")

MODEL_ID = "someorg/some-cross-encoder"


def _write_config(directory, architectures, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": architectures, **extra}))
    return str(directory)


# --- config-based detection -------------------------------------------------


def test_is_cross_encoder_true_for_sequence_classification(tmp_path):
    model_dir = _write_config(tmp_path / "m", ["ModernBertForSequenceClassification"])
    assert cross_encoder.is_cross_encoder(model_dir) is True


def test_is_cross_encoder_false_for_causal_lm(tmp_path):
    model_dir = _write_config(tmp_path / "m", ["Qwen3ForCausalLM"])
    assert cross_encoder.is_cross_encoder(model_dir) is False


def test_is_cross_encoder_false_when_config_unreadable(tmp_path):
    assert cross_encoder.is_cross_encoder(str(tmp_path / "missing")) is False


def test_is_cross_encoder_false_for_config_without_architectures(tmp_path):
    model_dir = _write_config(tmp_path / "m", [])
    assert cross_encoder.is_cross_encoder(model_dir) is False


def test_read_config_prefers_the_hf_cache_over_the_network(monkeypatch, tmp_path):
    import huggingface_hub

    cached = tmp_path / "cached.json"
    cached.write_text(json.dumps({"architectures": ["BertForSequenceClassification"]}))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))

    def no_network(*args, **kwargs):
        raise AssertionError("must not download when config.json is cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)

    assert cross_encoder.read_config(MODEL_ID) == {
        "architectures": ["BertForSequenceClassification"]
    }


def test_read_config_downloads_config_json_when_not_cached(monkeypatch, tmp_path):
    import huggingface_hub

    fetched = tmp_path / "fetched.json"
    fetched.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    seen = []

    def fake_download(repo_id, filename, **kwargs):
        seen.append((repo_id, filename))
        return str(fetched)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert cross_encoder.read_config(MODEL_ID) == {"architectures": ["Qwen3ForCausalLM"]}
    assert seen == [(MODEL_ID, "config.json")]


# --- calibration ------------------------------------------------------------


def test_calibrate_identity_defaults_map_zero_to_one_half():
    assert cross_encoder.calibrate(0.0, 0.0, 1.0) == 0.5


@pytest.mark.parametrize("bias,temperature", [(0.7, 1.5), (-2.0, 0.25), (3.0, 4.0)])
def test_calibrate_maps_the_bias_to_one_half(bias, temperature):
    assert cross_encoder.calibrate(bias, bias, temperature) == pytest.approx(0.5)


def test_calibrate_stays_strictly_inside_zero_and_one_for_extreme_logits():
    # math.exp overflows past ~709, so a naive sigmoid raises on a very low logit.
    for logit in (-1e6, -50.0, 50.0, 1e6):
        score = cross_encoder.calibrate(logit, 0.7, 1.5)
        assert 0.0 <= score <= 1.0


def test_calibrate_is_monotonic_in_the_logit():
    logits = [-30.0, -3.0, -0.1, 0.0, 0.1, 3.0, 30.0]
    scores = [cross_encoder.calibrate(x, 0.7, 1.5) for x in logits]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


def test_granite_has_a_fitted_calibration_entry():
    bias, temperature = _constants.RERANK_CALIBRATION[_constants.GRANITE_RERANK_MODEL]
    assert temperature > 0
    # Irrelevant pairs sit near logit 0.8 and relevant ones near 1.6; the median
    # matching to Qwen's scale puts the bias between them.
    assert 0.8 < bias < 1.6


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_calibrate_rejects_a_non_positive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        cross_encoder.calibrate(1.0, 0.0, temperature)


# --- the SiLU head fix ------------------------------------------------------


def _model_with_activation(activation):
    return SimpleNamespace(head=SimpleNamespace(act=activation))


def test_fix_classifier_activation_replaces_gelu_with_silu():
    model = _model_with_activation(nn.GELU(approx="precise"))
    cross_encoder.fix_classifier_activation(model, "silu")
    assert isinstance(model.head.act, nn.SiLU)


def test_fix_classifier_activation_is_a_no_op_when_already_silu():
    already = nn.SiLU()
    model = _model_with_activation(already)
    cross_encoder.fix_classifier_activation(model, "silu")
    assert model.head.act is already


@pytest.mark.parametrize("name", [None, "gelu"])
def test_fix_classifier_activation_leaves_other_activations_alone(name):
    gelu = nn.GELU(approx="precise")
    model = _model_with_activation(gelu)
    cross_encoder.fix_classifier_activation(model, name)
    assert model.head.act is gelu


# --- scoring ----------------------------------------------------------------


class FakeTokenizer:
    """Pair tokenizer: one token per word. Like the real fast tokenizer, it
    raises a plain Exception when truncation="only_second" is asked of a
    first sequence that alone exceeds max_length."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, document, **kwargs):
        self.calls.append(kwargs)
        max_length = kwargs["max_length"]
        if kwargs["truncation"] == "only_second" and len(query.split()) > max_length:
            raise Exception("Truncation error: Sequence to truncate too short")
        n = min(len(query.split()) + len(document.split()), max_length)
        return {"input_ids": [list(range(n))], "attention_mask": [[1] * n]}


class FakeModel:
    """pooler_output is the raw logit (load_cross_encoder switches the model
    to regression mode); here it is the number of input tokens."""

    def __call__(self, input_ids, attention_mask=None):
        return SimpleNamespace(pooler_output=mx.array([[float(input_ids.shape[1])]]))


def _encoder(tokenizer=None, bias=0.0, temperature=1.0):
    return cross_encoder.CrossEncoder(
        model=FakeModel(),
        tokenizer=tokenizer or FakeTokenizer(),
        bias=bias,
        temperature=temperature,
    )


def test_score_logits_returns_one_raw_logit_per_document_in_order():
    scores = cross_encoder.score_logits(_encoder(), "a b", ["x", "x y z", "x y"])
    assert scores == [3.0, 5.0, 4.0]
    assert all(isinstance(s, float) for s in scores)


def test_score_logits_of_no_documents_is_empty():
    assert cross_encoder.score_logits(_encoder(), "a b", []) == []


def test_score_logits_truncates_the_document_only_at_the_cap():
    tokenizer = FakeTokenizer()
    scores = cross_encoder.score_logits(_encoder(tokenizer), "short query", ["w " * 5000])
    assert tokenizer.calls[0]["truncation"] == "only_second"
    assert tokenizer.calls[0]["max_length"] == _constants.CROSS_ENCODER_MAX_TOKENS == 2048
    assert scores == [2048.0]


def test_score_logits_long_query_does_not_raise():
    tokenizer = FakeTokenizer()
    scores = cross_encoder.score_logits(_encoder(tokenizer), "q " * 3000, ["a short document"])
    assert [c["truncation"] for c in tokenizer.calls] == ["only_second", "longest_first"]
    assert scores == [2048.0]


# --- loading ----------------------------------------------------------------


def test_load_cross_encoder_fixes_the_head_once_and_reads_calibration(monkeypatch):
    model = SimpleNamespace(
        head=SimpleNamespace(act=nn.GELU(approx="precise")), is_regression=False
    )
    tokenizer = FakeTokenizer()
    monkeypatch.setattr("mlx_embeddings.utils.load", lambda model_id: (model, tokenizer))
    monkeypatch.setattr(
        cross_encoder, "read_config", lambda model_id: {"classifier_activation": "silu"}
    )
    monkeypatch.setitem(_constants.RERANK_CALIBRATION, MODEL_ID, (0.7, 1.5))
    applied = []
    real_fix = cross_encoder.fix_classifier_activation
    monkeypatch.setattr(
        cross_encoder,
        "fix_classifier_activation",
        lambda m, name: (applied.append(name), real_fix(m, name))[1],
    )

    encoder = cross_encoder.load_cross_encoder(MODEL_ID)

    assert applied == ["silu"]
    assert isinstance(model.head.act, nn.SiLU)
    assert model.is_regression is True  # pooler_output becomes the raw logit
    assert (encoder.model, encoder.tokenizer) == (model, tokenizer)
    assert (encoder.bias, encoder.temperature) == (0.7, 1.5)


def test_load_cross_encoder_defaults_to_identity_calibration(monkeypatch):
    model = SimpleNamespace(head=SimpleNamespace(act=nn.SiLU()), is_regression=False)
    monkeypatch.setattr("mlx_embeddings.utils.load", lambda model_id: (model, FakeTokenizer()))
    monkeypatch.setattr(cross_encoder, "read_config", lambda model_id: {})

    encoder = cross_encoder.load_cross_encoder("unknown/checkpoint")

    assert (encoder.bias, encoder.temperature) == (0.0, 1.0)
