"""rerank()'s choice of backend and its load-failure reporting, with fakes.

Needs no MLX and no weights: the backends are replaced, so this runs on CI
runners where mlx can't import."""

import importlib
import sys
import types

import pytest

from pyqmd_mlx.llm import RERANK_MODEL_ENV_VAR, RerankModelError

# pyqmd_mlx.llm re-exports the rerank *function* under the module's name.
rerank_module = importlib.import_module("pyqmd_mlx.llm.rerank")
cross_encoder = importlib.import_module("pyqmd_mlx.llm._cross_encoder")

MODEL_ID = "someorg/some-reranker"


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    monkeypatch.setattr(rerank_module, "_rerank_cache", {})
    monkeypatch.setattr(rerank_module, "_cross_encoder_cache", {})


def _fail(*args, **kwargs):
    raise AssertionError("must not be called")


def test_cached_yes_no_model_never_reads_config(monkeypatch):
    rerank_module._rerank_cache[MODEL_ID] = ("model", "tokenizer")
    monkeypatch.setattr(cross_encoder, "is_cross_encoder", _fail)
    monkeypatch.setattr(rerank_module, "_rerank_yes_no", lambda q, docs, model: [0.25] * len(docs))

    assert rerank_module.rerank("q", ["a", "b"], model=MODEL_ID) == [0.25, 0.25]


def test_sequence_classification_config_routes_to_the_cross_encoder(monkeypatch):
    loads = []

    def load(model_id):
        loads.append(model_id)
        return cross_encoder.CrossEncoder(None, None, bias=0.7, temperature=1.5)

    monkeypatch.setattr(cross_encoder, "is_cross_encoder", lambda model_id: True)
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", load)
    monkeypatch.setattr(
        cross_encoder, "score_logits", lambda enc, q, docs: [2.0, 0.7, -1.0][: len(docs)]
    )
    monkeypatch.setattr(rerank_module, "_rerank_yes_no", _fail)

    scores = rerank_module.rerank("q", ["a", "b", "c"], model=MODEL_ID)

    assert scores == [cross_encoder.calibrate(x, 0.7, 1.5) for x in (2.0, 0.7, -1.0)]
    assert scores[1] == pytest.approx(0.5)
    rerank_module.rerank("q", ["a"], model=MODEL_ID)
    assert loads == [MODEL_ID]  # loaded once, then cached


def test_causal_lm_config_keeps_the_yes_no_backend(monkeypatch):
    monkeypatch.setattr(cross_encoder, "is_cross_encoder", lambda model_id: False)
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", _fail)
    monkeypatch.setattr(rerank_module, "_rerank_yes_no", lambda q, docs, model: [0.5] * len(docs))

    assert rerank_module.rerank("q", ["a"], model=MODEL_ID) == [0.5]


def test_empty_documents_return_empty_without_loading_or_reading_config(monkeypatch):
    monkeypatch.setattr(cross_encoder, "is_cross_encoder", _fail)
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", _fail)
    monkeypatch.setattr(rerank_module, "_rerank_yes_no", _fail)

    assert rerank_module.rerank("q", [], model=MODEL_ID) == []


def test_unreadable_config_falls_back_to_yes_no_and_reports_the_load_error(monkeypatch, tmp_path):
    def boom(model_id):
        raise ValueError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'")

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=boom))
    # The yes/no path imports mlx.core before it loads; stub it for runners without MLX.
    mlx_stub = types.ModuleType("mlx")
    mlx_stub.core = types.ModuleType("mlx.core")
    monkeypatch.setitem(sys.modules, "mlx", mlx_stub)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_stub.core)
    missing = str(tmp_path / "no-such-model")  # no config.json anywhere -> not a cross-encoder

    with pytest.raises(RerankModelError) as excinfo:
        rerank_module.rerank("q", ["a"], model=missing)

    message = str(excinfo.value)
    assert f"Could not load rerank model '{missing}'" in message
    assert "Repo id must be in the form" in message
    assert RERANK_MODEL_ENV_VAR in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_cross_encoder_load_failure_is_a_rerank_model_error(monkeypatch):
    def boom(model_id):
        raise RuntimeError("weights missing.")

    monkeypatch.setattr(cross_encoder, "is_cross_encoder", lambda model_id: True)
    monkeypatch.setattr(cross_encoder, "load_cross_encoder", boom)

    with pytest.raises(RerankModelError) as excinfo:
        rerank_module.rerank("q", ["a"], model=MODEL_ID)

    message = str(excinfo.value)
    assert f"Could not load rerank model '{MODEL_ID}': weights missing." in message
    assert RERANK_MODEL_ENV_VAR in message
    assert "\n" not in message  # one line, for the CLI's "Error: ..." output
