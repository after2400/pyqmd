"""Rerank-model resolution: PYQMD_RERANK_MODEL overrides the default."""

from pyqmd_mlx.llm import (
    DEFAULT_RERANK_MODEL,
    RERANK_MODEL_ENV_VAR,
    RerankModelError,
    resolve_rerank_model,
)
from pyqmd_mlx.store import Store


def test_env_var_name():
    assert RERANK_MODEL_ENV_VAR == "PYQMD_RERANK_MODEL"


def test_default_rerank_model_is_unchanged():
    assert DEFAULT_RERANK_MODEL == "mlx-community/Qwen3-Reranker-0.6B-mxfp8"


def test_rerank_model_error_is_a_runtime_error():
    assert issubclass(RerankModelError, RuntimeError)


def test_resolve_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(RERANK_MODEL_ENV_VAR, raising=False)
    assert resolve_rerank_model() == DEFAULT_RERANK_MODEL


def test_resolve_defaults_when_empty(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "")
    assert resolve_rerank_model() == DEFAULT_RERANK_MODEL


def test_resolve_uses_env_value(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "ibm-granite/granite-embedding-reranker-english-r2")
    assert resolve_rerank_model() == "ibm-granite/granite-embedding-reranker-english-r2"


def test_store_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(RERANK_MODEL_ENV_VAR, raising=False)
    assert Store(":memory:")._rerank_model == DEFAULT_RERANK_MODEL


def test_store_uses_env_when_no_argument(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "someorg/custom-reranker")
    assert Store(":memory:")._rerank_model == "someorg/custom-reranker"


def test_store_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "someorg/custom-reranker")
    store = Store(":memory:", rerank_model="someorg/explicit")
    assert store._rerank_model == "someorg/explicit"


def test_rerank_model_error_is_in_both_expected_exception_tuples():
    from pyqmd_mlx.cli._errors import EXPECTED_EXCEPTIONS as cli_expected
    from pyqmd_mlx.mcp._errors import EXPECTED_EXCEPTIONS as mcp_expected

    assert RerankModelError in cli_expected
    assert RerankModelError in mcp_expected
