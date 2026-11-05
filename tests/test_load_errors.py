"""One clean message for a model that cannot be loaded or downloaded, whatever
the network did (blocked repo, TLS inspection, offline). No models, no network."""

import ssl
import types

import httpx
import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from pyqmd_mlx.llm import EmbeddingModelError, ExpansionModelError, RerankModelError
from pyqmd_mlx.llm._constants import (
    DEFAULT_EXPAND_MODEL,
    DEFAULT_RERANK_MODEL,
    GRANITE_EXPAND_MODEL,
)
from pyqmd_mlx.llm._load_errors import explain_failure, load_failure_message


class HfHubHTTPError(Exception):
    def __init__(self, message, status):
        super().__init__(message)
        self.response = types.SimpleNamespace(status_code=status)


class GatedRepoError(HfHubHTTPError):
    pass


class RepositoryNotFoundError(HfHubHTTPError):
    pass


def _chain(outer, inner):
    outer.__cause__ = inner
    return outer


def test_a_403_is_reported_as_denied_access():
    reason = explain_failure(HfHubHTTPError("403 Client Error: Forbidden for url https://x", 403))
    assert "access to the repository was denied" in reason
    assert "network filter" in reason


def test_a_gated_repo_is_reported_as_denied_access():
    assert "denied" in explain_failure(GatedRepoError("gated", 401))


def test_repository_not_found_says_the_id_may_be_wrong_or_hidden():
    reason = explain_failure(RepositoryNotFoundError("Repository Not Found", 404))
    assert "was not found" in reason
    assert "check the id" in reason


def test_a_tls_verification_failure_points_at_the_ca_bundle():
    inner = ssl.SSLCertVerificationError(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    )
    outer = _chain(httpx.ConnectError("connection failed"), inner)
    reason = explain_failure(outer)
    assert "TLS verification" in reason
    assert "SSL_CERT_FILE" in reason


def test_tls_wins_over_the_wrapping_connection_error():
    outer = _chain(
        RuntimeError("could not download"),
        _chain(httpx.ConnectError("boom"), ssl.SSLError("CERTIFICATE_VERIFY_FAILED")),
    )
    assert "TLS verification" in explain_failure(outer)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("[Errno 8] nodename nor servname provided"),
        httpx.ConnectTimeout("timed out"),
        OfflineModeIsEnabled("offline"),
        ConnectionError("Connection reset by peer"),
    ],
)
def test_connection_problems_say_huggingface_could_not_be_reached(exc):
    assert "Hugging Face could not be reached" in explain_failure(exc)


def test_anything_else_keeps_its_own_message_on_one_line():
    reason = explain_failure(ValueError("weights missing.\nsecond line\n"))
    assert reason == "weights missing"


def test_a_long_message_is_trimmed():
    assert len(explain_failure(ValueError("x" * 1000))) <= 305


def test_a_default_set_model_suggests_the_granite_set():
    message = load_failure_message(
        "query-expansion",
        DEFAULT_EXPAND_MODEL,
        HfHubHTTPError("Forbidden", 403),
        role="expand",
        env_var="PYQMD_EXPAND_MODEL",
    )
    assert message.startswith(f"Could not load query-expansion model '{DEFAULT_EXPAND_MODEL}': ")
    assert "pyqmd models use granite" in message
    assert "PYQMD_MODEL_SET=granite" in message
    assert (
        "Set PYQMD_EXPAND_MODEL to a Hugging Face repo id or a local MLX model directory" in message
    )


def test_the_default_reranker_also_suggests_granite():
    message = load_failure_message(
        "rerank",
        DEFAULT_RERANK_MODEL,
        ValueError("boom"),
        role="rerank",
        env_var="PYQMD_RERANK_MODEL",
    )
    assert "pyqmd models use granite" in message


def test_an_explicit_choice_is_not_told_to_switch_sets():
    message = load_failure_message(
        "query-expansion",
        GRANITE_EXPAND_MODEL,
        ValueError("boom"),
        role="expand",
        env_var="PYQMD_EXPAND_MODEL",
    )
    assert message == (
        f"Could not load query-expansion model '{GRANITE_EXPAND_MODEL}': boom. "
        "Set PYQMD_EXPAND_MODEL to a Hugging Face repo id or a local MLX model directory to override."
    )


def test_the_embedding_model_has_no_alternative_to_suggest():
    message = load_failure_message(
        "embedding", "mlx-community/embeddinggemma-300m-8bit", ValueError("boom"), role="embed"
    )
    assert "models use" not in message
    assert "pyqmd pull" in message


def test_the_verb_can_be_download():
    message = load_failure_message(
        "rerank",
        "org/x",
        ValueError("boom"),
        role="rerank",
        env_var="PYQMD_RERANK_MODEL",
        verb="download",
    )
    assert message.startswith("Could not download rerank model 'org/x': boom.")


def test_error_classes_are_runtime_errors():
    for cls in (ExpansionModelError, RerankModelError, EmbeddingModelError):
        assert issubclass(cls, RuntimeError)
