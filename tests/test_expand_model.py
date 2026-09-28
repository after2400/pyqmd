"""Query-expansion model resolution (Hub default, PYQMD_EXPAND_MODEL
override) and clean load-failure reporting."""

import sys
import types

import pytest

from pyqmd_mlx.llm import (
    DEFAULT_EXPAND_MODEL,
    EXPAND_MODEL_ENV_VAR,
    ExpansionModelError,
    expand_query,
    resolve_expand_model,
)
from pyqmd_mlx.llm.expand import _expand_cache, _seed_for
from pyqmd_mlx.store import Store


def test_default_expand_model_is_a_hub_repo_id():
    assert DEFAULT_EXPAND_MODEL == "after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6"
    assert EXPAND_MODEL_ENV_VAR == "PYQMD_EXPAND_MODEL"


def test_resolve_expand_model_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv(EXPAND_MODEL_ENV_VAR, raising=False)
    assert resolve_expand_model() == DEFAULT_EXPAND_MODEL


def test_resolve_expand_model_defaults_when_env_empty(monkeypatch):
    monkeypatch.setenv(EXPAND_MODEL_ENV_VAR, "")
    assert resolve_expand_model() == DEFAULT_EXPAND_MODEL


def test_resolve_expand_model_uses_env_value(monkeypatch):
    monkeypatch.setenv(EXPAND_MODEL_ENV_VAR, "/opt/models/qmd-query-expansion-1.7b-mlx")
    assert resolve_expand_model() == "/opt/models/qmd-query-expansion-1.7b-mlx"


def test_store_uses_default_expand_model_when_env_unset(monkeypatch):
    monkeypatch.delenv(EXPAND_MODEL_ENV_VAR, raising=False)
    assert Store(":memory:")._expand_model == DEFAULT_EXPAND_MODEL


def test_store_picks_up_env_expand_model(monkeypatch):
    monkeypatch.setenv(EXPAND_MODEL_ENV_VAR, "someorg/custom-expander")
    assert Store(":memory:")._expand_model == "someorg/custom-expander"


def test_store_explicit_expand_model_beats_env(monkeypatch):
    monkeypatch.setenv(EXPAND_MODEL_ENV_VAR, "someorg/custom-expander")
    store = Store(":memory:", expand_model="someorg/explicit")
    assert store._expand_model == "someorg/explicit"


def _fake_mlx_lm(monkeypatch, load, generate=lambda *a, **k: ""):
    """Stand-in for mlx_lm so fast tests never import MLX (CI runs on Ubuntu)."""
    fake = types.SimpleNamespace(load=load, generate=generate)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)


def test_expand_query_wraps_load_failure(monkeypatch):
    original = ValueError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'")

    def _raise(model_id):
        raise original

    _fake_mlx_lm(monkeypatch, _raise)

    with pytest.raises(ExpansionModelError) as excinfo:
        expand_query("anything", model="/nonexistent/model-dir")

    message = str(excinfo.value)
    assert message.startswith("Could not load query-expansion model '/nonexistent/model-dir': ")
    assert "Repo id must be in the form" in message
    assert (
        "Set PYQMD_EXPAND_MODEL to a Hugging Face repo id or a local MLX model directory" in message
    )
    assert excinfo.value.__cause__ is original
    assert isinstance(excinfo.value, RuntimeError)


def test_failed_expansion_load_is_not_cached(monkeypatch):
    def _raise(model_id):
        raise OSError("offline")

    _fake_mlx_lm(monkeypatch, _raise)

    with pytest.raises(ExpansionModelError):
        expand_query("anything", model="someorg/unreachable")

    assert "someorg/unreachable" not in _expand_cache


def test_expansion_error_message_has_no_doubled_period(monkeypatch):
    def _raise(model_id):
        raise ValueError("Use `repo_type` argument if needed.")

    _fake_mlx_lm(monkeypatch, _raise)

    with pytest.raises(ExpansionModelError) as excinfo:
        expand_query("anything", model="/nonexistent/dir")

    assert "if needed. Set PYQMD_EXPAND_MODEL" in str(excinfo.value)


def test_seed_for_is_pinned():
    # sha256("pinned-model\npinned query")[:4], big-endian. Pinned so an
    # accidental change to the derivation (which would silently change
    # every query's results) fails loudly.
    assert _seed_for("pinned query", "pinned-model") == 3026612762


def test_seed_for_depends_on_query_and_model():
    base = _seed_for("heart attack risk", "org/model-a")
    assert _seed_for("heart attack risk", "org/model-a") == base
    assert _seed_for("heart attack risks", "org/model-a") != base
    assert _seed_for("heart attack risk", "org/model-b") != base
    assert 0 <= base < 2**32


def test_expand_query_samples_with_a_per_query_seed(monkeypatch):
    import pyqmd_mlx.llm.expand as expand_mod

    calls = []

    def fake_generate(model, tokenizer, prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return "lex: heart attack <think>x</think>\n\nvec: heart attack\u2028risk factors\n"

    seen_messages = []

    def apply_chat_template(messages, tokenize, add_generation_prompt):
        seen_messages.append(messages)
        return messages[-1]["content"]

    tokenizer = types.SimpleNamespace(apply_chat_template=apply_chat_template)
    _fake_mlx_lm(monkeypatch, lambda model_id: (object(), tokenizer), fake_generate)
    monkeypatch.setattr(expand_mod, "_expand_cache", {})
    monkeypatch.setattr(expand_mod, "_build_grammar_tables", lambda tok: "tables")
    monkeypatch.setattr(expand_mod, "_keyed_sampler", lambda seed: ("sampler", seed))
    monkeypatch.setattr(expand_mod, "_logits_processors", lambda: ["processors"])
    monkeypatch.setattr(expand_mod, "_grammar_processor", lambda tables: ("grammar", tables))

    model = "test/fake-expander"
    lines = expand_query("heart attack risk", model=model)
    expand_query("heart attack risk", model=model)
    expand_query("stroke risk", model=model)

    # Node keeps in-line think text and splits on "\n" only.
    assert lines == ["lex: heart attack <think>x</think>", "vec: heart attack\u2028risk factors"]
    assert calls[0]["prompt"] == "/no_think Expand this search query: heart attack risk"
    assert calls[0]["max_tokens"] == 600
    assert calls[0]["logits_processors"] == ["processors", ("grammar", "tables")]
    assert calls[0]["sampler"] == ("sampler", _seed_for("heart attack risk", model))
    assert calls[1]["sampler"] == calls[0]["sampler"]
    assert calls[2]["sampler"] == ("sampler", _seed_for("stroke risk", model))
    assert seen_messages[0] == [
        {"role": "system", "content": expand_mod._SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think Expand this search query: heart attack risk"},
    ]


def test_expand_query_reports_a_non_byte_level_tokenizer(monkeypatch):
    import pyqmd_mlx.llm.expand as expand_mod

    tokenizer = types.SimpleNamespace(
        get_vocab=lambda: {"☃": 0}, added_tokens_decoder={}, eos_token_ids={0}
    )
    _fake_mlx_lm(monkeypatch, lambda model_id: (object(), tokenizer))
    monkeypatch.setattr(expand_mod, "_expand_cache", {})

    with pytest.raises(ExpansionModelError, match="isn't byte-level BPE"):
        expand_query("q", model="test/snowman-tokenizer")
    assert "test/snowman-tokenizer" not in expand_mod._expand_cache


def test_system_prompt_is_node_llama_cpps_default():
    # node-llama-cpp v3.20.0, dist/config.js defaultChatSystemPrompt --
    # what Node's LlamaChatSession sends ahead of expandQuery's prompt.
    from pyqmd_mlx.llm.expand import _SYSTEM_PROMPT

    assert _SYSTEM_PROMPT == (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead "
        "of answering something incorrectly. If you don't know the answer to a question, don't "
        "share false information."
    )
