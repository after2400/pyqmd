import importlib
from types import SimpleNamespace

import pytest

from pyqmd_mlx.llm import embed

# The package re-exports the embed() function under the submodule's name,
# so `import pyqmd_mlx.llm.embed as m` would bind the function.
embed_module = importlib.import_module("pyqmd_mlx.llm.embed")

MODEL = "mlx-community/embeddinggemma-300m-8bit"


class _StubTokenizer:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length
        self.calls = []

    def batch_encode_plus(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": "ids", "attention_mask": "mask"}


class _StubModel:
    def __call__(self, input_ids, attention_mask=None):
        return SimpleNamespace(text_embeds=SimpleNamespace(tolist=lambda: [[1.0, 0.0]]))


def _stub(monkeypatch, model_max_length=2048):
    tokenizer = _StubTokenizer(model_max_length)
    monkeypatch.setattr(
        embed_module, "get_or_load", lambda cache, model, load: (_StubModel(), tokenizer)
    )
    return tokenizer


def test_document_texts_are_formatted_with_the_title(monkeypatch):
    tokenizer = _stub(monkeypatch)
    embed(["chunk one"], MODEL, kind="document", title="Rivers")
    assert tokenizer.calls[0][0] == ["title: Rivers | text: chunk one"]


def test_query_texts_ignore_the_title(monkeypatch):
    tokenizer = _stub(monkeypatch)
    embed(["deltas"], MODEL, kind="query", title="Rivers")
    assert tokenizer.calls[0][0] == ["task: search result | query: deltas"]


@pytest.mark.parametrize(
    ("model_max_length", "expected"), [(2048, 2048), (512, 512), (10**30, 2048)]
)
def test_truncates_at_the_smaller_of_2048_and_the_model_limit(
    monkeypatch, model_max_length, expected
):
    tokenizer = _stub(monkeypatch, model_max_length)
    embed(["x"], MODEL, kind="document")
    kwargs = tokenizer.calls[0][1]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == expected


def test_bad_kind_raises_before_loading_the_model(monkeypatch):
    def fail_load(*args):
        raise AssertionError("model loaded")

    monkeypatch.setattr(embed_module, "get_or_load", fail_load)
    with pytest.raises(ValueError, match="kind must be"):
        embed(["x"], MODEL, kind="passage")
