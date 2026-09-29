"""Slow integration tests that load real MLX models. Run with: uv run pytest -m slow"""

import math
import re

import pytest

from pyqmd_mlx.llm import embed, expand_query, rerank, resolve_expand_model


@pytest.mark.slow
def test_embed_returns_normalized_vectors_of_expected_dimension():
    vectors = embed(["hello world", "goodbye world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 768  # embeddinggemma-300M output dimension
    for vector in vectors:
        assert all(math.isfinite(x) for x in vector)


@pytest.mark.slow
def test_rerank_scores_relevant_document_higher():
    query = "What is the boiling point of water at sea level?"
    documents = [
        "Water boils at 100 degrees Celsius at standard sea-level atmospheric pressure.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    ]

    scores = rerank(query, documents)

    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]


@pytest.mark.slow
def test_rerank_scores_equal_full_sequence_logits_scores():
    # rerank() applies the output layer to the last position only; the
    # reference below is the original all-positions computation. Scores
    # must match exactly, not approximately.
    import importlib

    import mlx.core as mx

    from pyqmd_mlx.llm._prompts import build_rerank_prompt, score_from_logits

    rerank_module = importlib.import_module("pyqmd_mlx.llm.rerank")
    query = "How do I rotate the API keys?"
    documents = [
        "Rotate API keys from the settings page, then restart the service.",
        "The office is closed on public holidays.",
        "Key rotation checklist. " + "Revoke the old key after the new one is live. " * 150,
    ]

    scores = rerank(query, documents)

    model, tokenizer = rerank_module._rerank_cache[rerank_module.DEFAULT_RERANK_MODEL]
    yes = tokenizer.convert_tokens_to_ids("yes")
    no = tokenizer.convert_tokens_to_ids("no")
    reference = []
    for document in documents:
        token_ids = tokenizer.encode(build_rerank_prompt(query, document), add_special_tokens=False)
        last_logits = model(mx.array([token_ids]))[0, -1, :]
        reference.append(
            score_from_logits(float(last_logits[yes].item()), float(last_logits[no].item()))
        )
    assert scores == reference


@pytest.mark.slow
@pytest.mark.requires_expansion_weights
def test_expand_query_returns_nonempty_lines():
    lines = expand_query("authentication configuration", model=resolve_expand_model())

    assert len(lines) > 0
    assert all(isinstance(line, str) and line.strip() for line in lines)
    assert all(re.match(r"^(hyde|lex|vec):", line) for line in lines)
    assert {"lex", "vec"} <= {line.split(":", 1)[0] for line in lines}


@pytest.mark.slow
@pytest.mark.requires_expansion_weights
def test_expand_query_is_deterministic_per_query():
    model = resolve_expand_model()
    first = expand_query("authentication configuration", model=model)

    assert expand_query("authentication configuration", model=model) == first
    other = expand_query("heart attack risk factors", model=model)
    assert other
    assert other != first


@pytest.mark.slow
@pytest.mark.requires_expansion_weights
def test_expand_query_output_follows_nodes_grammar():
    model = resolve_expand_model()
    for query in [
        "authentication configuration",
        "heart attack risk factors",
        "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors.",
    ]:
        lines = expand_query(query, model=model)
        assert lines
        assert all(re.fullmatch(r"(lex|vec|hyde): [^\n]+", line) for line in lines), lines


@pytest.mark.slow
@pytest.mark.requires_expansion_weights
def test_vocab_token_bytes_agree_with_decode():
    from pyqmd_mlx.llm._cache import get_or_load
    from pyqmd_mlx.llm._expand_grammar import vocab_token_bytes
    from pyqmd_mlx.llm.expand import _expand_cache, _load

    _, tokenizer, _ = get_or_load(_expand_cache, resolve_expand_model(), _load)
    special = {tid for tid, t in tokenizer.added_tokens_decoder.items() if t.special}

    checked = 0
    for tid, data in enumerate(vocab_token_bytes(tokenizer)):
        if data is None:
            assert tid in special
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # a partial UTF-8 byte sequence; decode() would give U+FFFD
        assert tokenizer.decode([tid]) == text, tid
        checked += 1
    assert checked > 100_000
