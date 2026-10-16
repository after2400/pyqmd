import pytest

from pyqmd_mlx.llm._prompts import (
    build_rerank_prompt,
    format_doc_for_embedding,
    format_embedding_inputs,
    format_query_for_embedding,
    is_qwen3_embedding_model,
    score_from_logits,
)


def test_is_qwen3_embedding_model_matches_qwen_embed_names():
    assert is_qwen3_embedding_model("mlx-community/Qwen3-Embedding-0.6B")
    assert is_qwen3_embedding_model("hf:some/embed-qwen-model")
    assert not is_qwen3_embedding_model("mlx-community/embeddinggemma-300m-8bit")


def test_format_query_for_embedding_default_nomic_style():
    result = format_query_for_embedding("cats", "mlx-community/embeddinggemma-300m-8bit")
    assert result == "task: search result | query: cats"


def test_format_query_for_embedding_qwen3_style():
    result = format_query_for_embedding("cats", "mlx-community/Qwen3-Embedding-0.6B")
    assert result == "Instruct: Retrieve relevant documents for the given query\nQuery: cats"


def test_format_doc_for_embedding_default_nomic_style():
    result = format_doc_for_embedding("cats are great", "mlx-community/embeddinggemma-300m-8bit")
    assert result == "title: none | text: cats are great"


def test_format_doc_for_embedding_default_nomic_style_with_title():
    result = format_doc_for_embedding(
        "cats are great", "mlx-community/embeddinggemma-300m-8bit", title="Cats"
    )
    assert result == "title: Cats | text: cats are great"


def test_format_doc_for_embedding_qwen3_style_no_title():
    result = format_doc_for_embedding("cats are great", "mlx-community/Qwen3-Embedding-0.6B")
    assert result == "cats are great"


def test_format_doc_for_embedding_qwen3_style_with_title():
    result = format_doc_for_embedding(
        "cats are great", "mlx-community/Qwen3-Embedding-0.6B", title="Cats"
    )
    assert result == "Cats\ncats are great"


def test_build_rerank_prompt_contains_query_and_document():
    prompt = build_rerank_prompt("what is mlx", "mlx is an array framework")
    assert prompt.startswith("<|im_start|>system\n")
    assert "<Query>: what is mlx" in prompt
    assert "<Document>: mlx is an array framework" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_score_from_logits_favors_higher_true_logit():
    high = score_from_logits(true_logit=5.0, false_logit=0.0)
    low = score_from_logits(true_logit=0.0, false_logit=5.0)
    assert high > 0.9
    assert low < 0.1


def test_score_from_logits_equal_logits_is_half():
    assert abs(score_from_logits(1.0, 1.0) - 0.5) < 1e-9


def test_format_embedding_inputs_documents_get_the_title():
    assert format_embedding_inputs(
        ["a", "b"], "mlx-community/embeddinggemma-300m-8bit", "document", "T"
    ) == ["title: T | text: a", "title: T | text: b"]


def test_format_embedding_inputs_queries():
    assert format_embedding_inputs(["q"], "mlx-community/embeddinggemma-300m-8bit", "query") == [
        "task: search result | query: q"
    ]


GEMMA = "mlx-community/embeddinggemma-300m-8bit"
QWEN = "mlx-community/Qwen3-Embedding-0.6B"


def test_format_embedding_inputs_accepts_one_title_per_text():
    out = format_embedding_inputs(["a", "b"], GEMMA, "document", ["Doc > S1", "Doc > S2"])
    assert out == ["title: Doc > S1 | text: a", "title: Doc > S2 | text: b"]


def test_format_embedding_inputs_none_entries_render_as_none():
    out = format_embedding_inputs(["a", "b"], GEMMA, "document", ["T", None])
    assert out == ["title: T | text: a", "title: none | text: b"]


def test_format_embedding_inputs_per_text_titles_for_qwen_models():
    out = format_embedding_inputs(["a", "b"], QWEN, "document", ["T > S", None])
    assert out == ["T > S\na", "b"]


def test_format_embedding_inputs_a_single_title_is_unchanged():
    out = format_embedding_inputs(["a", "b"], GEMMA, "document", "T")
    assert out == ["title: T | text: a", "title: T | text: b"]


def test_format_embedding_inputs_rejects_a_title_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="2 entries for 3 texts"):
        format_embedding_inputs(["a", "b", "c"], GEMMA, "document", ["x", "y"])


def test_format_embedding_inputs_queries_ignore_titles():
    out = format_embedding_inputs(["q"], GEMMA, "query", ["wrong", "length"])
    assert out == format_embedding_inputs(["q"], GEMMA, "query")
