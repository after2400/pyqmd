from pyqmd_mlx.llm._prompts import (
    build_rerank_prompt,
    format_doc_for_embedding,
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
