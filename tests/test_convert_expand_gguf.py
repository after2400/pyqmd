import pytest
from convert_expand_gguf import (
    BASE_REVISION,
    GGUF_REPO,
    GGUF_REVISION,
    NEW_REPO,
    check_tensor_names,
    gguf_to_hf_name,
    has_lex_and_vec,
    quant_bits,
    render_model_card,
)


@pytest.mark.parametrize(
    ("gguf", "hf"),
    [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
        ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
        ("blk.3.ffn_norm.weight", "model.layers.3.post_attention_layernorm.weight"),
        ("blk.12.attn_q_norm.weight", "model.layers.12.self_attn.q_norm.weight"),
        ("blk.12.attn_k_norm.weight", "model.layers.12.self_attn.k_norm.weight"),
        ("blk.27.attn_q.weight", "model.layers.27.self_attn.q_proj.weight"),
        ("blk.27.attn_k.weight", "model.layers.27.self_attn.k_proj.weight"),
        ("blk.27.attn_v.weight", "model.layers.27.self_attn.v_proj.weight"),
        ("blk.27.attn_output.weight", "model.layers.27.self_attn.o_proj.weight"),
        ("blk.5.ffn_gate.weight", "model.layers.5.mlp.gate_proj.weight"),
        ("blk.5.ffn_up.weight", "model.layers.5.mlp.up_proj.weight"),
        ("blk.5.ffn_down.weight", "model.layers.5.mlp.down_proj.weight"),
    ],
)
def test_gguf_to_hf_name_maps_every_qwen3_tensor_kind(gguf, hf):
    assert gguf_to_hf_name(gguf) == hf


@pytest.mark.parametrize(
    "name", ["blk.0.attn_qkv.weight", "rope_freqs.weight", "blk.x.ffn_up.weight"]
)
def test_gguf_to_hf_name_rejects_unknown_tensors(name):
    with pytest.raises(ValueError, match="unmapped GGUF tensor"):
        gguf_to_hf_name(name)


def test_check_tensor_names_accepts_match_ignoring_lm_head():
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    check_tensor_names(expected | {"lm_head.weight"}, expected)


def test_check_tensor_names_reports_missing_and_extra():
    with pytest.raises(ValueError, match="missing=.*model.norm.weight.*extra=.*model.bogus"):
        check_tensor_names(
            {"model.embed_tokens.weight", "model.bogus"},
            {"model.embed_tokens.weight", "model.norm.weight"},
        )


def test_has_lex_and_vec():
    assert has_lex_and_vec("hyde: passage: with colon\nlex: kw\nvec: phrase")
    assert not has_lex_and_vec("lex: kw only\nhyde: passage")
    assert not has_lex_and_vec("Here is a cooking guide.\nNote: boil water")


def test_model_card_states_license_source_and_provenance():
    card = render_model_card(mlx_lm_version="0.31.3")
    assert card.startswith("---\nlicense: mit\n")
    assert f"base_model: {GGUF_REPO}\n" in card
    assert GGUF_REVISION in card
    assert BASE_REVISION in card
    assert "mlx-lm 0.31.3" in card
    assert "scripts/convert_expand_gguf.py" in card
    assert NEW_REPO in card
    assert "mlx-4bit" not in card
    assert "Q4_K_M" in card


@pytest.mark.parametrize(
    ("path", "bits"),
    [
        ("model.embed_tokens", 6),
        ("model.layers.0.self_attn.v_proj", 6),  # first eighth
        ("model.layers.0.mlp.down_proj", 6),
        ("model.layers.5.self_attn.v_proj", 6),  # (5 - 3) % 3 == 2
        ("model.layers.5.mlp.down_proj", 6),
        ("model.layers.27.self_attn.v_proj", 6),  # last eighth
        ("model.layers.27.mlp.down_proj", 6),
        ("model.layers.3.self_attn.v_proj", 4),
        ("model.layers.4.mlp.down_proj", 4),
        ("model.layers.0.self_attn.q_proj", 4),
        ("model.layers.27.self_attn.o_proj", 4),
        ("model.layers.5.mlp.gate_proj", 4),
        ("model.layers.5.mlp.up_proj", 4),
    ],
)
def test_quant_bits_follows_q4_k_m_layout(path, bits):
    assert quant_bits(path, num_layers=28) == bits
