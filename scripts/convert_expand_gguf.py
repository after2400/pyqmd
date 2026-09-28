"""Convert Node qmd's query-expansion GGUF into the mixed 4/6-bit MLX model pyqmd hosts.

Source: tobil/qmd-query-expansion-1.7B-gguf (MIT), the weights Node qmd
ships. mlx_lm.convert can't read GGUF, so this maps the GGUF tensor names onto
Qwen3's HF layout itself, writes a bf16 MLX dir, then quantizes it with
mlx_lm.convert using llama.cpp's Q4_K_M layer choices. See
docs/specs/2026-09-24-expand-model-gguf-weights-design.md and
docs/specs/2026-09-25-expand-model-requantize-design.md.

usage: uv run scripts/convert_expand_gguf.py <out-dir>
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

GGUF_REPO = "tobil/qmd-query-expansion-1.7B-gguf"
GGUF_FILE = "qmd-query-expansion-1.7B-f16.gguf"
GGUF_REVISION = "7816de0b72572c6c860ca1eddf97ba9e7fb8cc65"
BASE_REPO = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]
SMOKE_QUERIES = ["how long to cook noodles", "git rebase vs merge", "who is TDS motorsports"]
NEW_REPO = "after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6"
_GROUP_SIZE = 64

_TOP_LEVEL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
# llama.cpp's Qwen2/Qwen3 converter doesn't permute q/k, so names map 1:1.
_BLOCK = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.([a-z_]+)\.(weight|bias)$")


def gguf_to_hf_name(name: str) -> str:
    if name in _TOP_LEVEL:
        return _TOP_LEVEL[name]
    match = _BLOCK_RE.match(name)
    if not match or match.group(2) not in _BLOCK:
        raise ValueError(f"unmapped GGUF tensor: {name}")
    layer, kind, suffix = match.groups()
    return f"model.layers.{layer}.{_BLOCK[kind]}.{suffix}"


def check_tensor_names(converted: set[str], expected: set[str]) -> None:
    """Fail unless the converted names match the base model's, ignoring
    lm_head (tied to embed_tokens in Qwen3-1.7B, so present on one side only)."""
    missing = sorted(expected - converted - {"lm_head.weight"})
    extra = sorted(converted - expected - {"lm_head.weight"})
    if missing or extra:
        raise ValueError(f"tensor name mismatch: missing={missing[:5]} extra={extra[:5]}")


def quant_bits(path: str, num_layers: int) -> int:
    """6 for the tied embedding/output head, and for v_proj/down_proj on the
    layers llama.cpp's Q4_K_M keeps at Q6_K; 4 for everything else.

    The layer rule is mlx-lm's mixed_4_6 (a copy of llama.cpp's
    use_more_bits). mixed_4_6 itself only raises an lm_head module, which a
    tied model doesn't have, so the embedding is handled here."""
    if path == "model.embed_tokens":
        return 6
    parts = path.split(".")
    if len(parts) > 2 and parts[1] == "layers" and ("v_proj" in path or "down_proj" in path):
        i = int(parts[2])
        n = num_layers
        if i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2:
            return 6
    return 4


def has_lex_and_vec(text: str) -> bool:
    prefixes = {line.split(":", 1)[0].strip().lower() for line in text.splitlines() if ":" in line}
    return {"lex", "vec"} <= prefixes


def render_model_card(mlx_lm_version: str) -> str:
    return f"""---
license: mit
library_name: mlx
base_model: {GGUF_REPO}
pipeline_tag: text-generation
tags:
  - mlx
  - qwen3
  - query-expansion
  - qmd
---

# qmd-query-expansion-1.7B (MLX, mixed 4/6-bit)

MLX mixed 4/6-bit (affine, group size 64) conversion of
[{GGUF_REPO}](https://huggingface.co/{GGUF_REPO}), the query-expansion
model [qmd](https://github.com/tobi/qmd) ships. It follows llama.cpp's
`Q4_K_M` layer choices: the tied embedding/output head, and `v_proj` and
`down_proj` on the first and last eighth of layers and every third layer
between, are 6-bit; everything else is 4-bit. It's the default expansion
model of [pyqmd](https://github.com/after2400/pyqmd), qmd's Python/MLX port.

Given `/no_think Expand this search query: <query>` in the Qwen3 chat
template, it answers with typed lines:

```
hyde: a hypothetical passage that would answer the query
lex: keyword variation
vec: natural-language rephrasing
```

## Usage

```python
from mlx_lm import generate, load

model, tokenizer = load("{NEW_REPO}")
prompt = tokenizer.apply_chat_template(
    [{{"role": "user", "content": "/no_think Expand this search query: auth config"}}],
    tokenize=False,
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400))
```

## Provenance

- Weights: `{GGUF_FILE}` from `{GGUF_REPO}` at revision `{GGUF_REVISION}`.
- Config and tokenizer: `{BASE_REPO}` at revision `{BASE_REVISION}`.
- Converted with pyqmd's `scripts/convert_expand_gguf.py` (GGUF tensor names
  mapped onto Qwen3's layout, bf16, then `mlx_lm.convert` with a mixed
  4/6-bit `quant_predicate`, group size 64),
  mlx-lm {mlx_lm_version}.

## License

MIT, following the upstream fine-tune (`{GGUF_REPO}`). Base model:
[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0).
"""


def _smoke_check(out_dir: Path) -> list[str]:
    from mlx_lm import generate, load

    model, tokenizer = load(str(out_dir))
    failures = []
    for query in SMOKE_QUERIES:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"/no_think Expand this search query: {query}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = generate(model, tokenizer, prompt=prompt, max_tokens=400)
        print(f"--- {query}\n{text}\n")
        if not has_lex_and_vec(text):
            failures.append(query)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="directory to create (must not exist)")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        print(f"error: {args.out_dir} already exists; pick a fresh directory", file=sys.stderr)
        return 1

    import mlx.core as mx
    import mlx_lm
    from huggingface_hub import hf_hub_download
    from mlx_lm import convert

    gguf_path = hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REVISION)
    index_path = hf_hub_download(BASE_REPO, "model.safetensors.index.json", revision=BASE_REVISION)
    expected = set(json.loads(Path(index_path).read_text())["weight_map"])

    with tempfile.TemporaryDirectory() as tmp:
        bf16_dir = Path(tmp) / "bf16"
        bf16_dir.mkdir()
        for name in BASE_FILES:
            shutil.copy(hf_hub_download(BASE_REPO, name, revision=BASE_REVISION), bf16_dir / name)

        weights = {gguf_to_hf_name(k): v for k, v in mx.load(gguf_path).items()}
        check_tensor_names(set(weights), expected)
        tied = mx.array_equal(
            weights["lm_head.weight"], weights["model.embed_tokens.weight"]
        ).item()
        if tied:
            del weights["lm_head.weight"]
        config_path = bf16_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["tie_word_embeddings"] = tied
        config_path.write_text(json.dumps(config, indent=2))
        num_layers = config["num_hidden_layers"]
        mx.save_safetensors(
            str(bf16_dir / "model.safetensors"),
            {k: v.astype(mx.bfloat16) for k, v in weights.items()},
            metadata={"format": "mlx"},
        )

        def predicate(path, _module, *_):
            bits = quant_bits(path, num_layers)
            return {"group_size": _GROUP_SIZE, "bits": bits, "mode": "affine"}

        # mlx_lm.load (inside convert) is strict, so any shape mismatch fails here.
        convert(
            str(bf16_dir),
            mlx_path=str(args.out_dir),
            quantize=True,
            q_bits=4,
            q_group_size=_GROUP_SIZE,
            quant_predicate=predicate,
        )

    (args.out_dir / "README.md").write_text(render_model_card(mlx_lm_version=mlx_lm.__version__))

    failures = _smoke_check(args.out_dir)
    if failures:
        print(f"error: smoke check failed (no lex:+vec: lines) for: {failures}", file=sys.stderr)
        return 1
    print(
        f"OK: wrote {args.out_dir} (gguf {GGUF_REVISION[:7]}, base {BASE_REVISION[:7]}, "
        f"mlx-lm {mlx_lm.__version__}, tie_word_embeddings={tied})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
