# Fine-tuning a query-expansion model on a non-Qwen base (MLX, local)

Recipe behind `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6`, a fine-tune of
IBM's [Granite 4.0 1B](https://huggingface.co/ibm-granite/granite-4.0-1b) (Apache 2.0).
It follows Node qmd's SFT (`finetune/` in [tobi/qmd](https://github.com/tobi/qmd), MIT) but runs on
Apple silicon with `mlx_lm.lora`; the training data is Node's, pinned to the commit below.

Differences from Node's pipeline, and why:

- **No `/no_think`, no system message.** The prompt is `Expand this search query: <query>` (plus
  `\nQuery intent: ...` when an example has one). pyqmd serves a model with exactly the prompt
  it was trained on (`src/pyqmd_mlx/llm/expand.py`, `_PROMPT_STYLES`); only Qwen3 models get
  `/no_think` and Node's system prompt.
- **Each target ends with `\n`**, then the end-of-text token. pyqmd's output grammar allows
  end-of-text only after a newline, so a model trained without the newline runs on.
- **Targets are written as `{"text": ...}`** (prompt + target + `\n` + EOS) so no chat template
  can strip the newline.

## Steps

```sh
# Node's data, pinned to the commit that last changed it
git clone https://github.com/tobi/qmd && git -C qmd checkout 1fb2e2819e4024045203b4ea550ec793683baf2b

python finetune/prepare_data.py --node-finetune qmd/finetune \
    --model ibm-granite/granite-4.0-1b --out data/granite      # 2,806 train / 312 valid
mlx_lm.lora -c finetune/lora.yaml                              # ~30 min, ~8 GB, M4 Max
python finetune/fuse_quantize.py ibm-granite/granite-4.0-1b runs/granite/adapters final runs/granite/out
QMD_FINETUNE=qmd/finetune python finetune/score_expansions.py runs/granite/out/mlx-mixed-4-6 granite
```

`fuse_quantize.py` fuses the adapter into bf16 weights, then runs `mlx_lm.convert` with the same
mixed 4/6-bit recipe as `scripts/convert_expand_gguf.py` (`quant_bits`). To evaluate with
pyqmd: `PYQMD_EXPAND_MODEL=runs/granite/out/mlx-mixed-4-6 pyqmd query ...`.

Validation loss is lowest near iteration 1000 and rises afterwards (the run overfits), yet the
final checkpoint retrieves as well as iteration 1000 (see the spec's results), so the recipe keeps
Node's five epochs.
