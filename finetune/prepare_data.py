#!/usr/bin/env python3
"""Prepare Node qmd's expansion SFT data for an MLX LoRA run on a non-Qwen base.

usage: python finetune/prepare_data.py --node-finetune <qmd checkout>/finetune \
           --model ibm-granite/granite-4.0-1b --out data/granite

Differences from finetune/dataset/prepare_data.py (Node): prompt without
`/no_think`, no pyqmd system message (the template's own default, if any, applies
at train and serve alike), and the target ends with "\n" so the model learns
"...\n<EOS>", which pyqmd's output grammar allows (it forbids EOS mid-line).
Writes mlx_lm.lora's {"text": ...} format, so the template can't trim the "\n".
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

NODE_FINETUNE = (
    Path(sys.argv[sys.argv.index("--node-finetune") + 1]) if "--node-finetune" in sys.argv else None
)
if NODE_FINETUNE is None:
    sys.exit("--node-finetune <path to qmd/finetune> is required")
sys.path.insert(0, str(NODE_FINETUNE))
from dataset.schema import load_examples, output_items_to_text  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from pyqmd_mlx.llm.expand import _PLAIN_STYLE, _QWEN3_STYLE, _messages  # noqa: E402

# Run with pyqmd importable (PYTHONPATH=src), and not with `python -I`.
# The training prompt comes from the same function pyqmd serves with, so the two can't drift.
STYLES = {"plain": _PLAIN_STYLE, "qwen3": _QWEN3_STYLE}


def render(tok, query, intent, output_text, style) -> str:
    messages = _messages(query, style)
    if intent:
        messages[-1]["content"] += f"\nQuery intent: {intent.strip()}"
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + output_text + "\n" + tok.eos_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-finetune", required=True)
    ap.add_argument("--model", required=True, help="model dir/repo for the tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--prompt-style",
        choices=sorted(STYLES),
        default="plain",
        help="plain: bare user message; qwen3: '/no_think' plus Node's system prompt",
    )
    ap.add_argument("--split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    exs, seen = [], set()
    for f in sorted(glob.glob(str(Path(a.node_finetune) / "data" / "*.jsonl"))):
        for ex in load_examples(f):
            k = ex.query.lower().strip()
            if k not in seen:
                seen.add(k)
                exs.append(ex)
    random.seed(a.seed)
    random.shuffle(exs)
    cut = int(len(exs) * (1 - a.split))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", exs[:cut]), ("valid", exs[cut:])):
        with open(out / f"{name}.jsonl", "w") as fh:
            for ex in part:
                fh.write(
                    json.dumps(
                        {
                            "text": render(
                                tok,
                                ex.query,
                                ex.intent,
                                output_items_to_text(ex.output),
                                STYLES[a.prompt_style],
                            )
                        }
                    )
                    + "\n"
                )
        print(name, len(part))
    sample = json.loads(open(out / "train.jsonl").readline())["text"]
    print(repr(sample[:300]))
    print("...", repr(sample[-80:]))
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    print(
        "ends with eos id:",
        ids[-1] == tok.eos_token_id,
        "| prev token:",
        repr(tok.convert_ids_to_tokens(ids[-2])),
        "| bos first:",
        ids[0] == tok.bos_token_id,
    )
    lens = [len(tok(json.loads(line)["text"])["input_ids"]) for line in open(out / "train.jsonl")]
    print("tokens per example: max", max(lens), "mean", sum(lens) / len(lens))


main()
