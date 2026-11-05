#!/usr/bin/env python3
"""usage: fuse_quantize.py <base-model-dir> <adapter-dir> <checkpoint-file-or-'final'> <out-root>
Fuses a LoRA checkpoint into bf16 weights (out-root/bf16) and quantizes with the
mixed 4/6-bit recipe (out-root/mlx-mixed-4-6) via scripts/convert_expand_gguf.quant_bits."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

base, adapters, ckpt, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
use = out / "adapter"
use.mkdir(exist_ok=True)
shutil.copy(adapters / "adapter_config.json", use / "adapter_config.json")
shutil.copy(
    adapters / ("adapters.safetensors" if ckpt == "final" else ckpt), use / "adapters.safetensors"
)
bf16 = out / "bf16"
if bf16.exists():
    shutil.rmtree(bf16)
subprocess.run(
    [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        base,
        "--adapter-path",
        str(use),
        "--save-path",
        str(bf16),
    ],
    check=True,
)
spec = importlib.util.spec_from_file_location(
    "conv", Path(__file__).resolve().parent.parent / "scripts" / "convert_expand_gguf.py"
)
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
from mlx_lm import convert  # noqa: E402

n = json.loads((bf16 / "config.json").read_text())["num_hidden_layers"]
q = out / "mlx-mixed-4-6"
if q.exists():
    shutil.rmtree(q)
convert(
    str(bf16),
    mlx_path=str(q),
    quantize=True,
    q_bits=4,
    q_group_size=64,
    quant_predicate=lambda path, _m, *_: {
        "group_size": 64,
        "bits": conv.quant_bits(path, n),
        "mode": "affine",
    },
)
print("done", q)
