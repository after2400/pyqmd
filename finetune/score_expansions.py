#!/usr/bin/env python3
"""Score pyqmd's real expansion decode (sampled, grammar-constrained, seeded)
with Node's finetune reward rubric over evals/queries.txt.
usage: QMD_FINETUNE=<qmd checkout>/finetune python finetune/score_expansions.py <model> <label> [--salts 0,1,2]
Do not run it with python -I: that drops PYTHONPATH and can import another checkout."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

NODE = Path(os.environ["QMD_FINETUNE"])  # <qmd checkout>/finetune
sys.path.insert(0, str(NODE))
from reward import score_expansion_detailed  # noqa: E402

from pyqmd_mlx.llm.expand import expand_query  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("label")
ap.add_argument("--salts", default="0,1,2")
ap.add_argument("--out", default="results")
a = ap.parse_args()
queries = [
    line.strip()
    for line in (NODE / "evals/queries.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
salts = [None if s == "0" else s for s in a.salts.split(",")]
rows, t_all = [], []
for q in queries:
    for s in salts:
        t0 = time.time()
        lines = expand_query(q, a.model, salt=s)
        dt = time.time() - t0
        text = "\n".join(lines)
        r = score_expansion_detailed(q, text)
        runon = any(len(line) > 600 for line in lines)
        rows.append(
            {
                "query": q,
                "salt": s,
                "pct": r["percentage"],
                "rating": r["rating"],
                "runon": runon,
                "secs": dt,
                "text": text,
            }
        )
        t_all.append(dt)
pcts = [r["pct"] for r in rows]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"reward-{a.label}.json").write_text(json.dumps(rows, indent=1))
n = len(rows)
print(
    f"{a.label}: n={n} mean={sum(pcts) / n:.1f}% zero={sum(p == 0 for p in pcts)} "
    f"excellent={sum(r['rating'] == 'excellent' for r in rows)} runon={sum(r['runon'] for r in rows)} "
    f"mean_s={sum(t_all[1:]) / (n - 1):.2f} (excl. first/load) max_s={max(t_all):.2f}"
)

print(Counter(r["rating"] for r in rows))
