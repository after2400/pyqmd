"""Replay Store.query's hybrid pipeline over the SciFact qrels queries,
with each known pyqmd-vs-Node difference behind a switch, and report
MRR / nDCG@10 / Recall@10. Built for the 2026-09-24 Recall@10 gap spike
(see the roadmap's "Investigate pyqmd's Recall@10 gap against Node" entry).

The pipeline here is a copy of Store.query, not a call into it, so it can
drift as Store.query changes. Run with --check-store first: with default
flags it should reproduce Store.query's rankings exactly.

Switches:
  --depth   per-list retrieval depth (pyqmd: 40 = candidate_limit, Node: 20)
  --blend   'pyqmd' ((rrf/max + rerank)/2) | 'node' (tiered 1/rank blend)
  --expand  'seeded' (what ships: expand_query) | 'seeded:<salt>' (same
            sampler, seed = sha256(model, query, salt)) | 'greedy' (the
            pre-2026-09-25 decoding, for comparison) | 'none'
  --order   'pyqmd' (vec, fts, expansions) | 'node' (fts, lex..., vec, vec/hyde...)
  --drop    comma-separated expansion types to discard (lex, vec, hyde)

The summary line counts, per run: fallback (postprocess_expansion fell back
to its default set), malformed (an expansion with no parseable
lex:/vec:/hyde: line), and runon (an expansion with a line over
RUNON_CHARS characters).

The first run indexes data/scifact/corpus (built by
prepare_scifact_corpus.py) into <work-dir>/scifact-index.sqlite, the same
way parity/conftest.py's indexed_pyqmd_store does. Expansions and rerank
scores are cached in <work-dir>/cache.json, keyed by query (and expansion
mode), so re-running a variant only pays for what changed. Delete the cache
after changing a model or the rerank input.

Usage: uv run scripts/replay_query.py [--expand seeded:1] [--trace] ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # for parity._quality_baseline

from parity._quality_baseline import compute_metrics  # noqa: E402
from pyqmd_mlx.bench._metrics import recall_at_k  # noqa: E402
from pyqmd_mlx.llm import expand_query, rerank, resolve_expand_model  # noqa: E402
from pyqmd_mlx.llm._cache import get_or_load  # noqa: E402
from pyqmd_mlx.llm._constants import DEFAULT_RERANK_MODEL  # noqa: E402
from pyqmd_mlx.llm.expand import (  # noqa: E402
    _THINK_BLOCK_RE,
    _expand_cache,
    _generate_expansion,
    _load,
    _seed_for,
)
from pyqmd_mlx.store import Store  # noqa: E402
from pyqmd_mlx.store._chunking import chunk_document  # noqa: E402
from pyqmd_mlx.store._expansion import parse_expanded_lines, postprocess_expansion  # noqa: E402
from pyqmd_mlx.store._indexing import scan_and_register_collection  # noqa: E402
from pyqmd_mlx.store._rrf import get_hybrid_rrf_weights, reciprocal_rank_fusion  # noqa: E402
from pyqmd_mlx.store._types import RankedListMeta, RankedResult  # noqa: E402
from pyqmd_mlx.store.store import STRONG_SIGNAL_MIN_GAP, STRONG_SIGNAL_MIN_SCORE  # noqa: E402

DATA_DIR = REPO / "data" / "scifact"
DEFAULT_WORK_DIR = DATA_DIR / "replay"
CANDIDATE_LIMIT = 40
FINAL_LIMIT = 20
COLLECTION = "scifact"
# A line this long is a run-on: Node's grammar allows EOS only after "\n",
# so its last hyde: line sometimes rambles (21 of 150 Node expansions).
RUNON_CHARS = 600


def build_index(db: Path) -> None:
    corpus = DATA_DIR / "corpus"
    print(f"indexing {corpus} into {db} (one-time)...", file=sys.stderr)
    db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(str(db))
    store.add_collection(COLLECTION, str(corpus))
    scan_and_register_collection(store, str(corpus), "**/*.md", COLLECTION)
    for row in store.get_indexable_content(COLLECTION):
        store.index_content(row["hash"], row["doc"])
    store.close()


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"expand": {}, "rerank": {}}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache))


def load_queries():
    fixture = json.loads((DATA_DIR / "rerank-fixture.json").read_text())
    qrels: dict[str, set[str]] = {}
    with (DATA_DIR / "qrels-test.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return [(str(e["query_id"]), e["query"], qrels.get(str(e["query_id"]), set())) for e in fixture]


EXPANSION_SECONDS: list[float] = []


def greedy_expand(query: str, model_id: str) -> list[str]:
    """pyqmd's decoding before 2026-09-25: greedy, max_tokens=400."""
    from mlx_lm import generate

    model, tokenizer, _ = get_or_load(_expand_cache, model_id, _load)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"/no_think Expand this search query: {query}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = generate(model, tokenizer, prompt=prompt, max_tokens=400)
    text = _THINK_BLOCK_RE.sub("", text).strip()
    return [line for line in text.splitlines() if line.strip()]


def get_expansion(cache, query, mode, model_id) -> list[str]:
    if mode == "none":
        return []
    key = f"{mode}|{query}"
    if key not in cache["expand"]:
        start = time.perf_counter()
        if mode == "greedy":
            lines = greedy_expand(query, model_id)
        elif mode == "seeded":
            lines = expand_query(query, model_id)
        elif mode.startswith("seeded:"):
            salt = mode.split(":", 1)[1]
            lines = _generate_expansion(query, model_id, _seed_for(query, model_id, salt))
        else:
            raise SystemExit(f"unknown --expand mode: {mode}")
        EXPANSION_SECONDS.append(time.perf_counter() - start)
        cache["expand"][key] = lines
    return cache["expand"][key]


def get_rerank_scores(cache, query, docs) -> list[float]:
    keys = [f"{query}|{hashlib.sha1(d.encode()).hexdigest()}" for d in docs]
    missing = [(k, d) for k, d in zip(keys, docs) if k not in cache["rerank"]]
    if missing:
        scores = rerank(query, [d for _, d in missing], DEFAULT_RERANK_MODEL)
        for (k, _), s in zip(missing, scores):
            cache["rerank"][k] = s
    return [cache["rerank"][k] for k in keys]


def to_ranked(results):
    return [
        RankedResult(
            file=r.filepath, display_path=r.display_path, title=r.title, body=r.body, score=r.score
        )
        for r in results
    ]


def run_query(store, cache, query, cfg, model_id, trace=None):
    depth = cfg["depth"]
    fts = lambda q: store.search_fts(q, limit=depth, collection=COLLECTION)  # noqa: E731
    vec = lambda q: store.search_vec(q, limit=depth, collection=COLLECTION)  # noqa: E731

    probe = store.search_fts(query, limit=2, collection=COLLECTION)
    strong = (
        bool(probe)
        and probe[0].score >= STRONG_SIGNAL_MIN_SCORE
        and probe[0].score - (probe[1].score if len(probe) > 1 else 0.0) >= STRONG_SIGNAL_MIN_GAP
    )
    lines = [] if strong else get_expansion(cache, query, cfg["expand"], model_id)
    parts = [] if strong else postprocess_expansion(query, lines)
    expanded = not strong and cfg["expand"] != "none"
    if cfg["expand"] == "none":
        parts = []
    parts = [p for p in parts if p.type not in cfg["drop"]]

    orig_vec = ("vec", "original", query, vec(query))
    orig_fts = ("fts", "original", query, fts(query))
    exp = [
        ("fts", p.type, p.query, fts(p.query))
        if p.type == "lex"
        else ("vec", p.type, p.query, vec(p.query))
        for p in parts
    ]
    if cfg["order"] == "node":
        entries = (
            [orig_fts]
            + [e for e in exp if e[0] == "fts"]
            + [orig_vec]
            + [e for e in exp if e[0] == "vec"]
        )
        entries = [e for e in entries if e[3]]  # Node skips empty lists
    else:
        entries = [orig_vec, orig_fts] + exp

    lists = [to_ranked(e[3]) for e in entries]
    meta = [RankedListMeta(source=e[0], query_type=e[1], query=e[2]) for e in entries]
    fused = reciprocal_rank_fusion(lists, get_hybrid_rrf_weights(meta))[:CANDIDATE_LIMIT]

    query_terms = {t.lower() for t in query.split() if len(t) > 2}
    cands = []
    for r in fused:
        chunks = chunk_document(r.body, filepath=r.file, chunk_strategy="regex")
        best_text, best_score = chunks[0][0], -1.0
        for text, _pos in chunks:
            s = sum(1.0 for t in query_terms if t in text.lower())
            if s > best_score:
                best_score, best_text = s, text
        cands.append((r, best_text))

    scores = get_rerank_scores(cache, query, [c[1] for c in cands])
    final = {}
    if cfg["blend"] == "node":
        for i, ((r, _), rs) in enumerate(zip(cands, scores)):
            rank = i + 1
            w = 0.75 if rank <= 3 else 0.60 if rank <= 10 else 0.40
            final[r.file] = w * (1 / rank) + (1 - w) * rs
    else:
        max_rrf = max((c[0].score for c in cands), default=1.0) or 1.0
        for (r, _), rs in zip(cands, scores):
            final[r.file] = (r.score / max_rrf + rs) / 2

    ranked = sorted(final, key=lambda f: final[f], reverse=True)[:FINAL_LIMIT]
    stem = lambda f: Path(f).stem  # noqa: E731

    if trace is not None:
        trace.update(
            strong=strong,
            parts=[(p.type, p.query) for p in parts],
            fallback=any(
                p.type == "hyde" and p.query == f"Information about {query}" for p in parts
            ),
            malformed=expanded and not parse_expanded_lines(lines),
            runon=expanded and any(len(line) > RUNON_CHARS for line in lines),
            lists=[(e[0], e[1], e[2], [stem(x.filepath) for x in e[3]]) for e in entries],
            fused=[stem(r.file) for r in fused],
            rerank={stem(c[0].file): s for c, s in zip(cands, scores)},
            final=[(stem(f), final[f]) for f in ranked],
        )
    return [stem(f) for f in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--blend", default="pyqmd")
    ap.add_argument("--expand", default="seeded")
    ap.add_argument("--order", default="pyqmd")
    ap.add_argument("--drop", default="", help="comma-separated expansion types to discard")
    ap.add_argument("--trace", action="store_true", help="print per-stage ranks for missed queries")
    ap.add_argument("--dump", help="write per-query ranked lists to this JSON file")
    ap.add_argument("--check-store", action="store_true", help="compare against real Store.query")
    ap.add_argument(
        "--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="index + cache location"
    )
    args = ap.parse_args()
    cfg = dict(
        depth=args.depth,
        blend=args.blend,
        expand=args.expand,
        order=args.order,
        drop=[t for t in args.drop.split(",") if t],
    )

    db = args.work_dir / "scifact-index.sqlite"
    cache_path = args.work_dir / "cache.json"
    if not db.exists():
        build_index(db)
    store = Store(str(db))
    cache = load_cache(cache_path)
    model_id = resolve_expand_model()
    if args.expand != "none":
        get_or_load(_expand_cache, model_id, _load)
    queries = load_queries()
    ranked_lists, relevant, missed, dump = [], [], [], {}
    try:
        for qid, q, rel in queries:
            tr = {}
            ranked = run_query(store, cache, q, cfg, model_id, trace=tr)
            ranked_lists.append(ranked)
            relevant.append(rel)
            dump[qid] = {"query": q, "relevant": sorted(rel), **tr}
            if recall_at_k(ranked, rel, k=10) < 1.0:
                missed.append(qid)
            if args.check_store:
                real = [
                    Path(r.file).stem
                    for r in store.query(q, limit=FINAL_LIMIT, collection=COLLECTION)
                ]
                if real != ranked:
                    print(f"MISMATCH q{qid}: replay={ranked[:10]} store={real[:10]}")
    finally:
        save_cache(cache_path, cache)

    m = compute_metrics(ranked_lists, relevant)
    fallbacks = sum(1 for d in dump.values() if d.get("fallback"))
    malformed = sum(1 for d in dump.values() if d.get("malformed"))
    runons = sum(1 for d in dump.values() if d.get("runon"))
    print(
        f"{cfg}  mrr={m['mrr']:.4f} ndcg@10={m['ndcg_at_10']:.4f} "
        f"recall@10={m['recall_at_10']:.4f}  fallback={fallbacks}  "
        f"malformed={malformed}  runon={runons}  missed={missed}"
    )
    if EXPANSION_SECONDS:
        print(
            f"expansion: {len(EXPANSION_SECONDS)} generated, "
            f"mean {sum(EXPANSION_SECONDS) / len(EXPANSION_SECONDS):.2f}s, "
            f"max {max(EXPANSION_SECONDS):.2f}s",
            file=sys.stderr,
        )
    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, indent=1))

    if args.trace:
        for qid in missed:
            d = dump[qid]
            print(f"\n=== q{qid}: {d['query']}  relevant={d['relevant']}  strong={d['strong']}")
            for rel_id in d["relevant"]:
                for src, qtype, text, ids in d["lists"]:
                    pos = ids.index(rel_id) + 1 if rel_id in ids else "-"
                    print(f"  {src}/{qtype:8} rank={pos!s:>3}  {text[:80]}")
                fpos = d["fused"].index(rel_id) + 1 if rel_id in d["fused"] else "-"
                final_ids = [f for f, _ in d["final"]]
                rpos = final_ids.index(rel_id) + 1 if rel_id in final_ids else "-"
                print(
                    f"  fused rank={fpos}  rerank={d['rerank'].get(rel_id, '-')}  final rank={rpos}"
                )
                print(
                    f"  final top10 rerank scores: {[round(d['rerank'][f], 3) for f, _ in d['final'][:10]]}"
                )


if __name__ == "__main__":
    main()
