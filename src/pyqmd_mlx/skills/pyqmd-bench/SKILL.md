---
name: pyqmd-bench
description: Measure pyqmd search quality with bench fixtures — build fixture, run bench, interpret recall/MRR/nDCG, tune, re-bench. Use when the user wants to know whether a config change helped or hurt search. Read-only — never mutates the index.
---

# pyqmd-bench

Decide with numbers, not vibes. You build fixtures, run
`pyqmd bench`, interpret per-backend metrics, and recommend tuning —
and you never mutate the index (re-embedding after a tuning decision
belongs to `pyqmd-librarian`, on explicit user request only).

## Fixture format

No bundled default fixture exists — the user (or you, if asked to
help build one) writes `<fixture.json>` by hand:

```json
{
  "description": "notes search quality",
  "version": 1,
  "collection": "notes",
  "queries": [
    {
      "id": "auth-timeout",
      "query": "why do database connections time out under load",
      "type": "question",
      "description": "proxies to the runbook, not the API ref",
      "expected_files": ["notes/runbooks/db-timeouts.md"],
      "expected_in_top_k": 3
    }
  ]
}
```

Required per query: `id`, `query`, `expected_files`,
`expected_in_top_k`. Top level: `description`, `version`,
`collection` (null or omitted for all collections). Structured
multi-line queries (`lex:`/`vec:`/`hyde:`/`intent:` prefixes) are
rejected at load — write single plain-text queries.

## Run

```sh
pyqmd bench <fixture.json> [-c <name>] [--json] [--samples N]
```

Scores all 4 retrieval backends (bm25 / vector / hybrid /
full-reranked) with recall, MRR, and nDCG. `--json` for structured
output; `-c` restricts to one collection.

hybrid and full expand the query with a sampled model, so one run is one
draw. `--samples N` (e.g. 5) re-runs them with N different seeds and
shows each metric as `mean min–max`, per query and in the summary. It
costs about N× the hybrid/full time (rerank scores are reused across
samples). bm25 and vector don't sample and run once.

## Interpret and tune

- bm25 wins, others lose → exact-term problem; check chunking and
  titles before touching embeddings.
- vector wins, hybrid loses → fusion/rerank suspect; try
  `--no-rerank` comparison runs and `--min-score` sweeps.
- All backends miss → fixture or index gap: wrong
  `expected_files`, missing `context`, or unindexed material (a
  librarian question, not a tuning knob).
- Change one thing at a time (chunking, reranking, `min-score`),
  re-bench the same fixture, and report deltas per backend — use
  `--samples 5` for both runs and treat a hybrid/full difference inside
  the summary's min–max range as noise; never average away the backend
  that matters most to the user.
