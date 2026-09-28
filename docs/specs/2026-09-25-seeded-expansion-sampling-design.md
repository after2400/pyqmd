# Query expansion: seeded sampling and Node's system prompt instead of greedy decoding

**Date:** 2026-09-25
**Status:** Implemented (2026-09-25)
**Related:** `2026-09-10-python-mlx-rewrite-roadmap.md` (backlog entry
"Investigate pyqmd's Recall@10 gap against Node" and its 2026-09-24 spike
result); `2026-09-24-expand-model-gguf-weights-design.md` (kept greedy
decoding and deferred Node's sampling settings, which this design adopts);
`2026-09-24-qrels-margin-calibration-design.md` (the parity quality test
and its margins). §2 extends `scripts/replay_query.py`, which lands
with after2400/pyqmd#14.

## Problem

pyqmd's Recall@10 on SciFact's 30 qrels queries is 0.8333. Node's 30-run
mean is 0.8722, and the parity test passes with 0.007 to spare. The spike
traced the gap to how pyqmd decodes query expansion:

- `llm/expand.py` calls `mlx_lm.generate` with no sampler, so it decodes
  greedily with `max_tokens=400`. Node samples (`qmd/src/llm.ts`,
  `expandQuery`: temperature 0.7, topK 20, topP 0.8, presence penalty 0.5
  over the last 64 tokens, maxTokens 600), with a comment saying greedy
  decoding must not be used.
- Greedy output is consistently worse here. Its `vec:` lines drift away
  from the claim, and dropping them alone lifts recall to 0.900. Replaying
  the pipeline with Node's settings over 5 arbitrary global seeds gave
  Recall@10 0.867–0.933 (mean 0.893). All 5 beat greedy.
- The blend, retrieval depth, list order, and the user-message wording were
  each ruled out. The spike never varied the system prompt (see
  "Amendment").

The gguf-weights spec kept greedy because it's deterministic and passed the
18/18 format check. It never compared the two on retrieval quality, and it
listed Node's sampling settings as out of scope rather than rejected.

Node makes sampling repeatable by caching each expansion in `llm_cache` on
first use. That makes results depend on cache history, differ between
machines, and change after `cleanup`. It would also make `query` write to
the index. This design gets the same statistics a different way: each
query gets its own fixed seed, so no cache is needed.

### Amendment (2026-09-25): Node's system prompt

The first acceptance run (seeded sampling alone) failed §3's multi-salt
check: mean Recall@10 over salts 1–10 was 0.862 (0.827–0.927), below the
0.867 bar. The shipped seed scored 0.827, below greedy. Our draws also
spread about twice as widely as Node's 30 runs (σ 0.031 vs. 0.015).

A follow-up spike called Node's `LlamaCpp.expandQuery` directly (no store,
no `llm_cache`) for 5 draws per query and scored those expansions with
pyqmd's own retrieval and rerank through the replay harness:

| expansions                    | draws | Recall@10 mean (range) | σ     | MRR   | nDCG@10 | fallbacks / run |
| ----------------------------- | ----- | ---------------------- | ----- | ----- | ------- | --------------- |
| greedy                        | 1     | 0.833                  | –     | 0.784 | 0.787   | 0               |
| seeded                        | 10    | 0.862 (0.827–0.927)    | 0.031 | 0.771 | 0.785   | 0               |
| Node's own expansion text     | 5     | 0.884 (0.860–0.900)    | 0.019 | 0.777 | 0.796   | 0               |
| seeded + Node's system prompt | 10    | 0.884 (0.860–0.900)    | 0.018 | 0.793 | 0.807   | 1.9             |

- pyqmd's retrieval and rerank are fine. Given Node's expansion text, they
  score in line with Node's own 0.872. The whole gap is in the expansion
  text.
- Node's `session.prompt` goes through node-llama-cpp's `LlamaChatSession`,
  which prepends a default system message
  (`node-llama-cpp/dist/config.js`, `defaultChatSystemPrompt`, v3.20.0).
  pyqmd sent the user message alone. Adding that system message matches
  Node's mean and spread and beats greedy on all three metrics.
- The GGUF-vs-MLX quantization makes no measurable difference.
- Without Node's grammar, the system prompt sometimes pushes the model into
  commenting on the claim instead of expanding it ("The statement you
  provided is not supported by any known facts…"). That happened in 19 of
  300 draws (6%), and each fell back to `postprocess_expansion`'s default
  expansion. Node's grammar forbids that output. Recall didn't suffer here,
  but those queries lose their expansion. A format constraint is a separate
  roadmap item (see "Out of scope").

## Design

### 1. `llm/expand.py`

`expand_query(query, model)` keeps its signature and return value. Nothing
outside `llm/expand.py` changes in the product code.

**Decoding constants** at module level, each citing `qmd/src/llm.ts`
`expandQuery`:

| constant         | value |
| ---------------- | ----- |
| temperature      | 0.7   |
| top_k            | 20    |
| top_p            | 0.8   |
| presence penalty | 0.5   |
| presence context | 64    |
| `max_tokens`     | 600   |

Node's GBNF grammar stays out of scope (see below).

**`_SYSTEM_PROMPT`**: node-llama-cpp's `defaultChatSystemPrompt`, verbatim:

```text
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrectly. If you don't know the answer to a question, don't share false information.
```

It goes first in the chat-template messages, as `{"role": "system"}`,
before the unchanged user message. Its comment cites
`node-llama-cpp/dist/config.js` and says Node gets it implicitly through
`LlamaChatSession`.

**`_seed_for(query: str, model: str) -> int`**: the first 4 bytes of
`sha256(f"{model}\n{query}".encode("utf-8"))`, big-endian.

- `model` is the resolved id `expand_query` receives, so a
  `PYQMD_EXPAND_MODEL` override gets its own seeds. A local directory and a
  Hub id for the same weights seed differently. That's harmless.
- `--intent` isn't part of the seed, because expansion never sees the intent.

**`_keyed_sampler(seed: int)`**: returns a sampler for `mlx_lm.generate`
that holds its own PRNG key and never touches MLX's global RNG.

- mlx-lm's `make_sampler` always draws from the global RNG (its
  `categorical_sampling` is compiled over `mx.random.state`). mlx has no
  public way to restore a saved global state, so save-and-restore isn't an
  option.
- The sampler starts from `mx.random.key(seed)`. On each call it splits the
  key and applies `mlx_lm.sample_utils.apply_top_p` then `apply_top_k`, the
  same order `make_sampler` uses. It then returns
  `mx.random.categorical(logprobs / temperature, key=subkey)`.
- Logits processors come from `make_logits_processors(presence_penalty=0.5,
presence_context_size=64)`. These are deterministic and use no RNG.

**`_generate_expansion(query: str, model: str, seed: int) -> list[str]`**
does what `expand_query` does today, plus the system message, sampler, and
processors:

- load through `get_or_load`
- build the chat-template prompt from `[system, user]` messages
- run `generate(..., max_tokens=600, sampler=_keyed_sampler(seed),
logits_processors=...)`
- strip the `<think>` block and split into non-empty lines

`expand_query` becomes `return _generate_expansion(query, model,
_seed_for(query, model))`. The split exists so the replay harness can try
other seeds without a salt parameter in product code.

Everything else is unchanged:

- the user message
- `<think>` stripping
- line splitting
- `ExpansionModelError` wrapping
- `postprocess_expansion`'s fallback for output with no usable lines
- nothing writes to `llm_cache`

The MCP server already serializes every handler behind one lock
(`mcp/server.py`), so expansions never run concurrently, although the keyed
sampler wouldn't need that anyway.

**Result:** pyqmd stays run-to-run deterministic, on any machine, with or
without `cleanup`. It stops being greedy. Results for a given query could
change with an mlx or mlx-lm upgrade, because the kernels or sampling
utilities may change.

### 2. Replay harness

`scripts/replay_query.py` gains `--expand seeded[:<salt>]`:

- `seeded` calls the real `expand_query`, so it measures exactly what
  ships.
- `seeded:<salt>` calls `_generate_expansion` with the first 4 bytes of
  `sha256(f"{model}\n{query}\n{salt}")`, which gives an independent draw
  per salt with the same mechanism.
- The spike-only `sampled:<seed>` mode is removed. It seeded the global RNG
  once per query with the same seed for every query, and its sampler
  duplicates what `llm/expand.py` now provides.

Cache keys already include the mode string, so `seeded:3` and `seeded`
never collide. Cache keys don't include the prompt, though, so delete the
harness's `seeded*` expansion entries whenever the prompt changes.

### 3. Acceptance run (manual, once, during implementation)

Results go in the docs below.

1. **Replay check:** `--check-store --expand seeded` matches `Store.query`
   for all 30 queries.
2. **Multi-salt check:** salts 1–10, one pass each. Accept when all of
   these hold, with greedy's numbers from the spike:
   - mean Recall@10 ≥ 0.867 (greedy 0.833 + 1/30)
   - mean MRR ≥ 0.751 (greedy 0.784 − 1/30)
   - mean nDCG@10 ≥ 0.754 (greedy 0.787 − 1/30)

   If any check fails, stop and report back. Don't adjust constants, seeds,
   or thresholds to make it pass.

3. **Shipped seed:** `parity/test_quality.py::test_pyqmd_meets_qrels_mode_quality_bar`
   passes on all three metrics. Its numbers are recorded as they come out.
   Its exact score isn't a gate, and the seed derivation is never tuned
   against SciFact.
4. **Determinism:** two full `Store.query` passes over the 30 queries give
   identical rankings.
5. **Reported, not gated:**
   - mean and max expansion latency over the 30 queries, greedy vs. seeded
     (the budget rises from 400 to 600 tokens)
   - how often `postprocess_expansion` falls back or yields no parts,
     greedy vs. seeded, across the salts (the amendment's spike saw about
     6%)

### 4. Docs

- `parity/README.md`, "How the qrels-mode quality test's margin works":
  - pyqmd samples with a per-query seed and is deterministic, but no longer
    greedy.
  - The table's pyqmd column gets the new numbers, with a new date.
  - The "passes with only 0.007 to spare" paragraph is rewritten to
    describe the fix and the new headroom.
- `2026-09-24-expand-model-gguf-weights-design.md`: a dated note under
  "Decoding unchanged" saying this spec supersedes it. The original text
  stays.
- Roadmap: mark the Recall@10 backlog item done and link this spec and its
  plan.
- `COMMAND_STATUS.md`: add a sentence to the `query` row saying expansion
  uses Node's sampling settings and system prompt with a per-query seed, so
  results are deterministic, and that Node's output grammar isn't ported.
- Roadmap backlog: a new entry for constraining expansion output to
  `lex:`/`vec:`/`hyde:` lines, as Node's grammar does (added with this
  amendment).
- CLAUDE.md's command list and `SKILL.md`: no change, since no command or
  flag changes.

## Testing

Fast (no models, run by `just test-fast`):

- `_seed_for` returns a pinned value for a fixed `(query, model)`. It
  differs when only the query changes and when only the model changes.
- `_keyed_sampler`:
  - Two samplers with the same seed return the same token sequence over a
    fixed sequence of logits. Different seeds diverge.
  - The global RNG isn't touched: draw from `mx.random` after
    `mx.random.seed(0)` with and without running the sampler in between,
    and compare the values.
- `expand_query` wiring: with `mlx_lm.generate` monkeypatched and a fake
  model and tokenizer pre-loaded into `_expand_cache`, check that it passes
  `max_tokens=600`, a sampler, and logits processors, that the chat
  template gets `[system, user]` messages with `_SYSTEM_PROMPT` first, and
  that the same query yields the same seed on repeat calls.

Slow (real model, `just test-slow`):

- The same query expanded twice returns identical lines. Two different
  queries each return non-empty lines. This goes next to the existing
  `test_expand_query_returns_nonempty_lines` in `tests/test_integration.py`.

The acceptance run in §3 is manual and isn't part of any automated suite.
`parity/test_quality.py` already covers the shipped seed's quality going
forward.

## Out of scope

- **Node's GBNF grammar, or any output-format constraint.** mlx-lm has no
  GBNF support. With the system prompt, about 6% of expansions come out
  malformed and fall back to the default expansion (see "Amendment"). A
  logits processor that forces each line to start with `lex: `, `vec: `, or
  `hyde: ` would close that gap. It's a roadmap backlog item, kept separate
  so this fix ships on its own measured merits.
  _(2026-09-25: done. See `2026-09-25-expansion-output-grammar-design.md`.)_
- **Caching expansions in `llm_cache`.** Per-query seeds make it
  unnecessary, and caching would make `query` write to the index.
- **Aligning pyqmd's RRF/rerank blend with Node's.** The spike found
  pyqmd's blend differs and scores better on SciFact. That divergence is
  worth documenting or deciding on separately.
- **The empty original-query FTS list.** FTS5 ANDs every term, so
  sentence-length queries get 0 FTS hits in both implementations.
- **q94, q128, q132.** These are missed under every variant the spike
  tried.
