# Query expansion: constrain output to Node's `lex:`/`vec:`/`hyde:` grammar

**Date:** 2026-09-25
**Status:** Implemented (2026-09-25)
**Related:** `2026-09-10-python-mlx-rewrite-roadmap.md` (backlog entry
"Constrain query-expansion output to `lex:`/`vec:`/`hyde:` lines");
`2026-09-25-seeded-expansion-sampling-design.md` (its "Amendment" found the
gap and listed the grammar as out of scope); `scripts/replay_query.py` (the
replay harness the acceptance run uses).

## Problem

Node decodes query expansions under a GBNF grammar (`qmd/src/llm.ts`,
`expandQuery`):

```text
root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+
```

mlx-lm has no GBNF support, so pyqmd decodes unconstrained. With Node's
system prompt (added by the seeded-expansion fix), about 6% of pyqmd's
expansions (19 of 300 in that spike) come out as prose commenting on the
query ("The statement you provided is…"). `postprocess_expansion` then
falls back to its fixed `hyde`/`lex`/`vec` set, so those queries lose their
expansion. Node's grammar makes that output impossible.

This is a fidelity fix, not a known recall problem: SciFact's Recall@10 was
the same with and without the malformed draws.

### Findings that shape the design

- **Node has no `<think>` block.** With a grammar set, node-llama-cpp's
  `LlamaChat` skips its prefix-trigger handling, including Qwen's thought
  segment (`node-llama-cpp/dist/evaluator/LlamaChat/LlamaChat.js`: the
  `if (this.grammar != null)` branch returns before any segment
  detection). `QwenChatWrapper`'s default variation `"3"` doesn't open a
  thought segment at response start. So the grammar applies from the first
  generated token, and Node's first token must begin `lex`, `vec`, or
  `hyde`. pyqmd's model currently emits `<think>\n\n</think>\n\n` first,
  which `_THINK_BLOCK_RE` strips.
- **Tokenization.** In the Qwen3 tokenizer, `lex: foo\n` splits as `lex`
  `:` ` foo` `\n`, and `hyde` as `hy` `de`. The space after the colon
  usually joins the first content token. Only 8 tokens (`l`, `le`, `lex`,
  `v`, `ve`, `vec`, `h`, `hy`) are prefixes of a line start, but about
  2,160 tokens mix a newline with other characters (`.\n`, `}\n\n`). The
  constraint therefore has to walk each token's full text, not treat
  tokens as atoms.
- **Widths.** The model's logits are 151,936 wide; the tokenizer has
  151,669 tokens. The padding ids have no text and must be masked. EOS is
  `<|im_end|>` (151645) or `<|endoftext|>` (151643).
- **Added tokens.** The tokenizer has 26 added tokens. The 14 `<|…|>`
  control tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, the vision
  and box markers) are flagged special. The other 12 (`<think>`,
  `</think>`, `<tool_call>`, `<|fim_prefix|>`, …) aren't, and llama.cpp's
  grammar treats them as literal text. Node's captured expansions contain
  `</think>` inside `hyde:` lines, which confirms it.
- **Node's last line runs on.** The grammar allows EOS only right after a
  `\n`. The model's natural ending for its last `hyde:` line is
  `…best practices.<|im_end|>`, with no newline. With EOS blocked mid-line,
  it keeps writing until it happens to emit a newline or hits
  `max_tokens`, often repeating boilerplate and ending in junk
  (`…in.ách`). In the spike's cache of Node's own expansions (`node:1`–`5`,
  150 total), 21 (14%) have a line longer than 600 characters; the median
  longest line is 197 characters. pyqmd's unconstrained v1.0.4 expansions
  (608 in the same cache) have none. A throwaway pyqmd probe with the
  grammar applied reproduced it: 3 of 8 draws ran on, taking about 3.4s
  instead of 0.6s.

### Decision: match Node exactly

The constraint applies from the first generated token, so pyqmd drops the
`<think>` block as Node does. This conditions the model the same way
Node's is, and Node's expansion text already scored 0.884 through pyqmd's
retrieval (seeded-expansion spec, "Amendment"). The cost is that this
changes two things at once (the grammar and the missing think block). If
the acceptance numbers miss, a diagnostic spike separates the two (see
§3).

Matching Node also means porting its run-on last line. It costs latency
(roughly 14% of queries take about 3s longer to expand, about +0.4s on
average), but Node's text scored 0.884 through pyqmd's retrieval with its
run-ons included, and no measurement shows them hurting retrieval.
Allowing EOS mid-line after at least one complete line would stop the
run-on. That's a deliberate deviation, kept as a possible follow-up if the
latency matters in practice, measurable with the replay harness.

Alternatives considered and rejected:

- **A grammar library (xgrammar, llguidance) fed Node's GBNF verbatim.** A
  compiled dependency and an unverified mlx-lm integration, for a
  four-line grammar. Worth it only if more grammars were coming, and none
  are.
- **Reject and resample on a parse failure.** Not what Node does. It costs
  extra model calls on the failing queries and samples from a different
  distribution than a grammar would.

## Design

### 1. `llm/expand.py` and `llm/_expand_grammar.py`

`expand_query(query, model)` keeps its signature and return value. Nothing
outside `llm/` changes in the product code.

The automaton, the byte mapping, and the table build go in a new
pure-Python module, `llm/_expand_grammar.py`, with no MLX import, so CI's
Ubuntu leg (where `mlx` can't import) runs their tests. `llm/expand.py`
builds the tables at load time and owns the MLX-side processor.

**Automaton.** A hand-compiled byte-level DFA for Node's grammar, with a
comment quoting the GBNF and citing `qmd/src/llm.ts`. It has 13 states:

| state     | meaning                                | transitions                                |
| --------- | -------------------------------------- | ------------------------------------------ |
| `START0`  | line start, no complete line yet       | `l`→`L`, `v`→`V`, `h`→`H`                  |
| `START1`  | line start, at least one complete line | same as `START0`; EOS allowed here only    |
| `L`/`LE`  | read `l` / `le`                        | `e`→`LE`, `x`→`TYPE`                       |
| `V`/`VE`  | read `v` / `ve`                        | `e`→`VE`, `c`→`TYPE`                       |
| `H`…`HYD` | read `h` / `hy` / `hyd`                | `y`→`HY`, `d`→`HYD`, `e`→`TYPE`            |
| `TYPE`    | read a full type word                  | `:`→`COLON`                                |
| `COLON`   | read `:`                               | space→`SPACE`                              |
| `SPACE`   | read `: `, content needs one byte      | any byte but `\n`→`CONTENT`                |
| `CONTENT` | inside content                         | any byte but `\n`→`CONTENT`, `\n`→`START1` |

Every other (state, byte) pair is dead. It works on bytes rather than
characters. `\n` (0x0A) never occurs inside a multi-byte UTF-8 sequence,
so a token holding a partial UTF-8 character is handled correctly: its
bytes are content bytes like any other.

**Vocabulary tables.** Built once per loaded model and cached with the
model in `_expand_cache` (the entry becomes model, tokenizer, tables).

- Each token's raw bytes come from the tokenizer's byte-level BPE vocab
  through the inverse of the GPT-2 `bytes_to_unicode` table, not from
  `decode()`. `decode()` can clean up spaces and turns partial UTF-8 into
  U+FFFD.
- For each state and each token, walk the token's bytes and record the end
  state, or dead if any byte is dead. The result is, per state, a boolean
  mask over the full logits width (as an `mx.array`) and a next-state
  table.
- Added tokens flagged special (`<|im_start|>`, `<|im_end|>`, …) are dead
  everywhere, except the EOS ids (`tokenizer.eos_token_ids`), which are
  allowed in `START1` only.
- Added tokens not flagged special (`<think>`, `</think>`, …) are literal
  text, as in llama.cpp: their bytes are their content's UTF-8. They can
  appear inside content but never at a line start, so a leading think
  block stays impossible.
- A regular vocab token whose text isn't in the byte mapping raises, and
  `_load` reports it as an `ExpansionModelError`. It means a
  `PYQMD_EXPAND_MODEL` override uses a tokenizer that isn't byte-level BPE,
  and silently masking tokens would corrupt its output.
- Ids at or beyond the tokenizer's vocab size (the padding up to the
  logits width) are dead everywhere.
- The build walks about 2M (state, token) pairs, most of which die on the
  first byte. A throwaway probe on the real vocab took 0.13s. Its time is
  reported in the acceptance run, not gated.

**Processor.** `_grammar_processor(tables)` returns a fresh closure per
generation, passed to `generate` after the presence-penalty processor.
mlx-lm calls it as `processor(tokens, logits)`, where `tokens` is the
whole history, including the prompt.

- On its first call it records the history length and stays in `START0`.
- On each later call it advances the state by the newest token.
- It returns `mx.where(mask[state], logits, -inf)`.
- Presence penalties never un-mask a `-inf`, so the order relative to the
  penalty processor is for readability only.
- The grammar can't dead-end: `CONTENT` and both start states always have
  allowed tokens. A `max_tokens` cutoff mid-line leaves a partial last
  line, as it does in Node.

**Unchanged:** the keyed sampler, the seeding, the system and user
messages, `max_tokens=600`, `ExpansionModelError` wrapping,
and `postprocess_expansion` (its fallback still applies when no line
mentions the query's terms, a filter Node also applies and the grammar
can't prevent).

**Removed:** `_generate_expansion` no longer strips `<think>` blocks. Node
doesn't strip them, and the grammar makes a leading block impossible, so
stripping would only change text Node keeps (a literal `<think>…</think>`
inside a content line). `_THINK_BLOCK_RE` stays in `llm/expand.py` for the
replay harness's unconstrained `greedy` mode.

**Changed: line splitting.** Output splits on `\n` only, dropping empty
lines, as Node's `result.trim().split("\n")` does. `str.splitlines()` also
splits on `\r`, `\u2028`, and other separators, which the grammar allows
inside content, so it would break one grammar line into several.

**Result:** every expansion is one or more `type: content` lines, the last
possibly cut off by `max_tokens`. As in Node, the last line sometimes runs
on (see "Findings"). Results for a given query change from v1.0.4's, since
both the constraint and the missing think block change what the model sees
and may sample. They stay deterministic per (query, model).

### 2. Replay harness

`scripts/replay_query.py`:

- The `seeded` and `seeded:<salt>` modes call `expand_query` and
  `_generate_expansion`, so they exercise the constraint with no change.
- Its summary line gains `malformed=<n>`: the number of queries whose
  expansion has zero lines that `parse_expanded_lines` accepts. That's the
  format failure the grammar removes. The existing `fallback=` count stays
  as is, and still counts the query-term filter's fallbacks too.
- It also gains `runon=<n>`: the number of queries whose expansion has a
  line longer than 600 characters, the measure behind Node's 21 of 150.
- Cache keys don't include the prompt or the constraint, so clear the
  cached `seeded*` expansion entries before the acceptance run.

### 3. Acceptance run (manual, once, during implementation)

Results go in the docs below. The
reference numbers are the seeded-expansion fix's recorded salt means:
Recall@10 0.884, MRR 0.793, nDCG@10 0.807, 19 of 300 fallbacks.

1. **Replay check:** `--check-store --expand seeded` matches `Store.query`
   for all 30 queries.
2. **Format:** over salts 1–10 (280 expansions: 2 of the 30 queries take
   the strong-signal path and skip expansion), `malformed` is 0 in every
   run. `fallback` is reported next to it, not gated.
3. **Quality:** the salt means hold within 1/30 of the reference:
   - mean Recall@10 ≥ 0.851
   - mean MRR ≥ 0.760
   - mean nDCG@10 ≥ 0.774

   If any check in 2 or 3 fails, stop and report back. Don't adjust
   constants, seeds, or thresholds to make it pass. The next step is then a
   diagnostic spike, not a fix: replay the same salts with a throwaway
   variant that allows an empty `<think>\n\n</think>\n\n` before the lines,
   to attribute the change to the grammar or to the missing think block.

4. **Shipped seed:** `parity/test_quality.py::test_pyqmd_meets_qrels_mode_quality_bar`
   passes on all three metrics, and its numbers are recorded as they come
   out. Two full `Store.query` passes over the 30 queries give identical
   rankings.
5. **Reported, not gated:**
   - the table-build time
   - the run-on count over the 280 salt expansions, next to Node's 21 of
     150 (14%)
   - mean and max expansion latency over the salt runs, against the
     reference (0.48s / 0.69s). Expect the mean to rise with the run-ons.

### 4. Docs

- `COMMAND_STATUS.md`, `query` row: Node's output grammar is now ported,
  replacing the sentence saying it isn't.
- `parity/README.md`, the qrels-mode margin table: the shipped seed's new
  numbers with a new date, if they change.
- `2026-09-25-seeded-expansion-sampling-design.md`: a dated note under
  "Out of scope" pointing here. The original text stays.
- Roadmap: mark the backlog entry done and link this spec and its plan.
- CLAUDE.md's command list and `SKILL.md`: no change, since no command or
  flag changes.

## Testing

Fast (no models, run by `just test-fast`):

- **Automaton on strings.** Accepts `lex: a\n`, `vec: x y\nhyde: z\n`,
  and `lex: a` (a valid prefix that doesn't allow EOS). Rejects `lex:a`,
  `lex: \n`, `LEX: a`, `\n`, `foo`, the second `\n` of `lex: a\n\n`, and
  `<think>`.
- **Table build on a fake vocab** of about 20 byte-string tokens, with
  special ids and a logits width wider than the vocab:
  - each state's mask matches the automaton
  - a token spanning states (` foo\nlex`) ends in the right state
  - EOS is allowed in `START1` only
  - special added tokens and padding ids are always masked
  - a non-special added token (`</think>`) is allowed in `CONTENT` and dead
    at a line start
- **Byte mapping.** The inverse `bytes_to_unicode` table round-trips all
  256 bytes. Reading token bytes from a fake tokenizer gives the mapped
  bytes for regular tokens, UTF-8 content for non-special added tokens,
  `None` for special ones, and raises on a token outside the mapping.
- **Processor.** Driven by a scripted token sequence: the first call
  records the history length, the state advances one token per call,
  disallowed ids come out `-inf`, and allowed ids are unchanged.
- **Sampler with masks.** `_keyed_sampler` never picks a masked id across
  many draws with the real `top_p`/`top_k` settings, so both filters cope
  with `-inf`.
- **`expand_query` wiring.** The existing wiring test in
  `tests/test_expand_model.py` is extended: the logits processors now
  include the grammar processor, after the presence penalty, and output
  text is no longer think-stripped and splits on `\n` only. A table-build
  failure at load time is reported as an `ExpansionModelError`.

Slow (real model, `just test-slow`):

- Expanding several queries gives lines that all match
  `^(lex|vec|hyde): [^\n]+$`, so no leading `<think>` block. Two runs of
  the same query give identical lines.
- On the real tokenizer, the byte mapping agrees with
  `tokenizer.decode([id])` for every non-special token whose bytes are
  valid UTF-8.

The acceptance run in §3 is manual and isn't part of any automated suite.

## Out of scope

- **A general GBNF engine or grammar library.** One fixed grammar doesn't
  justify it.
- **The query-term filter's fallback.** `postprocess_expansion` drops
  lines that mention none of the query's terms, as Node does. That's
  unchanged.
- **Caching the tables on disk.** They're rebuilt once per process, when
  the model loads.
- **`pyqmd bench --samples N`.** A separate roadmap backlog item.
- **Stopping the run-on last line.** Allowing EOS mid-line would diverge
  from Node (see "Decision"). A possible follow-up if the latency matters.
