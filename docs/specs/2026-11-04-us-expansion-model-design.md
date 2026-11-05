# Query expansion on a US-made open-weight base — design

**Date:** 2026-11-04
**Status:** Implemented as an opt-in model (2026-11-04): `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6`,
a fine-tune of IBM Granite 4.0 1B, selected with `PYQMD_EXPAND_MODEL`. The default expansion model is unchanged
(the Qwen3-1.7B-based one). See "Granite 4.0 1B local experiment" at the end for what was built and measured; the
earlier sections are the design, written before the experiment.
**Related:** the opt-in Granite reranker (the reranker half of the same effort; this spec doesn't redo it); `2026-09-24-expand-model-gguf-weights-design.md` (where the
current weights come from); `2026-09-25-expand-model-requantize-design.md` (the mixed 4/6-bit
recipe and the 30-salt rule); `2026-09-25-expansion-output-grammar-design.md` (the byte-level
grammar this spec has to keep working); `2026-09-25-seeded-expansion-sampling-design.md`
(seeded sampling, the system prompt); `2026-09-24-expand-model-hf-default-design.md`
(`PYQMD_EXPAND_MODEL`).

## Why

Goal: a pyqmd model set whose weights come from US organisations.

| Role            | Default today                           | Maker        | US-made?                                                 |
| --------------- | --------------------------------------- | ------------ | -------------------------------------------------------- |
| Embedding       | `embeddinggemma-300m-8bit`              | Google       | yes                                                      |
| Rerank          | `Qwen3-Reranker-0.6B-mxfp8`             | Alibaba      | no — opt-in Granite cross-encoder exists (reranker spec) |
| Query expansion | fine-tune of Qwen3-1.7B (mixed 4/6-bit) | Alibaba base | **no**                                                   |

Expansion is the last non-US piece and is independent of the reranker. Unlike the
reranker, there is no ready-made US-made expansion model: the current one is a _fine-tune_
(SFT of Qwen3-1.7B on the `hyde:`/`lex:`/`vec:` format, in the sibling Node `qmd` repo's
`finetune/`). A US-made replacement therefore means **re-running that SFT on a different
base**, then converting and hosting the result. The work is mostly integration and
evaluation, not research.

"US-made" here means the _pretraining_ of the base was done by a US organisation. Our own
fine-tune on top is ours. A US-branded fine-tune of a foreign base would not count, and a
foreign fine-tune of a US base (none considered) would be fine.

## Scope

In: choice of base; the changes `llm/expand.py` and `_expand_grammar.py` need for it; the
training pipeline for the new base; conversion, hosting and roll-out; evaluation plan and
gates; risks and an effort estimate.

Out: any training or hosting (each needs the owner's go-ahead, below); changing the
embedding or rerank models; Node `qmd` itself; a default flip (a separate, tiny change after
the gates pass); expansion quality work unrelated to the base swap (prompt tuning, sampler
sweeps).

## 1. Candidates (state as of 2026-11-04)

Checked against Hugging Face model cards, `config.json` and the Hub API today. "Run on
pyqmd today" means mlx-lm 0.31.3 (the version in the project venv) has the architecture.

| Base                                                        | Maker (US?)                                                                     | License                                                                                                       | Size                                                                                                           | MLX / mlx-lm                                                                                                                                                        | Verdict                                                                                                                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Granite 4.0 1B** (`ibm-granite/granite-4.0-1b`, Oct 2025) | IBM — yes                                                                       | Apache 2.0                                                                                                    | 1.6B dense (config `granitemoehybrid`, 40 layers, all attention, 100,352 vocab, tied embeddings, 128K context) | arch `granitemoehybrid` is in mlx-lm 0.31.3; `mlx-community/granite-4.0-1b-4bit` exists                                                                             | **Recommended**                                                                                                                                |
| **Gemma 4 E2B** (`google/gemma-4-E2B-it`, Mar 2026)         | Google — yes                                                                    | Apache 2.0 (since this generation; Gemma 3 and earlier were the Gemma terms)                                  | 2.3B effective / 5.1B with per-layer embeddings; 35 layers, 262,144 vocab, multimodal checkpoint               | `gemma4`/`gemma4_text` in mlx-lm 0.31.3 (its `sanitize` drops the vision/audio towers); `mlx-community/gemma-4-e2b-it-4bit` exists but was converted with `mlx_vlm` | **Fallback / challenger**                                                                                                                      |
| Gemma 4 E4B (`google/gemma-4-E4B-it`)                       | Google — yes                                                                    | Apache 2.0                                                                                                    | 4.5B effective / 8B total                                                                                      | `mlx-community/gemma-4-e4b-it-4bit`, 5.15 GB on disk                                                                                                                | Too big for a per-query model; only if E2B is too weak                                                                                         |
| Granite 4.0 350M (`ibm-granite/granite-4.0-350m`)           | IBM — yes                                                                       | Apache 2.0                                                                                                    | 350M dense, 28 layers, 32K context                                                                             | `mlx-community/granite-4.0-350m-4bit`                                                                                                                               | Optional cheap third arm; likely too weak for `hyde:` text                                                                                     |
| Granite 4.2 3B (`ibm-granite/granite-4.2-3b`, Aug 2026)     | IBM — yes                                                                       | Apache 2.0                                                                                                    | 3B dense (`GraniteForCausalLM`, 40 layers)                                                                     | arch `granite` supported; no `mlx-community` 4.2 build listed (4.1 3B has them)                                                                                     | Second fallback; ~2× the current model. Thinking is on by default (the template ends the prompt with `<think>`), needs `enable_thinking=False` |
| SmolLM3 3B (`HuggingFaceTB/SmolLM3-3B`)                     | Hugging Face — arguable (US-incorporated, a large part of the team is in Paris) | Apache 2.0                                                                                                    | 3B                                                                                                             | `smollm3` arch, `mlx-community/SmolLM3-3B-4bit`                                                                                                                     | Passes the tokenizer probe; ownership is arguable and its template injects a long default system prompt. Not pursued                           |
| Llama 3.2 1B/3B                                             | Meta — yes                                                                      | Llama 3.2 Community License (not OSI): "Built with Llama" notice, derived model names must start with "Llama" | 1.2B / 3.2B                                                                                                    | yes                                                                                                                                                                 | Passes the tokenizer probe; the license shapes how the model can be named and hosted. Sept 2024, oldest option. Rejected                       |
| OLMo 3 (Ai2)                                                | Ai2 — yes                                                                       | Apache 2.0                                                                                                    | smallest instruct model is 7B (Nov 2025); Olmo-Hybrid 7B (Jan–Feb 2026)                                        | `olmo3` arch                                                                                                                                                        | Too big. Searched for a 1B/3B: none                                                                                                            |
| Liquid LFM2.5 1.2B / LFM3 1.7B                              | Liquid AI — yes                                                                 | "LFM Open License" (HF tag `other`), not Apache                                                               | 1.2B / 1.7B (LFM3 released 2026-10-27)                                                                         | `lfm2` arch; LFM3 MLX builds from Liquid                                                                                                                            | The Node `finetune/` tree already has an LFM2 experiment config. License terms not read in full; not Apache. Not pursued                       |
| Phi-4-mini                                                  | Microsoft — yes                                                                 | MIT                                                                                                           | 3.8B                                                                                                           | not checked                                                                                                                                                         | Feb 2025, no newer Phi LLM on the Hub (Phi-tiny-MoE is the newest, Jun 2025). Not evaluated further                                            |
| NVIDIA Nemotron 3.5 Bolt-E4B                                | NVIDIA, but a **post-train of Gemma 4 E4B**                                     | HF tag `other` (secondary sources disagree: OpenMDW-1.1 vs Gemma's Apache)                                    | ~4B effective                                                                                                  | not checked                                                                                                                                                         | No reason to prefer it to the Gemma 4 E4B it derives from. Rejected                                                                            |
| NVIDIA Nemotron 3.5 Embed 1B/8B, Rerank 8B                  | NVIDIA, but built on Mistral's Ministral-3                                      | `other`                                                                                                       | —                                                                                                              | —                                                                                                                                                                   | Not US lineage; irrelevant to expansion                                                                                                        |

Notes:

- Granite had no new small model after 4.0 Nano (Oct 2025): 4.1 (Apr 2026) and 4.2
  (Aug 2026) are 3B/8B/30B only. So "Granite 4.0 1B" is the latest _small_ Granite, not
  an old pick.
- Parameter counts on Hub pages disagree with the tables inside them (Granite 4.0 1B shows
  "1.6B" in IBM's table and 2B in the safetensors metadata). Treat sizes as ±20%.
- The `mlx-community` Gemma 4 builds are `mlx_vlm` conversions of the multimodal
  checkpoint. A text-only expansion model should be converted by us from our own merged
  fine-tune (§4), not taken from those.

### Sizes (Hub file sizes read 2026-11-04; text-only Gemma and memory are estimates)

| Model                              | Parameters                    | bf16 on disk        | 4-bit MLX on disk                                                                                                                                            | Expected resident memory while expanding                                                                                                           |
| ---------------------------------- | ----------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Current (Qwen3-1.7B mixed 4/6-bit) | 1.7B                          | ~3.4 GB             | ~1.0 GB                                                                                                                                                      | ~1.2 GB                                                                                                                                            |
| Granite 4.0 1B                     | 1.63B                         | 3.26 GB (measured)  | 0.92 GB (measured, `mlx-community/granite-4.0-1b-4bit`)                                                                                                      | ~1.1–1.3 GB                                                                                                                                        |
| Gemma 4 E2B                        | 5.12B stored (2.3B effective) | 10.25 GB (measured) | 3.55 GB (measured, `mlx-community/gemma-4-e2b-it-4bit`, includes ~0.47B bf16 vision/audio parameters); ~2.6 GB text-only after our own conversion (estimate) | ~2.8–3.2 GB, because MLX maps the whole file; per-layer embeddings are only looked up per token, so speed tracks the 2.3B effective size, not 5.1B |

Memory is weights plus a small KV cache (Granite: about 80 MB per 1,000 tokens at fp16;
expansion prompts are under 100 tokens and output is capped at 600) plus working buffers. It is
on top of the embedding and rerank models that the same `query` process also loads. These
resident figures are my arithmetic, not measurements; Phase 4 records the real numbers.

### Recommendation

**Granite 4.0 1B first, Gemma 4 E2B as the challenger/fallback**, decided by a small
bake-off (§3, Phase 3) rather than on paper — each SFT run costs a few dollars, while
integration work costs days.

Why Granite first:

1. It is the closest drop-in. Its tokenizer passes the existing grammar's byte-level
   mapping _unchanged_ (§2: all 100,352 tokens), its template is plain, and it has no
   thinking mode to suppress in 4.0 (no empty-think-block handling).
2. Same size class as the current model (1.6B vs 1.7B), so latency and download size should
   stay near today's (~0.9 s mean, ~1.0 GB). These are expectations to measure, not facts.
3. One vendor and one license (Apache 2.0, IBM) for both reranker and expansion; the reranker
   spec already commits to IBM attribution.

Why Gemma 4 E2B is the challenger rather than the pick: it is likely the stronger model
(5.1B stored parameters, more world knowledge for `hyde:` text, and the same maker as the
embedding model), but it costs a grammar change (§2.1), a heavier download (my estimate:
~2.6 GB at 4-bit text-only, about 2.6× today's; see "Sizes" below), a newer and
less-trodden training stack (per-layer embeddings, multimodal checkpoint, `transformers` ≥ 5.5
per the generation config), and a template that differs more. If Granite 4.0 1B misses the
quality gate (§5), Gemma 4 E2B is the next step with no further design work needed; Granite
4.2 3B is the last resort.

If the owner wants to skip the bake-off and pick on paper: pick Granite 4.0 1B.

## 2. Compatibility with the current decode path

Everything below was probed on 2026-11-04 by loading each model's tokenizer files (not
weights) and running the real `vocab_token_bytes` from `_expand_grammar.py` plus
`apply_chat_template` (throwaway scripts; recreate from this
description).

### 2.1 Tokenizer and the byte-level grammar

`vocab_token_bytes` maps token text to raw bytes through GPT-2's `bytes_to_unicode` table and
**raises** on any regular token outside it (`tests/test_expand_model.py`:
"isn't byte-level BPE").

| Tokenizer                 | Result            | Detail                                                                                                                                                                                                                            |
| ------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-1.7B (today)        | OK                | 151,669 tokens, 0.08 s; logits are 151,936 wide, so 267 padding ids are masked                                                                                                                                                    |
| Granite 4.0 350M / 4.2 3B | **OK, unchanged** | 100,352 tokens, 0.05 s; logits width = vocab, no padding                                                                                                                                                                          |
| SmolLM3 3B, Llama 3.2 1B  | OK                | 128,256 tokens                                                                                                                                                                                                                    |
| **Gemma 4 E2B**           | **fails**         | `ValueError: token 105955 ('▁bä') isn't byte-level BPE`. SentencePiece-style BPE: 261,864 tokens use `▁` for a space (137,542 contain it), 256 byte-fallback tokens `<0x00>`…`<0xFF>`, 24 added tokens all special, 262,144 total |

So the claim in the brief holds: **Gemma needs a second decoding path in `vocab_token_bytes`**;
the Qwen/Granite/Llama path needs none. The extra path is small and I prototyped it on the real
Gemma vocab: `<0xNN>` → that one byte, otherwise `text.replace("▁", " ")` as UTF-8. Over all
262,120 non-special tokens, the result equals `tokenizer.decode([id])` for every token whose
bytes are valid UTF-8 (0 mismatches; the rest are partial UTF-8 byte-fallback pieces, which the
automaton handles on bytes). Gemma tokenizes `lex: foo\n` as `lex` `:` `▁foo` `\n` — the same
shape as Qwen's, so the automaton's state logic needs no change; the table build over 262K
tokens should take about 0.25 s (estimate from the 0.13 s measured on Qwen3).

Design for the second path: `vocab_token_bytes` detects the alphabet from the vocab
(byte-fallback tokens present and `▁` common → SentencePiece) instead of keying on a model
name, tries the GPT-2 path first, and keeps raising `ValueError` for anything that matches
neither. The existing "isn't byte-level BPE" error stays for genuinely unknown tokenizers; its
message and test are updated to name both supported alphabets. This change is only needed if
Gemma enters the bake-off; Granite does not need it, so it is split into its own task (§3,
Phase 1b) that can be skipped.

### 2.2 Chat template and the prompt

Rendered with `apply_chat_template(..., add_generation_prompt=True)` for
`[system, user("/no_think Expand this search query: foo")]`:

| Model       | Prompt text (abridged)                                                                                                                                                    | Notes                                                                                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen3       | `<\|im_start\|>system\n…<\|im_end\|>\n<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n`                                                                       | `/no_think` is a Qwen3 control word                                                                                                                          |
| Granite 4.0 | `<\|start_of_role\|>system<\|end_of_role\|>…<\|end_of_text\|>\n<\|start_of_role\|>user<\|end_of_role\|>…<\|end_of_text\|>\n<\|start_of_role\|>assistant<\|end_of_role\|>` | **With no system message the template inserts a default one** ("You are a helpful assistant. Please ensure responses are professional, accurate, and safe.") |
| Granite 4.2 | `<\|im_start\|>…` ChatML, ending `<\|im_start\|>assistant\n<think>\n`                                                                                                     | thinking on by default; `enable_thinking=False` gives `<think></think>`                                                                                      |
| Gemma 4     | `<bos><\|turn>system\n…<turn\|>\n<\|turn>user\n…<turn\|>\n<\|turn>model\n`                                                                                                | assistant role is `model`; no empty thought block by default; system turn optional                                                                           |
| Llama 3.2   | adds "Cutting Knowledge Date / Today Date" to every system message                                                                                                        |                                                                                                                                                              |
| SmolLM3     | adds a long metadata/`/think` system block                                                                                                                                |                                                                                                                                                              |

Consequences for `expand.py`:

1. **`/no_think` must be conditional.** `_generate_expansion` hard-codes
   `f"/no_think Expand this search query: {query}"`. It is meaningless noise for Granite and
   Gemma, and training would have to include it too. Decision: the user message is
   `Expand this search query: {query}` for every base except Qwen3 (the current model, and any
   Qwen3 override via `PYQMD_EXPAND_MODEL`), which keeps `/no_think`. The switch is the loaded
   model's `config.json` `model_type` (the same "config, not model name" rule as the
   reranker spec's backend choice), through one small function so the table is testable.
2. **System prompt.** pyqmd always sends Node's default system prompt (`_SYSTEM_PROMPT`),
   because Node gets it implicitly and it measurably helped _Node's_ model (seeded-sampling
   spec, "Amendment"). That finding is about the Qwen3 model's training/inference mismatch.
   For a new base we control both sides, so **train and serve with the same prompt**.
   Decision (my call, open for the owner): the new model is trained and served with _no
   pyqmd-supplied system message_; whatever default system text the base's template adds
   (Granite) is added identically in both places because both use `apply_chat_template`.
   Qwen3 keeps `_SYSTEM_PROMPT`. The system-message choice is a per-model setting beside the
   `/no_think` one.
3. **Train/serve prompt drift is the main correctness risk**, because the two codebases
   (Node `finetune/` and pyqmd) are separate. Mitigation: a golden-string test in pyqmd that
   renders the exact prompt for a fixed query for each supported `model_type` (the strings
   come from the real tokenizers' templates, committed as text, not the tokenizers), and the
   training pipeline's data prep asserts it produces a prefix-identical string for the same
   query.
4. **Thinking templates.** Granite 4.2 and any future thinking base needs
   `enable_thinking=False` passed to `apply_chat_template`; Gemma 4 defaults to off. The
   grammar forbids a leading `<think>` anyway (`<think>` is a non-special added token that is
   dead at a line start), but Granite 4.2's template _ends the prompt_ with `<think>\n`, so the
   generated text would start inside a think block. A flag per `model_type` covers it. Not
   needed for Granite 4.0 1B or Gemma 4.

### 2.3 EOS handling

- Granite 4.0: EOS is `<|end_of_text|>` (id 100257); config and generation config agree.
  Granite 4.2 uses `<|im_end|>`, which is _also_ id 100257 in its tokenizer.
- Gemma 4: config lists `[1, 106]` (`<eos>`, `<turn|>`); generation config adds 50
  (`<|tool_response>`). The model ends a turn with `<turn|>` (106). mlx-lm builds
  `tokenizer.eos_token_ids` from the config list, so the grammar's "EOS allowed only at
  line start" set will be right _provided the converted `config.json` keeps the list_. The
  conversion script asserts that the turn-end id is in the converted model's EOS set.
- `_grammar_processor` / `build_tables` need no change: they take whatever
  `tokenizer.eos_token_ids` says.

### 2.4 Logits width and masking

`_state_mask` already masks ids beyond the tokenizer's vocab. Qwen3's logits are 267 wider
than its tokenizer. Granite and Gemma have logits width equal to vocab size, so the
padding branch is a no-op for them. Gemma also applies `final_logit_softcapping` (30.0)
inside the model, before the processors, so masking is unaffected.

### 2.5 The run-on last line (a training fix, not a decoder fix)

The grammar allows EOS only right after `\n`. The training targets end with _no_ trailing
newline (`output_items_to_text` joins lines with `"\n"`), so a fine-tuned model ends
`…last line<EOS>`, and under the grammar that is forbidden; the model then writes on until a
newline or `max_tokens` (the 26% → 7% run-on problem the re-quantization spec chased, and
Node's own 14%).

For a new base we are training anyway, so **make the training target end with `\n`** and
the model learns to emit `\n<EOS>`, which the grammar allows. Check at data-prep time that
the rendered training text really ends `\n<eos-token>`: some chat templates strip message
content (Gemma's and Granite's templates may `trim`), in which case the prep script builds
the final text by concatenation instead of through the template. If this works, run-ons
should drop to near zero and the 6-bit-head part of the quantization recipe stops being
necessary for run-on control (still re-measure). Fallback if it doesn't: allow EOS mid-line
after at least one complete line (the grammar spec's deferred "possible follow-up"). That
deviates from Node's grammar, but there's no Node parity to keep for a different model's
output.

### 2.6 Everything else in the path

Unchanged and base-independent: seeded sampler (`_keyed_sampler`), temperature 0.7, top-k 20,
top-p 0.8, presence penalty 0.5, `max_tokens=600`, `postprocess_expansion`, the replay harness.
The sampling constants are Node's, chosen for Qwen3; Gemma's own recommended sampling is
temperature 1.0, top-p 0.95, top-k 64. Not re-tuned here (out of scope); if the new model
underperforms a sweep is a lever, to be listed in the evaluation results rather than
quietly applied.

## 3. Training

### 3.1 The existing pipeline (sibling `qmd` checkout, `finetune/`)

- **Data:** 12 JSONL files in `data/` (3,579 lines; 3,118 unique queries after the
  case-insensitive dedup `prepare_data.py` does → 2,806 train / 312 val). The README's
  "~2,290 examples" is stale. Strict schema (`query`, `output: [[type, text], …]`, optional
  `intent`). Hyde-first ordering is applied at render time.
- **Prep:** `dataset/prepare_data.py` renders each example with the **Qwen3** chat template
  (`QMD_BASE_MODEL` env var picks the tokenizer, default `Qwen/Qwen3-1.7B`), prompt
  `/no_think Expand this search query: <q>` (plus `\nQuery intent: …` when `intent` is set),
  and deletes the empty `<think>\n\n</think>\n\n` Qwen3 inserts.
- **Train:** `train.py sft --config configs/sft.yaml` (or the self-contained
  `jobs/sft.py` for Hugging Face Jobs): TRL `SFTTrainer`, LoRA r=16 / α=32 on all of
  q/k/v/o/gate/up/down, 5 epochs, batch 4 × 4 accumulation, lr 2e-4 cosine, max length 512,
  bf16; `train.py` also saves `embed_tokens`/`lm_head` in full ("critical for special
  tokens"). Reported: A10G (24 GB), ~45 min, ~$1.50; final eval loss 0.304, token accuracy
  93.8%. Loss is on the whole sequence (no completion-only masking).
- **Eval:** `eval.py` generates greedily with `transformers` from `evals/queries.txt`
  (51 queries; the README says 31) and scores with `reward.py`'s rubric. Reported baseline
  for the Qwen3-1.7B SFT model: average 92.0%, 30/30 "excellent".
- The pipeline's GGUF export and `convert_gguf.py` are irrelevant here: they exist because
  Node runs llama.cpp.
- There is also a LFM2 experiment config (`experiments/lfm2/`) and an unfinished Unsloth
  script for Qwen3.5 (its log shows it failed on a Mac with no GPU). Neither is a template
  for this work, but the LFM2 one shows the repo already expects non-Qwen bases.

### 3.2 What changes for a new base

| Piece                         | Change                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat template / prompt        | Prompt without `/no_think`; data prep uses the new base's template (`QMD_BASE_MODEL`) and drops the Qwen-specific empty-think strip; **append `\n` to every target** (§2.5); no pyqmd system message (§2.2). Prefix-identical to pyqmd's prompt, asserted.                                                                                                                                                                               |
| LoRA targets                  | q/k/v/o are the same names in Granite and Gemma. Granite 4.0's `granitemoehybrid` code in `transformers` may name the dense-MLP projections differently from `gate_proj`/`up_proj`/`down_proj`, and Gemma 4's multimodal checkpoint has vision/audio modules that also contain `q_proj` etc. **Unverified**: print the module names on a meta-device model first and restrict targets to the language model (regex), as a Phase 2 check. |
| `modules_to_save`             | `embed_tokens`/`lm_head` full-saves are enormous on Gemma (262K × 1536, plus per-layer embeddings that must NOT be trained or duplicated). For Gemma: do not save the embeddings; the format is learned in the transformer layers, and no new tokens are added. For Granite (100K × 2048, tied) it is fine either way; test without.                                                                                                     |
| Double BOS                    | Gemma's template emits `<bos>` in the text and `SFTTrainer` may add another when tokenizing a `text` field. Use the `messages`-style field or set `add_special_tokens=False`; assert on the first training sample's token ids.                                                                                                                                                                                                           |
| `reward.py`                   | `CHAT_TEMPLATE_TOKENS` and `clean_model_output` only know Qwen's `<\|im_start\|>`/`<\|im_end\|>`/`<\|endoftext\|>`. Add Granite's (`<\|start_of_role\|>`, `<\|end_of_role\|>`, `<\|end_of_text\|>`) and Gemma's (`<\|turn>`, `<turn\|>`, `<bos>`, `<eos>`) so leakage is a hard failure for the new models too. Otherwise their leakage would score as valid prose.                                                                      |
| `eval.py`                     | Hard-codes `/no_think`, `config.tie_word_embeddings = False`, and greedy decoding. Make the prompt come from the shared builder and the model class from the base; keep greedy as the _format_ check.                                                                                                                                                                                                                                    |
| `transformers` / TRL versions | Gemma 4 needs a recent `transformers` (the generation config in the Hub repo is stamped 5.5.0.dev0); Granite needs `GraniteMoeHybrid` support. Pin and test in the job image; this is the likeliest source of Phase 2 delay.                                                                                                                                                                                                             |

### 3.3 Data

Reuse Node's 3,118 examples unchanged (same targets, plus the trailing newline). No new data
in the first round. The README labels the data MIT; where the examples came from is not
documented in the tree I read (it looks synthetic), so **provenance is an open question
(§7)**. If the first model misses the gate on `hyde:` quality, more data is the first lever:
regenerate targets for the same queries with a stronger US-made model, which would need its own
license check.

### 3.4 Where the training code lives

Recommendation: a new `finetune/` directory in the pyqmd repo (scripts and configs, not
shipped in the wheel), containing the adapted `prepare_data`, `train`, `reward`, `eval`,
the data files, and a README that credits Node qmd (MIT). Reasons: the Hub model card
needs a reproducible recipe to point to, as the current model's card points at
`scripts/convert_expand_gguf.py`; the shared prompt builder test (§2.2) is far easier to keep
honest in one repo; and the sibling checkout is the upstream's tree, not ours to patch.
Alternative: keep the training scripts in a private repo. This is the owner's call.

### 3.5 Compute and cost (estimates, not measurements)

The reported baseline is ~45 min on one 24 GB A10G for Qwen3-1.7B. HF Jobs prices today
(from the Hub pricing page): A10G-large $1.50/h, L40S $1.80/h, A100-large (80 GB) $2.50/h,
H200 $5.00/h, billed by the minute.

| Run                                                                          | Hardware                            | Estimate          |
| ---------------------------------------------------------------------------- | ----------------------------------- | ----------------- |
| Granite 4.0 1B, SFT                                                          | A10G-large                          | ~45–60 min, ~$1–2 |
| Gemma 4 E2B, SFT (5.1B stored parameters; ~10 GB in bf16 before activations) | A100-large preferred; A10G is tight | ~1–2 h, ~$3–5     |
| Granite 4.0 350M                                                             | A10G-large                          | ~20–30 min, <$1   |
| Evaluation / reruns                                                          | —                                   | ~$5–10 of slack   |

Budget for the whole bake-off with two retries each: well under $30. The owner approves
spending before any job starts, and a `--timeout` is set on every job (the default is 30
minutes). The M4 Max in this checkout (128 GB unified memory) could also run a LoRA locally
with `mlx_lm.lora`; I have not checked that its tuner supports `granitemoehybrid` or
`gemma4`, and its recipe differs from TRL's, so cloud TRL stays the primary path for fidelity
with Node's recipe. Local MLX LoRA is a possible cheap smoke test.

### 3.6 Hosting

Hub repos under `after2400/`, like the current model:

- `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6` — the shipped MLX model.
- `after2400/qmd-query-expansion-granite-1b` — merged bf16 weights (so the conversion is
  reproducible and others can re-quantize), with the SFT adapter.

Names for Gemma follow the same pattern. Apache 2.0 has no naming requirement (unlike
Llama's). Model cards: `license: apache-2.0`, `base_model: ibm-granite/granite-4.0-1b`,
Apache attribution to IBM, Node qmd's MIT-licensed recipe and data credited, the exact
training command, revisions and library versions, and the eval table. **Uploading is
outward-facing and needs the owner's explicit go-ahead**, repo by repo.

## 4. Conversion and shipping

### 4.1 Conversion

The existing `scripts/convert_expand_gguf.py` exists only because Node's checkpoint was a
GGUF with no usable safetensors. The new model is trained with PEFT, so the merged result
is ordinary HF safetensors and **`mlx_lm.convert` converts it directly**. A new
`scripts/convert_expand_model.py`:

1. Downloads the merged model at a pinned revision.
2. Calls `mlx_lm.convert` with `quant_predicate` from the existing `quant_bits` (the
   `mixed_4_6` layer rule plus a 6-bit tied embedding), group size 64.
3. Asserts the EOS set contains the model's turn-end id (§2.3), that the chat template and
   tokenizer files were copied, and that `config.json` `model_type` is one `expand.py`'s prompt
   table knows.
4. Runs the same 3-query smoke test as today (each output has a `lex:` and a `vec:` line).
5. Renders the model card.

`quant_bits`, the model-card renderer and the smoke test move out of
`convert_expand_gguf.py` into a shared module so the two scripts don't diverge (the GGUF
script stays for the current model's reproducibility). Granite's quantization specifics:
tied embedding (6-bit, as for Qwen), plus its μP-style scalars (embedding/residual/logits
multipliers) which `mlx-lm`'s `granitemoehybrid` model applies itself; the 6-bit-head
recipe was validated on Qwen only, so **re-check run-on and quality at the new bases' 4/6-bit
and 8-bit before choosing the shipped quantization**. Gemma adds a trap: per-layer
embedding tables are `Embedding` modules too; confirm what `mlx_lm.convert` does with them
and that tables aren't quantized below usable precision. Gemma's checkpoint is
`Gemma4ForConditionalGeneration`, so confirm `mlx_lm.convert` writes a text-only model that
`mlx_lm.load` reads back (its `sanitize` already skips the vision/audio weights).

### 4.2 Roll-out

1. **Opt-in only.** `PYQMD_EXPAND_MODEL=after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6`
   works with no default change once §2 lands. The `DEFAULT_EXPAND_MODEL` stays on the
   Qwen3-based repo, so Node parity and today's behaviour are untouched.
2. Document in `CLAUDE.md` (Architecture), `COMMAND_STATUS.md` (`query` row), and
   `SKILL.md` only if the env var is user-facing (as the reranker spec noted, borderline).
3. **Gate for ever changing the default** (§5.3). A flip is a separate tiny change + spec
   amendment, as for the reranker. The old repo stays on the Hub so
   `PYQMD_EXPAND_MODEL=after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6` restores today's
   behaviour.

### 4.3 pyqmd changes (the whole list)

- `llm/expand.py`: per-`model_type` prompt settings (`/no_think` on/off, system prompt
  on/off, `enable_thinking`), read from the loaded model's config; unknown `model_type`
  falls back to "no `/no_think`, no system message". The loader currently returns
  `(model, tokenizer, tables)`; add the prompt settings to that cache entry.
- `llm/_expand_grammar.py`: SentencePiece branch in `vocab_token_bytes` (Gemma only,
  Phase 1b).
- `scripts/convert_expand_model.py` + shared module (§4.1).
- `scripts/replay_query.py`: no change expected; `--expand seeded:<salt>` already works
  on any model via `PYQMD_EXPAND_MODEL`. Its `fallback`/`malformed`/`runon` counts are the
  format gates.
- Tests: fast, no weights — prompt golden strings per `model_type`; the prompt table's
  fallback; `vocab_token_bytes` on a fake SentencePiece vocab (Gemma phase); the existing
  "isn't byte-level" error test updated. Slow: expansions on the real model parse as
  `^(lex|vec|hyde): [^\n]+$`.
- Commit types: `feat(llm)` for source under `src/pyqmd_mlx/` (it ships in the wheel);
  `feat(scripts)`; `docs`/`specs` never release. Every commit gets the model co-author
  trailer.

## 5. Evaluation plan

One change at a time: **the reranker stays Qwen3-Reranker-0.6B, the embedding model, index
and chunking stay as they are; only the expansion model varies.** Everything is compared to
the _current_ model measured in the same session on the same salts (library upgrades shift
numbers; comparing to old recorded figures would confound this).

### 5.1 Stage A — format and reward (cheap, before any retrieval run)

On the 51 queries in `evals/queries.txt`:

1. `eval.py` (greedy, HF): average reward ≥ 90% and no hard failure (score 0). Reference:
   92.0% for the current model per the Node README. Greedy is only a format check; it is not
   how pyqmd decodes.
2. The same score on **pyqmd's actual decode** (sampled, grammar-constrained, seed per
   query, a few salts): a small script feeds `expand_query` output through `reward.py`.
   Pass: mean ≥ the current model's under the same script, and zero chat-template leakage (an absolute 90% bar was wrong: the current model scores 79.7% under pyqmd's decode; see the local experiment results).
3. Compare against the current model run through the same script, not just the README number.

### 5.2 Stage B — retrieval, SciFact (and ConditionalQA)

Using `scripts/replay_query.py --expand seeded:<salt>` over **30 salts** (the requantization
spec found 10 salts only good to ±0.008 Recall@10 and concluded future gates should use
30), on the SciFact qrels queries with `data/scifact`. The current model's recorded 30-salt
mean is Recall@10 0.860 (range 0.827–0.900); re-measure it.

Gates, new model vs the current one, same salts:

| Check                                                     | Pass                                                                                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Malformed (expansion with no parseable line)              | 0 in every salt                                                                                                            |
| Run-ons (a line > 600 chars)                              | ≤ 13% (Node's rate); target ≈ 0 with the trailing-newline fix                                                              |
| Mean Recall@10, MRR, nDCG@10 over 30 salts                | each ≥ the current model's mean − 1/30 (the rule used by the grammar and seeded-sampling specs); report the standard error |
| Mean expansion latency                                    | ≤ 1.5× the current model's; report mean and max                                                                            |
| `parity/test_quality.py` (qrels mode) on the shipped seed | passes (threshold 0.8261 Recall@10 as of now)                                                                              |

If a gate fails, stop and report; don't tune seeds, constants or thresholds to pass (the
standing rule in the earlier specs). The next move is a diagnostic spike, e.g. is it `hyde:`
quality (drop-one-type replay, `--drop`) or format.

`data/conditionalqa` is the second corpus (297 queries, qrels). `replay_query.py` is built
for SciFact, so running ConditionalQA needs a small runner (the heading-breadcrumbs
experiment scripts did similar); sizing it is part of the plan. It guards against tuning to
SciFact's long scientific-claim queries, which expansion barely helps (the GGUF spec's finding).
It is a second arm, not a hard gate in the first round.

### 5.3 Stage C — end to end, and the gate for the default

1. `pyqmd bench <fixture> --samples 5` on the owner's real notes corpus with their hand-written
   fixture (as in the reranker spec, this run is the owner's: it reads their index). Pass: no
   worse than the current model by more than the sampling spread `--samples` reports.
2. Download size and cold-start load time recorded (a Hub fetch is the user's first-run
   experience); expected Granite ≈ 0.9–1.0 GB, Gemma E2B ≈ 2.6 GB text-only (see "Sizes").
3. `just test-fast`, `just test-slow` with `PYQMD_EXPAND_MODEL` set, `just test-parity` pass.

**Default flip** needs all of: Stage A, Stage B, Stage C items, the owner's real-corpus
result, the model card's license text checked, and the owner's explicit decision that model
parity with Node's expansion is no longer a goal for the default (§7).

The 30-salt replay costs about an hour per model, mostly generation and rerank; the cache in
`<work-dir>/cache.json` makes reruns cheap. Delete cached `seeded*` expansions when the model
or prompt changes.

## 6. Risks and rejected alternatives

**Risks**

- **Quality:** a 1.6B Granite may write weaker `hyde:` passages than Qwen3-1.7B. Stage A/B
  measure it; Gemma 4 E2B is the answer. The SFT format-learning part is easy (it taught Qwen3
  at 93.8% token accuracy); the retrieval effect is what could regress.
- **Train/serve drift** between two repos (§2.2): golden-string tests, asserted prefix.
- **Trailing-newline fix may not work** if a template trims message content (§2.5); fall
  back to the manual concatenation or the mid-line EOS variant.
- **Library churn:** Gemma 4 on `transformers`/TRL/PEFT, and mlx-lm's `gemma4` path, are
  months old. Granite 4.0 is a year old and better trodden. The mlx-lm installed is 0.31.3;
  the project pins `>=0.31.3`, so a conversion made with a newer version must load on 0.31.3.
- **Quantization recipe** was tuned for Qwen3; re-measure (§4.1).
- **Seed change:** the per-query seed hashes the model id, so the new model's draws differ;
  that is why Stage B uses 30 salts, not one seed.
- **Provenance and license of the training data** (§3.3) and the exact license text of the
  chosen base (read the repo's LICENSE file before upload; Hub tags say Apache 2.0).
- **Pricing figures** come from the Hub docs page today; not guaranteed.

**Rejected**

- **Zero-shot prompting a US instruct model** with no fine-tune: the original pipeline exists
  because off-the-shelf small models don't hold the `lex:`/`vec:`/`hyde:` format and entity
  preservation; the grammar would force the format but not the content quality. Not measured
  here; a possible cheap baseline arm (prompt + grammar on Granite 4.0 1B / Gemma 4 E2B,
  few-shot) to see how much the SFT buys.
- **Nemotron Bolt-E4B:** a Gemma 4 post-train; no benefit over Gemma itself, murky license.
- **Llama 3.2:** license terms (naming, notice) and age. **OLMo 3:** no model under 7B.
  **LFM:** non-Apache license. **SmolLM3:** ownership ambiguous, 3B, noisy template.
- **Taking the `mlx-community` Gemma 4 conversions** as the base: they are multimodal
  `mlx_vlm` builds; we fine-tune before converting anyway.
- **Distilling from the current Qwen model's outputs** to make training data: derivative of
  the non-US model's outputs; defeats the purpose.

## 7. Decisions taken

1. **Base:** Granite 4.0 1B, chosen on the evidence below; the Gemma 4 E2B bake-off was not needed.
2. **Training code:** `finetune/` in this repository, crediting Node qmd (MIT) for the recipe and data.
3. **Training data:** Node's expansion set, pinned to a commit; its original source is undocumented in that repository.
4. **Compute:** local `mlx_lm.lora` on Apple silicon, no cloud spend.
5. **Node parity for expansion:** the default stays on the Qwen3-based model, so parity with Node is untouched; the Granite model is opt-in.
6. **Hosting:** `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6` (Apache 2.0).
7. **ConditionalQA arm:** run in the first round, together with NFCorpus and CQADupStack-unix.

## 8. Phases and effort estimate

Elapsed working time, assuming no surprises in tooling; waits for approvals not counted.

| Phase | Work                                                                                                                           | Estimate                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| 0     | Owner decisions (§7); read each base's LICENSE; plan                                                                           | 0.5 day (writing the plan)                          |
| 1     | pyqmd prep: per-`model_type` prompt settings, golden-string tests, shared conversion module                                    | 1 day                                               |
| 1b    | SentencePiece branch in the grammar (only if Gemma runs)                                                                       | 0.5 day                                             |
| 2     | Adapt the training pipeline in `finetune/`: shared prompt builder, trailing newline, reward tokens, LoRA target check, dry-run | 1–1.5 days (more if Gemma 4's training stack bites) |
| 3     | Train Granite 4.0 1B (+ Gemma 4 E2B; optionally Granite 4.0 350M), Stage A scoring                                             | 0.5–1 day, mostly waiting on jobs                   |
| 4     | Convert, Stage B 30-salt replays for each arm vs the current model                                                             | 1 day                                               |
| 5     | Pick, host (owner's go-ahead), docs, opt-in roll-out, branch/PR                                                                | 0.5–1 day                                           |

**About 4–6 working days for Granite alone; 6–9 days with the Gemma arm.** Compute cost is
small (< $30); the time goes to integration, template/tokenizer details and the 30-salt runs.
A default flip, if the owner wants it, is a further half day plus the owner's real-corpus run.
The estimate's biggest uncertainty is library support for Gemma 4 (training side) and for
Granite's `granitemoehybrid` LoRA targets; it is dominated by Phase 2.

## Verification notes (what was and wasn't checked)

Checked on 2026-11-04: Hub model cards and `config.json` for Gemma 4 E2B/E4B, Granite 4.0
350M/1B/micro and 4.2 3B, LFM2.5 1.2B, Llama 3.2 1B; Hub API listings for Google, IBM, Ai2,
Liquid, NVIDIA, Microsoft, SmolLM3 and `mlx-community`; the real `vocab_token_bytes` and chat
templates against the Gemma 4 E2B, Granite 4.0 350M, Granite 4.2 3B, SmolLM3, Llama 3.2 (via an
Unsloth mirror, since Meta's repo is gated) and Qwen3 tokenizers; the architectures present in
the installed mlx-lm.

Not checked: any model's weights, anything about `mlx-lm` loading Granite 4.0 1B or Gemma 4
for real (needs a weights download, ~1 GB for Granite 4-bit and more for Gemma; ask first),
LoRA target-module names, `transformers`/TRL/PEFT behaviour, the LFM or Phi license texts and
tokenizers, Granite 4.0 1B's rendered template (its 350M sibling's was checked), and
sources for Nemotron Bolt-E4B beyond Hub tags and secondary coverage.

## Granite 4.0 1B local experiment (2026-11-04)

Run entirely on the development machine (Apple M4 Max, 128 GB) with no Hugging Face
spend: `mlx_lm.lora` for training, `mlx_lm fuse` + `mlx_lm.convert` for the MLX model,
pyqmd's own `replay_query.py` for retrieval. Scratch work lives outside the repo; the
product change is one small edit (below). This answers Phase 0–4 for Granite only; Gemma 4 E2B
was not run.

### What was done

- **Data:** Node's 12 JSONL files, deduplicated as Node does: 2,806 train / 312 validation
  (matches Node's `dataset_info.json`). Prompt `Expand this search query: <q>` (no
  `/no_think`, no pyqmd system message); the template's own default system text for Granite
  applies at train and serve alike. Target = hyde-first lines **plus a trailing `\n`**, then the
  EOS token (`...\n<|end_of_text|>`); verified in the token ids (`\n` is the token before EOS).
  Written as `{"text": ...}` so the template can't trim the newline. Mean 125 tokens per
  example, max 224.
- **Training:** `mlx_lm.lora` on the bf16 weights, LoRA r=16, scale 2.0 (= α/r of Node's
  α=32), no dropout, all attention and MLP projections in all 40 layers (20.3M trainable
  parameters, 1.25%), 5 epochs (3,500 minibatches of 4, accumulation 4, lr 2e-4 cosine with 3%
  warmup), max length 512. About 30 minutes at ~2 it/s and 8.2 GB peak memory. The LoRA
  target names in mlx-lm's model are the plain `q/k/v/o_proj` and `gate/up/down_proj` (that
  was the open question for the TRL/PEFT side; it stays open there).
- **Validation loss** (whole-sequence, includes the prompt, so not comparable to Node's
  0.304): 0.692 (iter 500), **0.648 (iter 1000, best)**, 0.702, 0.726, 0.828, 0.923, 0.932
  (iter 3500). Training loss fell to ~0.13: Node's 5-epoch recipe overfits on this base. Two
  checkpoints were evaluated: 1000 (best validation loss) and 3500 (Node's recipe as is).
- **Conversion:** `mlx_lm fuse` into bf16 (3.0 GB), then `mlx_lm.convert` with the existing
  `quant_bits` mixed 4/6-bit recipe (4.984 bits per weight): **977 MB on disk** per checkpoint
  (the Hub's 4-bit build is 0.92 GB). `mlx_lm.convert` and `mlx_lm.load` handled the
  `granitemoehybrid` architecture with no changes.
- **Code:** `src/pyqmd_mlx/llm/expand.py` chooses the prompt by the loaded model's
  `args.model_type` (`qwen3` → `/no_think` + Node's system prompt, anything else → bare user
  message), with fast tests; 53 expansion tests pass. The tokenizer needed no change (§2.1).

### How the evaluation was corrected

Two mistakes in my first pass were found and fixed; the numbers below are the corrected ones.

1. The first Granite runs used `python -I`, which ignores `PYTHONPATH`, so they imported the main
   checkout's old `expand.py` and served Granite the Qwen3-style prompt (`/no_think` plus Node's
   system message) although it was trained on the plain one ("arm B", kept below as a data point).
2. The corrected SciFact rerun then forgot `PYQMD_EXPAND_MODEL`, so it replayed Qwen (the output
   lines were byte-identical to Qwen's). It was rerun with the model set.

Memory: running several of these jobs at once exhausted memory (swap filled, the training run
and one evaluation died with Metal out-of-memory errors). Training was resumed from its iteration-2500 checkpoint;
the evaluations were rerun one at a time with MLX's buffer cache capped at 3 GB and a swap watchdog.

### Results

Arms: **A** = trained and served with the plain prompt (the design above, checkpoint 3500);
**B** = the same model served with the Qwen3-style prompt by accident; **C** = trained _and_ served
with the Qwen3-style prompt (resumed from its checkpoint 2500 after the crash, so not a clean
run). Qwen = the current default model.

**Stage A, reward rubric** (Node's `reward.py`, 51 queries × 3 salts = 153 expansions through
pyqmd's real decode):

| Model                                  | Mean score    | Score 0 | Run-ons | Mean expansion time, idle machine |
| -------------------------------------- | ------------- | ------- | ------- | --------------------------------- |
| Current (Qwen3-1.7B)                   | 79.7%         | 0       | 3       | 0.41 s                            |
| Granite, arm A                         | **84.4%**     | 0       | 0       | 0.48 s                            |
| Granite, arm B (ckpt 3500 / ckpt 1000) | 83.5% / 82.9% | 0       | 0       |                                   |
| Granite, arm C                         | 80.5%         | 0       | 0       |                                   |

The idle-machine times (51 × 3 expansions each, nothing else running) show Granite about 17%
slower than Qwen per expansion, not equal; earlier "same speed" figures came from runs that shared the GPU.

**Stage B, SciFact** (30 salts, `replay_query.py`, 30 qrels queries, current Qwen3 reranker):

| Model                              | Recall@10             | MRR        | nDCG@10    | Malformed | Run-ons (of 840) |
| ---------------------------------- | --------------------- | ---------- | ---------- | --------- | ---------------- |
| Current (Qwen3-1.7B)               | 0.9069 (sd 0.028)     | 0.7980     | 0.8164     | 0         | 54               |
| Granite, arm A                     | **0.9267** (sd 0.017) | **0.8280** | **0.8400** | 0         | 0                |
| Granite, arm B (accidental prompt) | 0.9258                | 0.8271     | 0.8386     | 0         | 0                |

Paired over the 30 salts, Granite A − Qwen: MRR +0.0300 (se 0.0043, better in 27 of 30 salts),
nDCG@10 +0.0237 (se 0.0038, 27 of 30), Recall@10 +0.0198 (se 0.0052, 17 of 30).

**Stage B, ConditionalQA** (120 sampled train questions, 10 salts, page-level metrics, full
pipeline with the Qwen3 reranker):

| Model                        | page MRR         | page nDCG@10     | page Recall@10   |
| ---------------------------- | ---------------- | ---------------- | ---------------- |
| Current (Qwen3-1.7B)         | 0.5971           | 0.6323           | 0.7558           |
| Granite, arm A               | 0.5942           | 0.6304           | 0.7575           |
| Granite A − Qwen (paired se) | −0.0029 (0.0040) | −0.0019 (0.0034) | +0.0017 (0.0051) |
| Granite, arm B               | 0.5918           | 0.6275           | 0.7525           |
| Granite, arm C               | 0.6013           | 0.6372           | 0.7642           |

A tie. Arm C is nominally ahead of Qwen (+0.004 MRR, +0.005 nDCG@10, +0.008 Recall@10, 1–1.6
standard errors) and of arm A, but the difference is within noise and arm C has not been
run on the other corpora.

**Stage B, NFCorpus** (200 sampled test queries, graded relevance, ~42 relevant documents per
query, 5 salts, full pipeline; short NutritionFacts-style titles as queries):

| Setup                        | nDCG@10          | MRR              | Recall@10        | Hit@10           |
| ---------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| No expansion                 | 0.3875           | 0.5751           | 0.1904           | 0.760            |
| Current (Qwen3-1.7B)         | 0.3875           | 0.5752           | 0.1878           | 0.756            |
| Granite, arm A               | 0.3908           | 0.5777           | 0.1861           | 0.745            |
| Granite A − Qwen (paired se) | +0.0033 (0.0014) | +0.0025 (0.0023) | −0.0018 (0.0003) | −0.0110 (0.0038) |

**Stage B, CQADupStack-unix** (47,382 posts, 200 sampled test queries, ~1.6 relevant
duplicates per query, 5 salts):

| Setup                        | nDCG@10          | MRR              | Recall@10        | Hit@10           |
| ---------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| No expansion                 | 0.4491           | 0.4470           | 0.5680           | 0.655            |
| Current (Qwen3-1.7B)         | 0.4419           | 0.4342           | 0.5715           | 0.647            |
| Granite, arm A               | 0.4380           | 0.4235           | 0.5763           | 0.649            |
| Granite A − Qwen (paired se) | −0.0039 (0.0073) | −0.0107 (0.0081) | +0.0048 (0.0084) | +0.0020 (0.0072) |

Expansion does not help on either new corpus: with no expansion the numbers are the same or
slightly better. That is a property of those corpora (clear, short queries; judgments
that only mark duplicates or citations), not specific to Granite, and it is why the SciFact claims
(long, declarative) are where the two models differ.

### Reading the results against §5's gates

- Malformed = 0 on SciFact; run-ons 0 against 6% for Qwen: the trailing-newline training
  target worked (§2.5) ✓.
- Every metric on every corpus is above the current model's mean − 1/30. Granite is clearly
  better on SciFact (+0.02 to +0.03, 4–7 standard errors), tied on ConditionalQA and
  NFCorpus, and within noise but nominally lower on CQADupStack-unix (MRR −0.011, se 0.008) ✓.
- Latency ≤ 1.5× ✓ (about 1.17×, idle machine).
- The absolute 90% reward bar in §5.1 was wrong as written; relative to the current model Granite is +4.7 points ✓.
- `parity/test_quality.py` (qrels mode, the shipped seed) passes for the default Qwen3 model after the
  prompt-settings change and for Granite arm A (this was run with the correct prompt: MRR 0.8241, nDCG@10 0.8347,
  Recall@10 0.9267 against Node's 0.7579 / 0.7781 / 0.8722) ✓. One seed's draw.
- Not run: `just test-slow`/`test-parity` in full, the owner's `pyqmd bench` on a real notes corpus,
  Gemma 4 E2B, arm C on SciFact, NFCorpus and CQADupStack-unix.
- Caveats: one training run per arm, English-only data, and neither new corpus shows an
  expansion benefit at all, so the short-query claim that motivated adding them is untested here.

### Where this leaves the decision

Granite 4.0 1B trained and served with the plain prompt (arm A) passes every gate and is the
model to ship opt-in. The Qwen3-style prompt does not need to be adopted: arm B's apparent
advantage was just Granite's real advantage seen under a different prompt, and arm C is not
better on the clean metrics (reward 80.5% vs 84.4%). The Gemma 4 E2B bake-off is not needed.
The remaining questions are the owner's own-notes `pyqmd bench` run once a larger notes corpus is available, and whether to make Granite
the default later (a separate change). Hosting: `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6`.
