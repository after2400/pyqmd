"""expand_query(): qmd's own fine-tuned query-expansion model, run via mlx-lm.

Ported from finetune/eval.py's prompt format. Decodes with Node's sampling
settings under Node's output grammar (see _expand_grammar), seeded per
(model, query) so results are deterministic without Node's llm_cache.
"""

import hashlib
import os
import re

from ._cache import get_or_load
from ._constants import DEFAULT_EXPAND_MODEL, EXPAND_MODEL_ENV_VAR
from ._expand_grammar import DEAD, START0, GrammarTables, build_tables, vocab_token_bytes
from ._hub import load_quietly_if_cached

_expand_cache: dict = {}
# Used only by scripts/replay_query.py's unconstrained greedy mode: the
# grammar makes a leading <think> block impossible, and Node doesn't strip.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Node's decoding settings (qmd/src/llm.ts, expandQuery), which says greedy
# decoding must not be used. Greedy cost Recall@10 on SciFact (see
# docs/specs/2026-09-25-seeded-expansion-sampling-design.md).
_TEMPERATURE = 0.7
_TOP_K = 20
_TOP_P = 0.8
_PRESENCE_PENALTY = 0.5
_PRESENCE_CONTEXT = 64
_MAX_TOKENS = 600

# Node's session.prompt runs through node-llama-cpp's LlamaChatSession,
# which prepends this default system message (node-llama-cpp v3.20.0,
# dist/config.js defaultChatSystemPrompt). Without it pyqmd's expansions
# score measurably worse on SciFact (see the seeded-expansion spec's
# "Amendment").
_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead "
    "of answering something incorrectly. If you don't know the answer to a question, don't "
    "share false information."
)


def _seed_for(query: str, model: str, salt: str | None = None) -> int:
    """The sampling seed for one (model, query): the first 4 bytes of a
    sha256, big-endian. A per-query seed keeps expansion deterministic
    without Node's llm_cache. A salt picks a different, still repeatable
    draw -- bench --samples uses salt str(i) for sample i, the same seed as
    scripts/replay_query.py --expand seeded:<i>."""
    key = f"{model}\n{query}" if salt is None else f"{model}\n{query}\n{salt}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _keyed_sampler(seed: int):
    """A top-p/top-k/temperature sampler that carries its own PRNG key.

    mlx_lm's make_sampler always draws from MLX's global RNG, and mlx has no
    public way to restore a saved global state, so seeding it would leak into
    every later mx.random user. Filters apply in make_sampler's order."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_top_k, apply_top_p

    key = mx.random.key(seed)

    def sampler(logprobs):
        nonlocal key
        key, subkey = mx.random.split(key)
        logprobs = apply_top_p(logprobs, _TOP_P)
        logprobs = apply_top_k(logprobs, _TOP_K)
        return mx.random.categorical(logprobs * (1 / _TEMPERATURE), key=subkey)

    return sampler


def _logits_processors() -> list:
    from mlx_lm.sample_utils import make_logits_processors

    return make_logits_processors(
        presence_penalty=_PRESENCE_PENALTY, presence_context_size=_PRESENCE_CONTEXT
    )


def resolve_expand_model() -> str:
    """The effective expansion model: $PYQMD_EXPAND_MODEL when set and
    non-empty, else DEFAULT_EXPAND_MODEL."""
    return os.environ.get(EXPAND_MODEL_ENV_VAR) or DEFAULT_EXPAND_MODEL


class ExpansionModelError(RuntimeError):
    """The query-expansion model couldn't be loaded (bad repo id or path,
    not downloadable, ...). Replaces mlx_lm's raw exception so the CLI and
    MCP server can report one clean line that names the override."""


def _load(model_id: str):
    # Deferred -- see embed.py's _load for why: a module-level mlx_lm import
    # would tax every CLI command via pyqmd_mlx.store.Store, not just expand_query().
    from mlx_lm import load as _load_lm_model

    try:
        model, tokenizer = load_quietly_if_cached(model_id, _load_lm_model)
        tables = _build_grammar_tables(tokenizer)
    except Exception as exc:
        raise ExpansionModelError(
            f"Could not load query-expansion model '{model_id}': {str(exc).rstrip('.')}. "
            f"Set {EXPAND_MODEL_ENV_VAR} to a Hugging Face repo id or a local "
            "MLX model directory to override."
        ) from exc
    return model, tokenizer, tables


def _build_grammar_tables(tokenizer) -> GrammarTables:
    """Node's output grammar over this tokenizer's vocab (see _expand_grammar)."""
    return build_tables(vocab_token_bytes(tokenizer), tokenizer.eos_token_ids)


def _grammar_processor(tables: GrammarTables):
    """A logits processor that allows only continuations of Node's expansion
    grammar. Fresh per generation, since it tracks the grammar state.

    Node (qmd/src/llm.ts, expandQuery) applies its grammar from the first
    generated token, so no <think> block, and allows EOS only after a
    complete line. Its last hyde: line sometimes runs on because of that;
    pyqmd matches it (see the expansion-grammar spec)."""
    import mlx.core as mx

    state = START0
    seen: int | None = None

    def processor(tokens, logits):
        nonlocal state, seen
        # tokens is the whole history, prompt included. The first call comes
        # before any generated token; each later call adds one.
        if seen is not None:
            for token in tokens[seen:].tolist():
                state = tables.next_state[state][token]
        seen = tokens.size
        return mx.where(_state_mask(tables, state, logits.shape[-1]), logits, float("-inf"))

    return processor


def _state_mask(tables: GrammarTables, state: int, width: int):
    """One state's allowed-token mask at the logits' width. Qwen3's logits
    are wider than its tokenizer; the extra ids are always masked. Cached on
    tables, since each state's mask is reused across queries."""
    import mlx.core as mx

    key = (state, width)
    if key not in tables.mask_cache:
        row = tables.next_state[state][:width]
        allowed = [s != DEAD for s in row] + [False] * (width - len(row))
        tables.mask_cache[key] = mx.array(allowed)
    return tables.mask_cache[key]


def _generate_expansion(query: str, model: str, seed: int) -> list[str]:
    """One sampled expansion with an explicit seed. expand_query picks the
    seed; the replay harness calls this directly to try other seeds."""
    from mlx_lm import generate as _generate

    expand_model, tokenizer, tables = get_or_load(_expand_cache, model, _load)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"/no_think Expand this search query: {query}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = _generate(
        expand_model,
        tokenizer,
        prompt=prompt,
        max_tokens=_MAX_TOKENS,
        sampler=_keyed_sampler(seed),
        logits_processors=[*_logits_processors(), _grammar_processor(tables)],
    )
    # Node splits on "\n" only; splitlines() would also split on characters
    # the grammar allows inside content (\r, \u2028, ...).
    return [line for line in text.split("\n") if line.strip()]


def expand_query(
    query: str, model: str = DEFAULT_EXPAND_MODEL, salt: str | None = None
) -> list[str]:
    """Expand a search query into hyde:/lex:/vec: lines (see finetune/CLAUDE.md).

    Deterministic per (query, model, salt): the same inputs always give the
    same lines, on any machine, for a given mlx/mlx-lm version. salt=None is
    the default draw every search uses; bench --samples passes others."""
    return _generate_expansion(query, model, _seed_for(query, model, salt))
