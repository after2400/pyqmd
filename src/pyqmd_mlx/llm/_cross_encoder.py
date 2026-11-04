"""Cross-encoder rerank backend: BERT-style checkpoints that score a
(query, document) pair with a classification head, run through
mlx_embeddings.

rerank() routes here when the checkpoint's config.json lists a
`*ForSequenceClassification` architecture (see is_cross_encoder). The raw
logit is mapped to a 0..1 score with a per-model bias and temperature
(`RERANK_CALIBRATION`), because the blend with the RRF position and
`--min-score` assume irrelevant ~0 and relevant ~1.

Imports of mlx and mlx_embeddings stay inside functions, as in the other
backends: a module-level import would tax every CLI command that imports
Store.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _constants
from ._hub import load_quietly_if_cached


def read_config(model_id: str) -> dict | None:
    """The checkpoint's config.json as a dict: from a local model directory,
    else the HF cache, else downloaded (the small file only). None when it
    can't be read (unknown repo, offline, malformed), so callers fall back."""
    try:
        local = Path(model_id)
        if local.is_dir():
            path = local / "config.json"
        else:
            import huggingface_hub

            cached = huggingface_hub.try_to_load_from_cache(model_id, "config.json")
            path = (
                cached
                if isinstance(cached, str)
                else huggingface_hub.hf_hub_download(model_id, "config.json")
            )
        config = json.loads(Path(path).read_text())
    except Exception:
        return None
    return config if isinstance(config, dict) else None


def is_cross_encoder(model_id: str) -> bool:
    """True when config.json declares a sequence-classification architecture."""
    architectures = (read_config(model_id) or {}).get("architectures") or []
    return any(str(name).endswith("ForSequenceClassification") for name in architectures)


def calibrate(logit: float, bias: float, temperature: float) -> float:
    """sigmoid((logit - bias) / temperature), computed without overflow."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = (logit - bias) / temperature
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fix_classifier_activation(model, activation_name: str | None) -> None:
    """Make the classification head use the activation the checkpoint asks for.

    mlx-embeddings 0.1.0's ModernBertPredictionHead hardcodes GELU, while
    checkpoints such as IBM Granite's reranker declare
    `classifier_activation: "silu"`, which shifts every logit by about 0.1.
    (Its ModelArgs also drops the field, so the name comes from config.json.)
    A no-op for other activations and once upstream builds the head right."""
    if activation_name != "silu":
        return
    import mlx.nn as nn

    if not isinstance(model.head.act, nn.SiLU):
        model.head.act = nn.SiLU()


@dataclass
class CrossEncoder:
    model: Any
    tokenizer: Any
    bias: float
    temperature: float


def _load(model_id: str):
    from mlx_embeddings.utils import load

    return load(model_id)


def load_cross_encoder(model_id: str) -> CrossEncoder:
    model, tokenizer = load_quietly_if_cached(model_id, _load)
    config = read_config(model_id) or {}
    fix_classifier_activation(model, config.get("classifier_activation"))
    # In regression mode the model returns the head's raw logit as
    # pooler_output; otherwise it applies a sigmoid, which saturates for
    # confident pairs and loses their order.
    model.is_regression = True
    bias, temperature = _constants.RERANK_CALIBRATION.get(model_id, (0.0, 1.0))
    return CrossEncoder(model, tokenizer, bias, temperature)


def _encode_pair(tokenizer, query: str, document: str):
    kwargs = {"return_tensors": "np", "max_length": _constants.CROSS_ENCODER_MAX_TOKENS}
    try:
        return tokenizer(query, document, truncation="only_second", **kwargs)
    except Exception:
        # "only_second" raises when the query alone exceeds the cap (the
        # tokenizer's Exception is not a narrower type); trim both instead.
        return tokenizer(query, document, truncation="longest_first", **kwargs)


def score_logits(encoder: CrossEncoder, query: str, documents: list[str]) -> list[float]:
    """The raw classifier logit for each (query, document) pair, in order.
    One forward pass per pair; the document is truncated, never the query
    (unless the query alone is over the cap)."""
    import mlx.core as mx

    tokenizer = getattr(encoder.tokenizer, "_tokenizer", encoder.tokenizer)
    logits = []
    for document in documents:
        encoded = _encode_pair(tokenizer, query, document)
        output = encoder.model(
            mx.array(encoded["input_ids"]), attention_mask=mx.array(encoded["attention_mask"])
        )
        logits.append(float(output.pooler_output.item()))
    return logits
