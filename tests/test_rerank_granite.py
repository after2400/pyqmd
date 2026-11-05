"""IBM Granite reranker against PyTorch reference logits. Loads the real model
(~600 MB, downloaded into the HF cache on first run), so it is `slow`.

The reference logits come from transformers' ModernBertForSequenceClassification
on the same four query/passage pairs; the MLX port must match to 1e-3."""

import pytest

from pyqmd_mlx.llm import rerank
from pyqmd_mlx.llm._constants import GRANITE_RERANK_MODEL, RERANK_CALIBRATION
from pyqmd_mlx.llm._cross_encoder import load_cross_encoder, score_logits

pytestmark = pytest.mark.slow

PASSWORD = "how do I reset my password"
TIDES = "what causes tides"
REFERENCE = [
    (
        PASSWORD,
        "To reset your password, open Settings, choose Security, and click Reset password.",
        2.411,
    ),
    (PASSWORD, "The quarterly revenue grew 12% year over year driven by cloud services.", 0.754),
    (TIDES, "Tides are caused by the gravitational pull of the moon and sun on the oceans.", 2.793),
    (TIDES, "Python lists are mutable sequences that support slicing.", 0.688),
]


@pytest.fixture(scope="module")
def encoder():
    return load_cross_encoder(GRANITE_RERANK_MODEL)


@pytest.mark.parametrize("query,document,expected", REFERENCE)
def test_logit_matches_the_pytorch_reference(encoder, query, document, expected):
    (logit,) = score_logits(encoder, query, [document])
    assert logit == pytest.approx(expected, abs=1e-3)


def test_rerank_scores_the_relevant_passage_above_the_irrelevant_one():
    for query in (PASSWORD, TIDES):
        relevant, irrelevant = (d for q, d, _ in REFERENCE if q == query)
        high, low = rerank(query, [relevant, irrelevant], model=GRANITE_RERANK_MODEL)
        assert high > low
        assert 0.0 < low < high < 1.0


def test_rerank_applies_the_calibration_table_entry():
    bias, temperature = RERANK_CALIBRATION.get(GRANITE_RERANK_MODEL, (0.0, 1.0))
    query, document, expected_logit = REFERENCE[0]
    (score,) = rerank(query, [document], model=GRANITE_RERANK_MODEL)
    from pyqmd_mlx.llm._cross_encoder import calibrate

    assert score == pytest.approx(calibrate(expected_logit, bias, temperature), abs=1e-3)
