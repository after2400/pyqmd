"""Seeding for query expansion: the default seed, and the salted seeds
bench --samples and scripts/replay_query.py --expand seeded:<salt> use."""

import hashlib

from pyqmd_mlx.llm import expand as expand_mod
from pyqmd_mlx.llm.expand import _seed_for


def _sha_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def test_unsalted_seed_is_unchanged():
    assert _seed_for("vitamin d", "some/model") == _sha_seed("some/model\nvitamin d")


def test_salted_seed_appends_the_salt():
    assert _seed_for("vitamin d", "some/model", "3") == _sha_seed("some/model\nvitamin d\n3")


def test_salt_none_matches_no_salt():
    assert _seed_for("q", "m", None) == _seed_for("q", "m")


def test_different_salts_give_different_seeds():
    assert len({_seed_for("q", "m", s) for s in (None, "1", "2", "3")}) == 4


def test_expand_query_passes_the_salted_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(
        expand_mod,
        "_generate_expansion",
        lambda query, model, seed: seen.append(seed) or ["lex: x"],
    )

    expand_mod.expand_query("q", "m")
    expand_mod.expand_query("q", "m", salt="2")

    assert seen == [_seed_for("q", "m"), _seed_for("q", "m", "2")]
