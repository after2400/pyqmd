"""With the Granite set selected, no model of the default (Qwen-based) set is
ever requested: not to load, not to download, not to look up in the cache.

This is the guarantee behind running pyqmd on a network that rejects the
default models (see docs/specs/2026-11-04-model-sets-and-config-design.md,
section 5). The model-calling functions Store would use are replaced with
recorders, so the test sees every model id the pipeline asks for.
"""

import re
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app
from pyqmd_mlx.llm import _download, _hub
from pyqmd_mlx.llm._config import CONFIG_ENV_VAR
from pyqmd_mlx.llm._constants import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EXPAND_MODEL,
    DEFAULT_RERANK_MODEL,
    EXPAND_MODEL_ENV_VAR,
    GRANITE_EXPAND_MODEL,
    GRANITE_RERANK_MODEL,
    MODEL_SET_ENV_VAR,
    RERANK_MODEL_ENV_VAR,
)
from pyqmd_mlx.store import Store

runner = CliRunner()
QWEN_IDS = {DEFAULT_EXPAND_MODEL, DEFAULT_RERANK_MODEL}
GRANITE_IDS = {DEFAULT_EMBED_MODEL, GRANITE_EXPAND_MODEL, GRANITE_RERANK_MODEL}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A real index with one document, the three model functions replaced by
    recorders, and a fake hub that also records every id it is asked about."""
    for var in (MODEL_SET_ENV_VAR, EXPAND_MODEL_ENV_VAR, RERANK_MODEL_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "config.toml"))
    db = tmp_path / "index.sqlite"
    monkeypatch.setenv("PYQMD_DB", str(db))
    seen = types.SimpleNamespace(models=[], hub=[])

    def embed(texts, model, kind="query", title=None):
        seen.models.append(("embed", model))
        return [[1.0, 0.0]] * len(texts)

    def rerank(query, documents, model):
        seen.models.append(("rerank", model))
        return [1.0] * len(documents)

    def expand(query, model, salt=None):
        seen.models.append(("expand", model))
        return []

    monkeypatch.setattr("pyqmd_mlx.store.store._default_embed", embed)
    monkeypatch.setattr("pyqmd_mlx.store.store._default_rerank", rerank)
    monkeypatch.setattr("pyqmd_mlx.store.store._default_expand_query", expand)

    def snapshot_download(model_id, **_):
        seen.hub.append(model_id)
        return "/cache/" + model_id

    def lookup(model_id, *args, **_):
        seen.hub.append(model_id)
        return None

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lookup)
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.models.is_downloaded",
        lambda model_id: lookup(model_id) or False,
    )
    monkeypatch.setattr(_download, "is_downloaded", lambda model_id: lookup(model_id) or False)
    monkeypatch.setattr(_hub, "is_downloaded", lambda model_id: lookup(model_id) or False)

    store = Store(str(db))
    store.add_collection("notes", "/notes")
    body = "authentication setup guide"
    h = store.hash_content(body)
    store.insert_content(h, body, "2026-01-01T00:00:00Z")
    store.insert_document(
        "notes", "auth.md", "Auth", h, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"
    )
    store.index_content(h, body, model=DEFAULT_EMBED_MODEL)
    store.close()
    seen.models.clear()
    return seen


def _run_everything():
    """Every command that resolves or touches models."""
    for args in (
        ["query", "authentication", "--format", "json"],
        ["query", "authentication", "--no-rerank", "--format", "json"],
        ["vsearch", "authentication", "--format", "json"],
        ["status"],
        ["models", "show"],
        ["models", "list"],
        ["pull"],
        ["models", "use", "granite", "--download"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)


def _assert_no_default_set_model(seen):
    asked = {model for _role, model in seen.models} | set(seen.hub)
    assert not QWEN_IDS & asked, f"asked for default-set models: {QWEN_IDS & asked}"
    assert {GRANITE_EXPAND_MODEL, GRANITE_RERANK_MODEL} <= asked
    assert asked <= GRANITE_IDS | {"/notes"}


def test_the_set_chosen_by_models_use_never_touches_the_default_models(pipeline):
    assert runner.invoke(app, ["models", "use", "granite"]).exit_code == 0

    _run_everything()

    _assert_no_default_set_model(pipeline)


def test_the_set_chosen_by_the_environment_variable(pipeline, monkeypatch):
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "granite")

    _run_everything()

    _assert_no_default_set_model(pipeline)


def test_the_flags_alone_also_avoid_the_default_models(pipeline):
    result = runner.invoke(
        app,
        [
            "query",
            "authentication",
            "--rerank-model",
            GRANITE_RERANK_MODEL,
            "--expand-model",
            GRANITE_EXPAND_MODEL,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    asked = {model for _role, model in pipeline.models}
    assert asked == {DEFAULT_EMBED_MODEL, GRANITE_EXPAND_MODEL, GRANITE_RERANK_MODEL}


def test_without_a_choice_the_default_models_are_used(pipeline):
    assert runner.invoke(app, ["query", "authentication", "--format", "json"]).exit_code == 0

    asked = {model for _role, model in pipeline.models}
    assert QWEN_IDS <= asked


def test_no_other_code_path_names_a_default_model():
    """The default ids may only be named where they are defined, re-exported or
    used as a parameter default or in a docstring (llm/); any other module
    naming one could load it behind the user's selection."""
    src = Path(__file__).resolve().parent.parent / "src" / "pyqmd_mlx"
    pattern = re.compile(r"DEFAULT_(RERANK|EXPAND)_MODEL")
    hits = {
        str(path.relative_to(src))
        for path in src.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    }
    allowed = {
        "llm/_constants.py",
        "llm/__init__.py",
        "llm/rerank.py",
        "llm/expand.py",
        "llm/_load_errors.py",
    }
    assert hits <= allowed, f"unexpected references: {hits - allowed}"
