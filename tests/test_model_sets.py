"""Model sets and how each role's model is chosen (flag > role env var >
PYQMD_MODEL_SET > config explicit > config set > default). No models."""

import pytest

from pyqmd_mlx.llm import resolve_expand_model, resolve_rerank_model
from pyqmd_mlx.llm._config import CONFIG_ENV_VAR, ConfigError
from pyqmd_mlx.llm._constants import (
    DEFAULT_EXPAND_MODEL,
    DEFAULT_RERANK_MODEL,
    EXPAND_MODEL_ENV_VAR,
    GRANITE_EXPAND_MODEL,
    GRANITE_RERANK_MODEL,
    MODEL_SET_ENV_VAR,
    MODEL_SETS,
    RERANK_MODEL_ENV_VAR,
)
from pyqmd_mlx.llm._models import resolve_model_source

ROLES = {
    "expand": (
        resolve_expand_model,
        EXPAND_MODEL_ENV_VAR,
        DEFAULT_EXPAND_MODEL,
        GRANITE_EXPAND_MODEL,
    ),
    "rerank": (
        resolve_rerank_model,
        RERANK_MODEL_ENV_VAR,
        DEFAULT_RERANK_MODEL,
        GRANITE_RERANK_MODEL,
    ),
}


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A config file written from text; the env variables start unset."""
    for var in (EXPAND_MODEL_ENV_VAR, RERANK_MODEL_ENV_VAR, MODEL_SET_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "config.toml"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))

    def write(text):
        path.write_text(text, encoding="utf-8")
        return path

    return write


def test_the_two_sets():
    assert MODEL_SETS["default"].expand == DEFAULT_EXPAND_MODEL
    assert MODEL_SETS["default"].rerank == DEFAULT_RERANK_MODEL
    assert MODEL_SETS["granite"].expand == "after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6"
    assert MODEL_SETS["granite"].rerank == "ibm-granite/granite-embedding-reranker-english-r2"
    assert MODEL_SET_ENV_VAR == "PYQMD_MODEL_SET"


@pytest.mark.parametrize("role", ROLES)
def test_nothing_set_gives_the_default(config, role):
    resolve, _env, default, _granite = ROLES[role]
    assert resolve() == default


@pytest.mark.parametrize("role", ROLES)
def test_config_set_beats_the_default(config, role):
    resolve, _env, _default, granite = ROLES[role]
    config('[models]\nset = "granite"\n')
    assert resolve() == granite


@pytest.mark.parametrize("role", ROLES)
def test_config_explicit_key_beats_config_set(config, role):
    resolve, _env, _default, _granite = ROLES[role]
    config(f'[models]\nset = "granite"\n{role} = "org/explicit"\n')
    assert resolve() == "org/explicit"


@pytest.mark.parametrize("role", ROLES)
def test_set_env_var_beats_every_config_entry(config, monkeypatch, role):
    resolve, _env, default, _granite = ROLES[role]
    config(f'[models]\nset = "granite"\n{role} = "org/explicit"\n')
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "default")
    assert resolve() == default


@pytest.mark.parametrize("role", ROLES)
def test_role_env_var_beats_the_set_env_var(config, monkeypatch, role):
    resolve, env, _default, _granite = ROLES[role]
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "granite")
    monkeypatch.setenv(env, "org/from-env")
    assert resolve() == "org/from-env"


@pytest.mark.parametrize("role", ROLES)
def test_empty_strings_count_as_unset(config, monkeypatch, role):
    resolve, env, _default, granite = ROLES[role]
    config('[models]\nset = "granite"\n')
    monkeypatch.setenv(env, "")
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "")
    assert resolve() == granite


def test_the_set_env_var_selects_one_role_independently_of_the_other(config, monkeypatch):
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "granite")
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "org/other-reranker")
    assert resolve_expand_model() == GRANITE_EXPAND_MODEL
    assert resolve_rerank_model() == "org/other-reranker"


def test_an_unknown_set_in_the_env_var_is_a_config_error(config, monkeypatch):
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "nope")
    with pytest.raises(ConfigError, match=r"unknown model set 'nope'.*default, granite") as excinfo:
        resolve_expand_model()
    assert MODEL_SET_ENV_VAR in str(excinfo.value)


def test_an_unknown_set_in_the_config_file_names_the_file(config):
    path = config('[models]\nset = "nope"\n')
    with pytest.raises(ConfigError, match="unknown model set 'nope'") as excinfo:
        resolve_rerank_model()
    assert str(path) in str(excinfo.value)


def test_an_invalid_config_file_fails_resolution(config):
    config("[models\n")
    with pytest.raises(ConfigError):
        resolve_expand_model()


def test_sources_are_reported(config, monkeypatch):
    assert resolve_model_source("rerank") == (DEFAULT_RERANK_MODEL, "default")
    path = config('[models]\nset = "granite"\nexpand = "org/x"\n')
    assert resolve_model_source("rerank") == (
        GRANITE_RERANK_MODEL,
        f"config {path} (set = granite)",
    )
    assert resolve_model_source("expand") == ("org/x", f"config {path} (models.expand)")
    monkeypatch.setenv(MODEL_SET_ENV_VAR, "default")
    assert resolve_model_source("rerank") == (DEFAULT_RERANK_MODEL, f"${MODEL_SET_ENV_VAR}=default")
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "org/r")
    assert resolve_model_source("rerank") == ("org/r", f"${RERANK_MODEL_ENV_VAR}")
