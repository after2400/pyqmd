"""Which model each role uses: flag > role env var > PYQMD_MODEL_SET > config
file entry > config file set > built-in default (the flag is applied by the
caller, which passes an explicit model to Store).

See docs/specs/2026-11-04-model-sets-and-config-design.md.
"""

import os

from ._config import ConfigError as ConfigError  # noqa: PLC0414  (re-exported)
from ._config import config_path, load_models_config
from ._constants import (
    EXPAND_MODEL_ENV_VAR,
    MODEL_SET_ENV_VAR,
    MODEL_SETS,
    RERANK_MODEL_ENV_VAR,
    ModelSet,
)

ROLE_ENV_VARS = {"expand": EXPAND_MODEL_ENV_VAR, "rerank": RERANK_MODEL_ENV_VAR}


def model_set(name: str, where: str) -> ModelSet:
    """The named set, or a ConfigError naming `where` (the env var or file) it came from."""
    try:
        return MODEL_SETS[name]
    except KeyError:
        raise ConfigError(
            f"unknown model set '{name}' in {where} (valid sets: {', '.join(MODEL_SETS)})"
        ) from None


def resolve_model_source(role: str) -> tuple[str, str]:
    """(model id, where it came from) for 'expand' or 'rerank'."""
    env_var = ROLE_ENV_VARS[role]
    if value := os.environ.get(env_var):
        return value, f"${env_var}"
    if name := os.environ.get(MODEL_SET_ENV_VAR):
        return getattr(
            model_set(name, f"${MODEL_SET_ENV_VAR}"), role
        ), f"${MODEL_SET_ENV_VAR}={name}"
    config = load_models_config()
    path = config_path()
    if role in config:
        return config[role], f"config {path} (models.{role})"
    if "set" in config:
        name = config["set"]
        return getattr(model_set(name, str(path)), role), f"config {path} (set = {name})"
    return getattr(MODEL_SETS["default"], role), "default"


def resolve_model(role: str) -> str:
    return resolve_model_source(role)[0]
