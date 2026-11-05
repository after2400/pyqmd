"""pyqmd's config file: which model set (or models) the CLI and MCP server use.

TOML, read with tomllib. Only the `[models]` table exists so far:

    [models]
    set = "granite"          # a name in _constants.MODEL_SETS
    # expand = "org/repo"    # explicit overrides, beat `set`
    # rerank = "org/repo"

See docs/specs/2026-11-04-model-sets-and-config-design.md.
"""

import json
import os
import tempfile
import tomllib
from pathlib import Path

CONFIG_ENV_VAR = "PYQMD_CONFIG"
MODELS_KEYS = ("set", "expand", "rerank")


class ConfigError(ValueError):
    """The config file is unreadable or invalid. A ValueError, so the CLI
    prints one `Error:` line and exits 1, and the MCP server returns
    `is_error`."""


def config_path() -> Path:
    """$PYQMD_CONFIG, else $XDG_CONFIG_HOME/pyqmd/config.toml, else
    ~/.config/pyqmd/config.toml. Empty variables count as unset."""
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "pyqmd" / "config.toml"


def _read(path: Path) -> dict:
    """The whole file as a dict ({} when it does not exist)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the config file: {exc.strerror or exc}") from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def load_models_config() -> dict[str, str]:
    """The non-empty entries of the `[models]` table, validated. Unknown
    top-level tables are ignored; unknown keys under [models] are an error
    (a typo would otherwise silently do nothing)."""
    path = config_path()
    data = _read(path)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ConfigError(f"{path}: models must be a table, like [models]")
    out: dict[str, str] = {}
    for key, value in models.items():
        if key not in MODELS_KEYS:
            raise ConfigError(
                f"{path}: unknown key '{key}' under [models] (valid keys: {', '.join(MODELS_KEYS)})"
            )
        if not isinstance(value, str):
            raise ConfigError(f"{path}: models.{key} must be a string")
        if value:
            out[key] = value
    return out


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string (same escapes).
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
        return "[" + ", ".join(_scalar(x) for x in value) + "]"
    raise ConfigError(f"cannot rewrite a config value of type {type(value).__name__}")


def _dump(data: dict, prefix: str = "") -> list[str]:
    lines = [f"{k} = {_scalar(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for key, value in data.items():
        if isinstance(value, dict):
            name = f"{prefix}.{key}" if prefix else key
            lines += ["", f"[{name}]", *_dump(value, name)]
    return lines


def write_models_config(models: dict[str, str]) -> Path:
    """Replace the `[models]` table with `models`, keeping every other
    table, and write the file atomically. Comments are not preserved. An
    invalid existing file is never overwritten. Returns the path."""
    path = config_path()
    data = _read(path)
    if models:
        data["models"] = dict(models)
    else:
        data.pop("models", None)
    text = "\n".join(_dump(data)).lstrip("\n") + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
