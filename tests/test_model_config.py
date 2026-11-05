"""The pyqmd config file: where it lives, how it is validated, and how
`models use` rewrites it. No models, no network."""

import pytest

from pyqmd_mlx.llm._config import (
    CONFIG_ENV_VAR,
    ConfigError,
    config_path,
    load_models_config,
    write_models_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


def test_path_prefers_the_pyqmd_config_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "mine.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_path() == tmp_path / "mine.toml"


def test_path_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_path() == tmp_path / "xdg" / "pyqmd" / "config.toml"


def test_path_falls_back_to_the_home_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_path() == tmp_path / ".config" / "pyqmd" / "config.toml"


def test_empty_variables_count_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_ENV_VAR, "")
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_path() == tmp_path / ".config" / "pyqmd" / "config.toml"


def _use(monkeypatch, tmp_path, text=None):
    path = tmp_path / "config.toml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    return path


def test_a_missing_file_means_nothing_is_configured(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path)
    assert load_models_config() == {}


def test_a_valid_file_returns_only_the_keys_present(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, '[models]\nset = "granite"\nrerank = "org/repo"\n')
    assert load_models_config() == {"set": "granite", "rerank": "org/repo"}


def test_a_file_without_a_models_table_is_fine(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "[other]\nx = 1\n")
    assert load_models_config() == {}


def test_invalid_toml_names_the_file(monkeypatch, tmp_path):
    path = _use(monkeypatch, tmp_path, "[models\nset = 'granite'")
    with pytest.raises(ConfigError, match="config.toml") as excinfo:
        load_models_config()
    assert str(path) in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)


def test_a_non_string_value_is_rejected(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, "[models]\nset = 3\n")
    with pytest.raises(ConfigError, match="models.set must be a string"):
        load_models_config()


def test_an_unknown_models_key_is_rejected(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, '[models]\nsett = "granite"\n')
    with pytest.raises(ConfigError, match="unknown key 'sett'"):
        load_models_config()


def test_models_must_be_a_table(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, 'models = "granite"\n')
    with pytest.raises(ConfigError, match="models must be a table"):
        load_models_config()


def test_empty_strings_are_dropped(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, '[models]\nset = ""\nexpand = "org/repo"\n')
    assert load_models_config() == {"expand": "org/repo"}


def test_write_creates_the_directory_and_file(monkeypatch, tmp_path):
    path = tmp_path / "deep" / "dir" / "config.toml"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert write_models_config({"set": "granite"}) == path
    assert load_models_config() == {"set": "granite"}


def test_write_replaces_the_models_table_and_keeps_other_tables(monkeypatch, tmp_path):
    path = _use(
        monkeypatch,
        tmp_path,
        'title = "mine"\n[models]\nset = "default"\nrerank = "a/b"\n[other.nested]\nflag = true\nn = 2\nxs = ["a", "b"]\n',
    )
    write_models_config({"set": "granite"})
    assert load_models_config() == {"set": "granite"}
    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "mine"
    assert data["other"]["nested"] == {"flag": True, "n": 2, "xs": ["a", "b"]}


def test_write_handles_awkward_strings(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path)
    awkward = '/tmp/dir with "quotes" and \\ backslash/é'
    write_models_config({"expand": awkward})
    assert load_models_config() == {"expand": awkward}


def test_write_refuses_to_overwrite_an_invalid_file(monkeypatch, tmp_path):
    path = _use(monkeypatch, tmp_path, "[models\n")
    with pytest.raises(ConfigError):
        write_models_config({"set": "granite"})
    assert path.read_text(encoding="utf-8") == "[models\n"


def test_write_leaves_no_temp_files(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path)
    write_models_config({"set": "granite"})
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
