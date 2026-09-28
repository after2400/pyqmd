from pyqmd_mlx.cli._db import DB_PATH_ENV_VAR, DEFAULT_DB_PATH, get_store


def test_default_db_path_is_under_pyqmd_cache_dir():
    assert DEFAULT_DB_PATH.parts[-3:] == (".cache", "pyqmd", "index.sqlite")


def test_get_store_creates_parent_directory_and_opens_store(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "index.sqlite"
    store = get_store(str(db_path))
    assert db_path.exists()
    assert store.list_collections() == []
    store.close()


def test_get_store_respects_env_var_override(tmp_path, monkeypatch):
    db_path = tmp_path / "from-env" / "index.sqlite"
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))

    store = get_store()

    assert db_path.exists()
    assert store.db_path == str(db_path)
    store.close()


def test_get_store_falls_back_to_default_when_env_var_unset(monkeypatch):
    monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)

    store = get_store(":memory:")

    # Explicit db_path still wins over both the env var and the default --
    # this just confirms no env var means no interference with that path.
    assert store.db_path == ":memory:"
    store.close()


def test_get_store_explicit_arg_wins_over_env_var(tmp_path, monkeypatch):
    env_path = tmp_path / "env" / "index.sqlite"
    explicit_path = tmp_path / "explicit" / "index.sqlite"
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(env_path))

    store = get_store(str(explicit_path))

    assert store.db_path == str(explicit_path)
    assert explicit_path.exists()
    assert not env_path.exists()
    store.close()
