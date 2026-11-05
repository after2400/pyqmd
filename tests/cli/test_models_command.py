"""`pyqmd models` and `pyqmd pull`: choosing a model set and downloading only
what was chosen. The Hugging Face hub is faked; nothing touches the network."""

import types

import pytest
from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app
from pyqmd_mlx.llm import _download, _hub
from pyqmd_mlx.llm._config import CONFIG_ENV_VAR, load_models_config
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

runner = CliRunner()

QWEN_IDS = {DEFAULT_EXPAND_MODEL, DEFAULT_RERANK_MODEL}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A config file in tmp_path and a fake hub: `downloaded` is the set of
    repo ids in the cache, `requested` records every snapshot_download, and
    ids in `blocked` raise a 403-shaped error."""
    for var in (MODEL_SET_ENV_VAR, EXPAND_MODEL_ENV_VAR, RERANK_MODEL_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "config.toml"))
    state = types.SimpleNamespace(downloaded=set(), requested=[], blocked=set())

    def snapshot_download(model_id, **_):
        state.requested.append(model_id)
        if model_id in state.blocked:
            error = RuntimeError("403 Client Error: Forbidden for url")
            error.response = types.SimpleNamespace(status_code=403)
            raise error
        state.downloaded.add(model_id)
        return "/cache/" + model_id

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr(_hub, "is_downloaded", lambda model_id: model_id in state.downloaded)
    monkeypatch.setattr(_download, "is_downloaded", lambda model_id: model_id in state.downloaded)
    monkeypatch.setattr(
        "pyqmd_mlx.cli.commands.models.is_downloaded", lambda model_id: model_id in state.downloaded
    )
    return state


def test_bare_models_shows_the_current_choice(hub):
    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert DEFAULT_RERANK_MODEL in result.output
    assert DEFAULT_EXPAND_MODEL in result.output
    assert DEFAULT_EMBED_MODEL in result.output
    assert "not downloaded" in result.output
    assert "(not found)" in result.output
    assert hub.requested == []


def test_show_reports_where_each_choice_came_from_and_cache_state(hub, monkeypatch):
    hub.downloaded.add(GRANITE_RERANK_MODEL)
    assert runner.invoke(app, ["models", "use", "granite"]).exit_code == 0
    monkeypatch.setenv(EXPAND_MODEL_ENV_VAR, "org/from-env")

    result = runner.invoke(app, ["models", "show"])

    out = result.output
    assert f"{GRANITE_RERANK_MODEL}\n" in out
    assert "(set = granite); downloaded" in out
    assert f"org/from-env\n               ${EXPAND_MODEL_ENV_VAR}; not downloaded" in out
    assert "built in" in out


def test_show_marks_a_local_directory(hub, monkeypatch, tmp_path):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, str(tmp_path))

    result = runner.invoke(app, ["models", "show"])

    assert "local directory" in result.output


def test_list_names_both_sets_and_marks_the_active_one(hub):
    result = runner.invoke(app, ["models", "list"])

    assert result.exit_code == 0
    assert "default  (active)" in result.output
    assert "granite\n" in result.output
    assert GRANITE_EXPAND_MODEL in result.output

    assert runner.invoke(app, ["models", "use", "granite"]).exit_code == 0
    result = runner.invoke(app, ["models", "list"])
    assert "granite  (active)" in result.output
    assert "default\n" in result.output


def test_use_writes_the_config_and_touches_no_model(hub):
    result = runner.invoke(app, ["models", "use", "granite"])

    assert result.exit_code == 0
    assert "Using the 'granite' model set" in result.output
    assert load_models_config() == {"set": "granite"}
    assert hub.requested == []  # no download, and the Qwen models were never looked at


def test_use_removes_explicit_entries_that_would_still_win(hub, tmp_path):
    (tmp_path / "config.toml").write_text(
        '# my notes\n[models]\nexpand = "org/x"\nrerank = "org/y"\n', encoding="utf-8"
    )

    result = runner.invoke(app, ["models", "use", "granite"])

    assert load_models_config() == {"set": "granite"}
    assert "Removed explicit expand and rerank entries" in result.output
    assert "Comments in the config file are not preserved" in result.output


def test_use_warns_about_environment_variables_that_still_win(hub, monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV_VAR, "org/from-env")

    result = runner.invoke(app, ["models", "use", "granite"])

    assert (
        f"Note: {RERANK_MODEL_ENV_VAR} is set in this environment and still takes precedence"
        in (result.output)
    )


def test_use_rejects_an_unknown_set_and_writes_nothing(hub, tmp_path):
    result = runner.invoke(app, ["models", "use", "nope"])

    assert result.exit_code == 1
    assert "unknown model set 'nope'" in result.output
    assert "default, granite" in result.output
    assert not (tmp_path / "config.toml").exists()


def test_use_can_go_back_to_the_default_set(hub):
    runner.invoke(app, ["models", "use", "granite"])
    assert runner.invoke(app, ["models", "use", "default"]).exit_code == 0
    assert load_models_config() == {"set": "default"}


def test_use_download_fetches_exactly_the_chosen_sets_models(hub):
    result = runner.invoke(app, ["models", "use", "granite", "--download"])

    assert result.exit_code == 0
    assert set(hub.requested) == {DEFAULT_EMBED_MODEL, GRANITE_RERANK_MODEL, GRANITE_EXPAND_MODEL}
    assert not QWEN_IDS & set(hub.requested)
    assert result.output.count("downloaded") >= 3


def test_use_download_failure_keeps_the_choice_and_exits_1(hub):
    hub.blocked.add(GRANITE_EXPAND_MODEL)

    result = runner.invoke(app, ["models", "use", "granite", "--download"])

    assert result.exit_code == 1
    assert load_models_config() == {"set": "granite"}
    assert "Could not download query-expansion model" in result.output
    assert "access to the repository was denied" in result.output
    assert GRANITE_RERANK_MODEL in hub.requested  # the others were still attempted


def test_pull_downloads_the_active_models_and_skips_cached_ones(hub):
    hub.downloaded.add(DEFAULT_EMBED_MODEL)

    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 0
    assert set(hub.requested) == QWEN_IDS
    assert "cached" in result.output


def test_pull_follows_the_selected_set_and_never_asks_for_qwen(hub):
    runner.invoke(app, ["models", "use", "granite"])

    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 0
    assert set(hub.requested) == {DEFAULT_EMBED_MODEL, GRANITE_RERANK_MODEL, GRANITE_EXPAND_MODEL}


def test_pull_reports_a_blocked_default_model_with_the_way_out(hub):
    hub.blocked.update(QWEN_IDS)

    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 1
    assert "pyqmd models use granite" in result.output
    assert "Traceback" not in result.output
    assert DEFAULT_EMBED_MODEL in hub.requested  # the embedding model was not skipped


def test_pull_then_switching_to_granite_succeeds_after_a_block(hub):
    hub.blocked.update(QWEN_IDS)
    assert runner.invoke(app, ["pull"]).exit_code == 1
    hub.requested.clear()

    assert runner.invoke(app, ["models", "use", "granite"]).exit_code == 0
    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 0
    assert not QWEN_IDS & set(hub.requested)


def test_pull_still_accepts_the_node_flags(hub):
    assert runner.invoke(app, ["pull", "--refresh", "--progress"]).exit_code == 0


def test_download_progress_does_not_run_into_the_result_lines(hub, monkeypatch):
    """huggingface_hub's progress bars redraw one line with carriage returns and
    leave the cursor mid-line; the result lines must start on a fresh line and
    each download is announced before its bar."""
    import sys

    def snapshot_download(model_id, **_):
        hub.requested.append(model_id)
        sys.stderr.write(f"\rFetching {model_id}:  50%|#####     |")
        sys.stderr.write(f"\rFetching {model_id}: 100%|##########|")  # no newline, like tqdm
        hub.downloaded.add(model_id)
        return "/cache/" + model_id

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    result = runner.invoke(app, ["pull"])

    assert result.exit_code == 0
    lines = result.output.replace("\r", "\n").splitlines()
    status_lines = [line for line in lines if " downloaded " in line and "100%" in line]
    assert status_lines == []  # a result line sharing a line with a bar fragment
    for model_id in (DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_EXPAND_MODEL):
        assert any(line.startswith("Downloading ") and model_id in line for line in lines)
        assert any(line.strip().endswith(model_id) and "downloaded" in line for line in lines)
