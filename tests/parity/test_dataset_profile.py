import pytest

from parity.dataset_profile import load_dataset_profile, resolve_active_profile_path


def _write_yaml(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_load_dataset_profile_with_qrels(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "queries.json").write_text("[]")
    (tmp_path / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n")
    config = _write_yaml(
        tmp_path,
        "profile.yaml",
        "name: test\ncorpus_dir: ./corpus\nqueries_file: ./queries.json\nqrels_file: ./qrels.tsv\n",
    )

    profile = load_dataset_profile(config)

    assert profile.name == "test"
    assert profile.corpus_dir == tmp_path / "corpus"
    assert profile.queries_file == tmp_path / "queries.json"
    assert profile.qrels_file == tmp_path / "qrels.tsv"
    assert profile.has_qrels is True


def test_load_dataset_profile_without_qrels(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "queries.yaml").write_text("- one\n- two\n")
    config = _write_yaml(
        tmp_path, "profile.yaml", "name: mine\ncorpus_dir: ./corpus\nqueries_file: ./queries.yaml\n"
    )

    profile = load_dataset_profile(config)

    assert profile.qrels_file is None
    assert profile.has_qrels is False


def test_load_dataset_profile_resolves_paths_relative_to_config_file(tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "queries.yaml").write_text("- q\n")
    config = _write_yaml(
        subdir, "profile.yaml", "name: t\ncorpus_dir: ../corpus\nqueries_file: ../queries.yaml\n"
    )

    profile = load_dataset_profile(config)

    assert profile.corpus_dir == tmp_path / "corpus"
    assert profile.corpus_dir.is_dir()


def test_load_dataset_profile_rejects_missing_required_field(tmp_path):
    config = _write_yaml(tmp_path, "profile.yaml", "name: broken\ncorpus_dir: ./corpus\n")

    with pytest.raises(ValueError, match="queries_file"):
        load_dataset_profile(config)


def test_load_dataset_profile_rejects_nonexistent_corpus_dir(tmp_path):
    (tmp_path / "queries.yaml").write_text("- q\n")
    config = _write_yaml(
        tmp_path, "profile.yaml", "name: t\ncorpus_dir: ./nope\nqueries_file: ./queries.yaml\n"
    )

    with pytest.raises(ValueError, match="corpus_dir"):
        load_dataset_profile(config)


def test_missing_corpus_dir_error_omits_prepare_hint_for_a_custom_profile(tmp_path):
    (tmp_path / "queries.yaml").write_text("- q\n")
    config = _write_yaml(
        tmp_path, "profile.yaml", "name: mine\ncorpus_dir: ./nope\nqueries_file: ./queries.yaml\n"
    )

    with pytest.raises(ValueError) as exc_info:
        load_dataset_profile(config)

    assert "prepare_scifact_corpus.py" not in str(exc_info.value)


def test_missing_corpus_dir_error_hints_at_prepare_script_for_the_builtin_profile(
    monkeypatch, tmp_path
):
    (tmp_path / "queries.yaml").write_text("- q\n")
    config = _write_yaml(
        tmp_path,
        "scifact.yaml",
        "name: scifact\ncorpus_dir: ./nope\nqueries_file: ./queries.yaml\n",
    )
    monkeypatch.setattr("parity.dataset_profile._BUILTIN_DEFAULT", config.resolve())

    with pytest.raises(ValueError, match="prepare_scifact_corpus.py"):
        load_dataset_profile(config)


def test_missing_corpus_dir_error_hints_at_prepare_script_for_conditionalqa(monkeypatch, tmp_path):
    (tmp_path / "queries.yaml").write_text("- q\n")
    config = _write_yaml(
        tmp_path,
        "conditionalqa.yaml",
        "name: conditionalqa\ncorpus_dir: ./nope\nqueries_file: ./queries.yaml\n",
    )
    monkeypatch.setattr("parity.dataset_profile._BUILTIN_DEFAULT", (tmp_path / "scifact.yaml"))

    with pytest.raises(ValueError, match="prepare_conditionalqa_corpus.py"):
        load_dataset_profile(config)


def test_resolve_active_profile_path_prefers_explicit_arg(monkeypatch, tmp_path):
    monkeypatch.setenv("PARITY_DATASET_CONFIG", str(tmp_path / "env.yaml"))
    explicit = tmp_path / "explicit.yaml"

    assert resolve_active_profile_path(str(explicit)) == explicit


def test_resolve_active_profile_path_falls_back_to_env_var(monkeypatch, tmp_path):
    env_path = tmp_path / "env.yaml"
    monkeypatch.setenv("PARITY_DATASET_CONFIG", str(env_path))

    assert resolve_active_profile_path(None) == env_path


def test_resolve_active_profile_path_defaults_to_builtin_scifact(monkeypatch):
    monkeypatch.delenv("PARITY_DATASET_CONFIG", raising=False)

    result = resolve_active_profile_path(None)

    assert result.name == "scifact.yaml"
    assert result.parent.name == "datasets"


@pytest.mark.requires_scifact_corpus
def test_builtin_scifact_profile_loads_and_has_qrels():
    path = resolve_active_profile_path(None)
    profile = load_dataset_profile(path)

    assert profile.name == "scifact"
    assert profile.has_qrels is True
    assert profile.corpus_dir.is_dir()
    assert profile.queries_file.is_file()
