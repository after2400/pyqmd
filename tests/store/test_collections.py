from pyqmd_mlx.llm import DEFAULT_EMBED_MODEL
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import scan_and_register_collection
from pyqmd_mlx.store.store import CollectionRemoval


def test_add_and_get_collection():
    store = Store(":memory:")
    store.add_collection("notes", "/home/user/notes")
    collection = store.get_collection("notes")
    assert collection["name"] == "notes"
    assert collection["path"] == "/home/user/notes"
    assert collection["pattern"] == "**/*.md"
    assert collection["include_by_default"] == 1
    store.close()


def test_get_collection_returns_none_when_missing():
    store = Store(":memory:")
    assert store.get_collection("nope") is None
    store.close()


def test_add_collection_with_custom_pattern_and_context():
    store = Store(":memory:")
    store.add_collection("code", "/repo", pattern="**/*.py", context="Python source files")
    collection = store.get_collection("code")
    assert collection["pattern"] == "**/*.py"
    assert collection["context"] == "Python source files"
    store.close()


def test_list_collections_returns_all():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    names = {c["name"] for c in store.list_collections()}
    assert names == {"a", "b"}
    store.close()


def test_remove_collection():
    store = Store(":memory:")
    store.add_collection("temp", "/tmp/x")
    assert store.remove_collection("temp") is True
    assert store.get_collection("temp") is None
    store.close()


def test_remove_collection_returns_false_when_missing():
    store = Store(":memory:")
    assert store.remove_collection("nope") is False
    store.close()


def test_rename_collection():
    store = Store(":memory:")
    store.add_collection("old_name", "/x")
    store.rename_collection("old_name", "new_name")
    assert store.get_collection("old_name") is None
    assert store.get_collection("new_name")["path"] == "/x"
    store.close()


def test_add_collection_duplicate_name_raises():
    import sqlite3

    store = Store(":memory:")
    store.add_collection("dup", "/a")
    try:
        store.add_collection("dup", "/b")
        raise AssertionError("expected sqlite3.IntegrityError")
    except sqlite3.IntegrityError:
        pass
    store.close()


def test_set_collection_update_command_sets_and_clears():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")

    store.set_collection_update_command("notes", "git pull")
    assert store.get_collection("notes")["update_command"] == "git pull"

    store.set_collection_update_command("notes", None)
    assert store.get_collection("notes")["update_command"] is None
    store.close()


def test_set_collection_update_command_overwrites_existing():
    store = Store(":memory:")
    store.add_collection("notes", "/notes", update_command="old cmd")

    store.set_collection_update_command("notes", "new cmd")

    assert store.get_collection("notes")["update_command"] == "new cmd"
    store.close()


def test_set_collection_include_by_default_excludes_and_reincludes():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")

    store.set_collection_include_by_default("notes", False)
    assert store.get_collection("notes")["include_by_default"] == 0

    store.set_collection_include_by_default("notes", True)
    assert store.get_collection("notes")["include_by_default"] == 1
    store.close()


def test_set_collection_include_by_default_is_idempotent():
    store = Store(":memory:")
    store.add_collection("notes", "/notes", include_by_default=False)

    store.set_collection_include_by_default("notes", False)

    assert store.get_collection("notes")["include_by_default"] == 0
    store.close()


def test_get_default_collection_names_returns_none_when_nothing_excluded():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b")
    assert store.get_default_collection_names() is None
    store.close()


def test_get_default_collection_names_returns_included_subset():
    store = Store(":memory:")
    store.add_collection("a", "/a")
    store.add_collection("b", "/b", include_by_default=False)
    assert store.get_default_collection_names() == ["a"]
    store.close()


def test_resolve_full_path_returns_realpath_for_existing_file(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))

    resolved = store.resolve_full_path("notes", "a.md")

    assert resolved == str((tmp_path / "a.md").resolve())
    store.close()


def test_resolve_full_path_returns_none_for_missing_file(tmp_path):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))

    assert store.resolve_full_path("notes", "does-not-exist.md") is None
    store.close()


def test_resolve_full_path_returns_none_for_path_escaping_collection_root(tmp_path):
    outside = tmp_path.parent / "outside.md"
    outside.write_text("secret")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))

    assert store.resolve_full_path("notes", "../outside.md") is None
    store.close()
    outside.unlink()


def test_resolve_full_path_returns_none_for_unknown_collection():
    store = Store(":memory:")
    assert store.resolve_full_path("nope", "a.md") is None
    store.close()


def test_remove_collection_detailed_reports_deleted_docs_and_cleaned_hashes(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha")
    (tmp_path / "b.md").write_text("# B\nbravo")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    assert store.remove_collection_detailed("notes") == CollectionRemoval(
        deleted_docs=2, cleaned_hashes=2
    )
    assert store.get_collection("notes") is None


def test_remove_collection_detailed_keeps_hashes_shared_with_another_collection(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    for d in (a_dir, b_dir):
        d.mkdir()
        (d / "same.md").write_text("# Same\nidentical content")
    store = Store(":memory:")
    for name, d in (("a", a_dir), ("b", b_dir)):
        store.add_collection(name, str(d))
        scan_and_register_collection(store, str(d), "**/*.md", name)

    removal = store.remove_collection_detailed("a")
    assert removal.deleted_docs == 1
    assert removal.cleaned_hashes == 0


def test_remove_collection_detailed_returns_none_for_missing_collection():
    assert Store(":memory:").remove_collection_detailed("nope") is None


def test_embed_model_property_exposes_default_model():
    assert Store(":memory:").embed_model == DEFAULT_EMBED_MODEL
