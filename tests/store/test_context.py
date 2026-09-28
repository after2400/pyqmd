from pyqmd_mlx.store import Store


def test_add_context_sets_on_empty_map():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")

    assert store.add_context("notes", "", "Root context") is True

    assert store.get_collection("notes")["context"] == '{"": "Root context"}'
    store.close()


def test_add_context_overwrites_existing_prefix():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "People", "first")

    store.add_context("notes", "People", "second")

    assert store.list_all_contexts() == [
        {"collection": "notes", "path": "People", "context": "second"}
    ]
    store.close()


def test_add_context_unknown_collection_returns_false():
    store = Store(":memory:")
    assert store.add_context("nope", "", "text") is False
    store.close()


def test_remove_context_last_entry_clears_column():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")

    assert store.remove_context("notes", "") is True

    assert store.get_collection("notes")["context"] is None
    store.close()


def test_remove_context_one_of_several_keeps_the_rest():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root")
    store.add_context("notes", "People", "About people")

    assert store.remove_context("notes", "People") is True

    assert store.list_all_contexts() == [{"collection": "notes", "path": "", "context": "Root"}]
    store.close()


def test_remove_context_unknown_collection_returns_false():
    store = Store(":memory:")
    assert store.remove_context("nope", "") is False
    store.close()


def test_remove_context_unset_prefix_on_real_collection_returns_false():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    assert store.remove_context("notes", "People") is False
    store.close()


def test_list_all_contexts_empty_index_returns_empty_list():
    store = Store(":memory:")
    assert store.list_all_contexts() == []
    store.close()


def test_list_all_contexts_groups_and_orders_by_collection_name():
    store = Store(":memory:")
    store.add_collection("zeta", "/zeta")
    store.add_collection("alpha", "/alpha")
    store.add_context("zeta", "", "Zeta root")
    store.add_context("alpha", "", "Alpha root")
    store.add_context("alpha", "Sub", "Alpha sub")

    result = store.list_all_contexts()

    assert result == [
        {"collection": "alpha", "path": "", "context": "Alpha root"},
        {"collection": "alpha", "path": "Sub", "context": "Alpha sub"},
        {"collection": "zeta", "path": "", "context": "Zeta root"},
    ]
    store.close()


def test_list_all_contexts_omits_collection_with_no_context_set():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    assert store.list_all_contexts() == []
    store.close()


def test_get_context_for_path_no_match_returns_none():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    assert store.get_context_for_path("notes", "People/wife.md") is None
    store.close()


def test_get_context_for_path_single_root_match():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")
    assert store.get_context_for_path("notes", "People/wife.md") == "Root context"
    store.close()


def test_get_context_for_path_joins_general_to_specific():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")
    store.add_context("notes", "People", "People context")

    result = store.get_context_for_path("notes", "People/wife.md")

    assert result == "Root context\n\nPeople context"
    store.close()


def test_get_context_for_path_excludes_non_ancestor_prefix():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "Topics", "Topics context")
    assert store.get_context_for_path("notes", "People/wife.md") is None
    store.close()


def test_get_context_for_path_unknown_collection_returns_none():
    store = Store(":memory:")
    assert store.get_context_for_path("nope", "a.md") is None
    store.close()


def test_get_context_for_path_tolerates_legacy_non_json_context_value():
    """add_collection's own context param is a raw-string passthrough
    (pre-existing, unrelated feature) -- a value written that way must not
    crash resolution."""
    store = Store(":memory:")
    store.add_collection("notes", "/notes", context="a legacy raw string")
    assert store.get_context_for_path("notes", "a.md") is None
    store.close()


def test_get_context_for_file_qmd_uri_form():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")
    assert store.get_context_for_file("qmd://notes/a.md") == "Root context"
    store.close()


def test_get_context_for_file_bare_collection_slash_path_form():
    store = Store(":memory:")
    store.add_collection("notes", "/notes")
    store.add_context("notes", "", "Root context")
    assert store.get_context_for_file("notes/a.md") == "Root context"
    store.close()


def test_get_context_for_file_unknown_collection_returns_none():
    store = Store(":memory:")
    assert store.get_context_for_file("qmd://nope/a.md") is None
    store.close()


def test_detect_collection_for_path_exact_root(tmp_path):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    assert store.detect_collection_for_path(str(tmp_path)) == ("notes", "")
    store.close()


def test_detect_collection_for_path_nested_subdirectory(tmp_path):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    nested = tmp_path / "People" / "wife.md"
    assert store.detect_collection_for_path(str(nested)) == ("notes", "People/wife.md")
    store.close()


def test_detect_collection_for_path_unrelated_path_returns_none(tmp_path):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    other = tmp_path.parent / "elsewhere" / "a.md"
    assert store.detect_collection_for_path(str(other)) is None
    store.close()


def test_detect_collection_for_path_trailing_slash_on_stored_path(tmp_path):
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path) + "/")
    nested = tmp_path / "a.md"
    assert store.detect_collection_for_path(str(nested)) == ("notes", "a.md")
    store.close()
