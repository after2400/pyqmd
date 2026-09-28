from pyqmd_mlx.bench._pathmatch import canonicalize_ranked_ids, normalize_path, paths_match


def test_normalize_path_strips_scheme_and_collection():
    assert normalize_path("qmd://notes/docs/readme.md") == "docs/readme.md"


def test_normalize_path_lowercases_and_trims_slashes():
    assert normalize_path("/Docs/README.md/") == "docs/readme.md"


def test_normalize_path_leaves_bare_path_alone_besides_lowercasing():
    assert normalize_path("Docs/Readme.md") == "docs/readme.md"


def test_paths_match_exact_after_normalization():
    assert paths_match("qmd://notes/docs/readme.md", "docs/readme.md") is True


def test_paths_match_suffix_either_direction():
    assert paths_match("qmd://notes/docs/subdir/readme.md", "readme.md") is True
    # Either direction: the shorter path is a suffix of the longer one.
    assert paths_match("readme.md", "qmd://notes/docs/subdir/readme.md") is True


def test_paths_match_suffix_requires_a_path_segment_boundary():
    # Real-world false positive (2026-09-25): a
    # different file whose name merely ends with the expected filename
    # must not match -- only a suffix starting at a "/" counts.
    decoy = "qmd://notes/projects-0a1b2c3d-areas-alpha-notes-md.md"
    assert paths_match(decoy, "areas-alpha-notes-md.md") is False
    assert paths_match("areas-alpha-notes-md.md", decoy) is False
    assert paths_match("qmd://notes/docs/myreadme.md", "readme.md") is False


def test_paths_match_different_directories_do_not_match():
    assert paths_match("qmd://notes/docs/readme.md", "other/readme.md") is False


def test_paths_match_no_match():
    assert paths_match("qmd://notes/docs/readme.md", "other.md") is False


def test_canonicalize_ranked_ids_rewrites_matches_to_expected_string():
    result = canonicalize_ranked_ids(
        ["qmd://notes/docs/subdir/readme.md", "qmd://notes/unrelated.md"],
        ["readme.md"],
    )
    assert result == ["readme.md", "unrelated.md"]


def test_canonicalize_ranked_ids_credits_each_expected_file_only_once():
    # A bare-filename expectation can legitimately suffix-match several
    # indexed files; only the first (highest-ranked) one gets the credit.
    result = canonicalize_ranked_ids(
        ["qmd://notes/docs/readme.md", "qmd://notes/other/readme.md"],
        ["readme.md"],
    )
    assert result == ["readme.md", "other/readme.md"]


def test_canonicalize_ranked_ids_prefers_an_exact_match_over_a_suffix_match():
    # "docs/readme.md" suffix-matches both expected entries; it must claim
    # its exact twin, leaving the bare "readme.md" for the next result.
    result = canonicalize_ranked_ids(
        ["qmd://notes/docs/readme.md", "qmd://notes/other/readme.md"],
        ["readme.md", "docs/readme.md"],
    )
    assert result == ["docs/readme.md", "readme.md"]
