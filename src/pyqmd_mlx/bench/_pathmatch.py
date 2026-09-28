"""Fuzzy path matching between search results and a bench fixture's
expected_files, ported from Node's src/bench/score.ts (normalizePath/
pathsMatch). Kept separate from _metrics.py so the exact-set-membership
metric functions never need fuzzy-matching logic of their own --
canonicalize_ranked_ids does the fuzzy resolution up front and hands
plain exact string ids downstream."""


def normalize_path(p: str) -> str:
    """qmd://collection/docs/readme.md -> docs/readme.md; also lowercases
    and trims leading/trailing slashes. A bare path (no qmd:// scheme) is
    just lowercased and trimmed."""
    if p.startswith("qmd://"):
        without_scheme = p[len("qmd://") :]
        slash_idx = without_scheme.find("/")
        p = without_scheme[slash_idx + 1 :] if slash_idx >= 0 else without_scheme
    return p.lower().strip("/")


def paths_match(result: str, expected: str) -> bool:
    """Exact match after normalization, or either string a suffix of the
    other *at a path-segment boundary* -- lets a fixture author write a
    bare filename ("readme.md") and match a deeper indexed path
    ("docs/subdir/readme.md"), without also matching a different file
    whose name merely ends with it ("docs/myreadme.md"). Node's pathsMatch
    has no boundary check; see the bench design spec's path-matching
    section for this intentional divergence."""
    nr = normalize_path(result)
    ne = normalize_path(expected)
    if nr == ne:
        return True
    return nr.endswith("/" + ne) or ne.endswith("/" + nr)


def canonicalize_ranked_ids(ranked_paths: list[str], expected_files: list[str]) -> list[str]:
    """Rewrite each ranked result path to the exact expected_files string
    it matches (paths_match semantics), so callers can score with plain
    exact-set-membership afterward. Each expected_files entry is claimed
    by at most one ranked path -- the highest-ranked one that matches it,
    preferring an exact match over a suffix match -- so a bare-filename
    expectation matching several indexed files is credited once. A path
    matching no unclaimed expected_files entry is left as its own
    normalized form -- it simply won't be a member of the caller's
    relevant_ids set (and if that form happens to equal an already-claimed
    id, the metrics themselves count each relevant id only once)."""
    claimed: set[str] = set()
    canonical = []
    for path in ranked_paths:
        normalized = normalize_path(path)
        candidates = [e for e in expected_files if e not in claimed and paths_match(path, e)]
        match = next(
            (e for e in candidates if normalize_path(e) == normalized),
            candidates[0] if candidates else None,
        )
        if match is None:
            canonical.append(normalized)
        else:
            claimed.add(match)
            canonical.append(match)
    return canonical
