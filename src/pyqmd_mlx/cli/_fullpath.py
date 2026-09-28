"""--full-path resolution shared by get/multi-get/search/vsearch/query.
Node's own outputResults()/multiGet() compute this once per invocation,
then every format branch consults the result -- not re-resolved per
format. render_full_path mirrors Node's renderFullPath (qmd.ts:1228-1238);
apply_full_path mirrors the resolutions-map-then-consult pattern,
generalized across DisplayResult and DocumentEntry/MultiGetEntry since
both already expose mutable display_path/docid fields."""

import os


def render_full_path(fs_path: str) -> str:
    cwd = os.path.realpath(os.getcwd())
    if fs_path == cwd:
        return "./"
    if fs_path.startswith(cwd + os.sep):
        return "./" + fs_path[len(cwd) + 1 :]
    return fs_path


def apply_full_path(rows, store) -> int:
    """Mutates each row's display_path/docid in place when its qmd://
    path resolves to a real on-disk file; leaves it untouched (still
    addressable via its original docid) otherwise. Returns the number of
    rows that could not be resolved, for the caller to warn about."""
    unresolved = 0
    for row in rows:
        without_scheme = row.display_path.removeprefix("qmd://")
        collection, _, path = without_scheme.partition("/")
        resolved = store.resolve_full_path(collection, path) if path else None
        if resolved is None:
            unresolved += 1
            continue
        row.display_path = render_full_path(resolved)
        row.docid = None
    return unresolved


def format_fullpath_warning(unresolved: int) -> str:
    noun = "document" if unresolved == 1 else "documents"
    return (
        f"Warning: {unresolved} {noun} could not be resolved to a full "
        "path (file may have moved); showing qmd:// links instead."
    )
