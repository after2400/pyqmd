import pytest

from parity._text_normalize import TextSub, normalize, replace_paths


def test_strips_ansi_and_osc_sequences():
    text = "\x1b[32m✓\x1b[0m done\x1b]9;4;1;50\x07"
    assert normalize(text, {}) == "✓ done"


def test_replaces_placeholder_paths_longest_first():
    placeholders = {"/tmp/a": "<CORPUS_1>", "/tmp/a/b": "<CORPUS_2>"}
    assert normalize("x /tmp/a/b/c.md y /tmp/a", placeholders) == "x <CORPUS_2>/c.md y <CORPUS_1>"


def test_placeholder_matches_private_prefixed_form():
    placeholders = {"/var/folders/x/T/p1": "<CORPUS_1>"}
    assert normalize("Collection: /private/var/folders/x/T/p1 (**/*.md)", placeholders) == (
        "Collection: <CORPUS_1> (**/*.md)"
    )


def test_placeholder_matches_private_stripped_form():
    placeholders = {"/private/var/folders/x/T/p1": "<CORPUS_1>"}
    assert normalize("at /var/folders/x/T/p1/a.md", placeholders) == "at <CORPUS_1>/a.md"


@pytest.mark.parametrize(
    "raw",
    ["in 1s", "in 12s", "in 350ms", "in 1.5s", "in 2m 3s", "in 1h 4m"],
)
def test_durations_become_placeholder(raw):
    assert normalize(raw, {}) == "in <DURATION>"


def test_relative_time_becomes_ago_placeholder_not_duration():
    assert normalize("Updated:  0s ago", {}) == "Updated:  <AGO>"
    assert normalize("Updated:  3 hours ago", {}) == "Updated:  <AGO>"


def test_docids_become_placeholder():
    assert normalize("see #a1b2c3 now", {}) == "see #<DOCID> now"


def test_command_name_in_hint_text_is_normalized():
    assert normalize("Run 'qmd embed' to update", {}) == "Run '<CMD> embed' to update"
    assert normalize("Run 'pyqmd embed' to update", {}) == "Run '<CMD> embed' to update"
    assert normalize("Use 'qmd collection list'", {}) == "Use '<CMD> collection list'"


def test_qmd_uris_are_untouched():
    assert normalize("qmd://flow-a/ → qmd://flow-b/", {}) == "qmd://flow-a/ → qmd://flow-b/"


def test_trailing_whitespace_and_trailing_blank_lines_are_stripped():
    assert normalize("a  \n\nb\t\n\n\n", {}) == "a\n\nb"


def test_leading_and_interior_blank_lines_are_preserved():
    assert normalize("\nHeader\n\nbody\n", {}) == "\nHeader\n\nbody"


def test_text_subs_apply_before_whitespace_step_so_line_removal_leaves_no_stray_newline():
    sub = TextSub(r"^  Documents: \d+\n?", "", "pyqmd-only superset")
    assert normalize("Collection: x\n  Documents: 4\n", {}, [sub]) == "Collection: x"


def test_text_subs_run_in_order_and_see_normalized_text():
    subs = [
        TextSub(r"<CORPUS_1>", "<C>", "first"),
        TextSub(r"<C>/a\.md", "A", "second"),
    ]
    assert normalize("/tmp/c/a.md", {"/tmp/c": "<CORPUS_1>"}, subs) == "A"


def test_text_sub_requires_a_reason():
    with pytest.raises(ValueError):
        TextSub(r"x", "", "  ")


def test_text_sub_rejects_bad_regex():
    with pytest.raises(ValueError):
        TextSub(r"(", "", "broken")


def test_replace_paths_only_touches_paths():
    text = "\x1b[32m✓\x1b[0m /private/var/x/c in 1s\n"
    assert replace_paths(text, {"/var/x/c": "<CORPUS_1>"}) == "\x1b[32m✓\x1b[0m <CORPUS_1> in 1s\n"


def test_strips_private_mode_cursor_sequences():
    # Node's embed hides/shows the cursor on stderr: ESC[?25l ... ESC[?25h
    assert normalize("\x1b[?25l\x1b[?25h", {}) == ""


def test_carriage_returns_become_newlines():
    # Node's final embed bar is console.log("\r<bar> 100%"); both sides'
    # \r must normalize the same way, and \r\n must not double up.
    assert normalize("a\n\n\rbar 100%\r\nb", {}) == "a\n\n\nbar 100%\nb"
