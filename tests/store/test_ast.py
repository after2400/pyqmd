from pyqmd_mlx.store._ast import (
    detect_language,
    get_ast_break_points,
    get_ast_status,
    merge_break_points,
)
from pyqmd_mlx.store._types import BreakPoint


def test_detect_language_maps_known_extensions():
    assert detect_language("foo.py") == "python"
    assert detect_language("foo.ts") == "typescript"
    assert detect_language("foo.mts") == "typescript"
    assert detect_language("foo.cts") == "typescript"
    assert detect_language("foo.tsx") == "tsx"
    assert detect_language("foo.jsx") == "tsx"
    assert detect_language("foo.js") == "javascript"
    assert detect_language("foo.mjs") == "javascript"
    assert detect_language("foo.cjs") == "javascript"
    assert detect_language("foo.go") == "go"
    assert detect_language("foo.rs") == "rust"


def test_detect_language_unsupported_extension_returns_none():
    assert detect_language("foo.md") is None
    assert detect_language("foo.txt") is None
    assert detect_language("foo") is None


def test_detect_language_is_case_insensitive():
    assert detect_language("FOO.PY") == "python"


def test_get_ast_status_reports_all_six_languages():
    status = get_ast_status()
    assert {lang.language for lang in status.languages} == {
        "typescript",
        "tsx",
        "javascript",
        "python",
        "go",
        "rust",
    }


def test_get_ast_status_available_when_grammars_load():
    # The six grammar packages are regular dependencies (Task 1), so on any
    # machine that can run the test suite at all, every language loads.
    status = get_ast_status()
    assert status.available is True
    assert all(lang.available for lang in status.languages)
    assert all(lang.error is None for lang in status.languages)


def test_get_ast_status_reports_failure_for_broken_grammar(monkeypatch):
    # _GRAMMAR_CACHE/_FAILED_LANGUAGES/_GRAMMAR_LOAD_ERRORS are plain
    # module-level dict/set state, not monkeypatch-tracked attributes --
    # only monkeypatch.setitem(GRAMMAR_LOADERS, ...) auto-reverts at
    # teardown. Mutating the other three directly means this test must
    # restore them itself, or a broken "rust" entry leaks into every test
    # that runs afterward in the same process (they'd all see rust as
    # permanently failed, since _load_grammar short-circuits on
    # _FAILED_LANGUAGES before ever consulting GRAMMAR_LOADERS again).
    from pyqmd_mlx.store import _ast

    def _broken_loader():
        raise RuntimeError("grammar load boom")

    monkeypatch.setitem(_ast.GRAMMAR_LOADERS, "rust", _broken_loader)
    _ast._GRAMMAR_CACHE.pop("rust", None)
    _ast._FAILED_LANGUAGES.discard("rust")
    _ast._GRAMMAR_LOAD_ERRORS.pop("rust", None)

    try:
        status = _ast.get_ast_status()

        rust_status = next(lang for lang in status.languages if lang.language == "rust")
        assert rust_status.available is False
        assert "grammar load boom" in rust_status.error
        assert status.available is True  # the other 5 languages still loaded
    finally:
        _ast._FAILED_LANGUAGES.discard("rust")
        _ast._GRAMMAR_LOAD_ERRORS.pop("rust", None)


def test_get_ast_break_points_python_class_and_function():
    content = (
        "import os\n\n"
        "class Foo:\n"
        "    @staticmethod\n"
        "    def bar():\n"
        "        pass\n\n"
        "def baz():\n"
        "    pass\n"
    )
    points = get_ast_break_points(content, "sample.py")
    types_at_pos = {p.pos: p.type for p in points}
    scores_at_pos = {p.pos: p.score for p in points}

    assert types_at_pos[content.index("import os")] == "ast:import"
    assert types_at_pos[content.index("class Foo")] == "ast:class"
    assert scores_at_pos[content.index("class Foo")] == 100
    assert types_at_pos[content.index("@staticmethod")] == "ast:decorated"
    assert types_at_pos[content.index("def baz")] == "ast:func"
    assert scores_at_pos[content.index("def baz")] == 90


def test_get_ast_break_points_typescript_export_class():
    content = "export class Foo {\n  bar() {}\n}\n"
    points = get_ast_break_points(content, "sample.ts")
    types_at_pos = {p.pos: p.type for p in points}

    assert types_at_pos[content.index("export class")] == "ast:export"
    assert types_at_pos[content.index("class Foo")] == "ast:class"
    assert types_at_pos[content.index("bar()")] == "ast:method"


def test_get_ast_break_points_tsx_uses_tsx_grammar():
    content = "export function App() {\n  return <div>hi</div>;\n}\n"
    points = get_ast_break_points(content, "sample.tsx")
    assert any(p.type == "ast:func" for p in points)


def test_get_ast_break_points_javascript_arrow_function():
    content = "const handler = () => {\n  return 1;\n};\n"
    points = get_ast_break_points(content, "sample.js")
    assert any(p.type == "ast:func" for p in points)


def test_get_ast_break_points_go_function():
    content = 'package main\n\nfunc main() {\n\tprintln("hi")\n}\n'
    points = get_ast_break_points(content, "sample.go")
    types_at_pos = {p.pos: p.type for p in points}
    assert types_at_pos[content.index("func main")] == "ast:func"


def test_get_ast_break_points_rust_struct_and_impl():
    content = (
        "struct Point {\n    x: i32,\n}\n\nimpl Point {\n    fn new() -> Self { todo!() }\n}\n"
    )
    points = get_ast_break_points(content, "sample.rs")
    types_at_pos = {p.pos: p.type for p in points}
    assert types_at_pos[content.index("struct Point")] == "ast:struct"
    assert types_at_pos[content.index("impl Point")] == "ast:impl"


def test_get_ast_break_points_unsupported_extension_returns_empty():
    assert get_ast_break_points("# just markdown", "notes.md") == []


def test_get_ast_break_points_non_ascii_content_converts_byte_to_char_offset():
    content = "# café note é é é\ndef bar():\n    pass\n"
    points = get_ast_break_points(content, "sample.py")
    func_point = next(p for p in points if p.type == "ast:func")
    assert func_point.pos == content.index("def bar")


def test_get_ast_break_points_parse_failure_returns_empty(monkeypatch):
    from pyqmd_mlx.store import _ast

    # Fresh caches: a Parser memoized by an earlier test would otherwise
    # bypass the monkeypatched class entirely.
    _ast._clear_ast_caches()

    class _BoomParser:
        def __init__(self, *_a, **_kw):
            pass

        def parse(self, *_a, **_kw):
            raise RuntimeError("parse boom")

    monkeypatch.setattr("tree_sitter.Parser", _BoomParser)

    try:
        assert get_ast_break_points("def foo(): pass", "sample.py") == []
    finally:
        _ast._clear_ast_caches()


def test_get_ast_break_points_returns_break_point_instances():
    content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
    points = get_ast_break_points(content, "sample.py")
    assert points
    assert all(isinstance(p, BreakPoint) for p in points)
    assert [p.pos for p in points] == sorted(p.pos for p in points)


def test_merge_break_points_combines_and_sorts():
    regex_points = [
        BreakPoint(pos=5, score=20, type="blank"),
        BreakPoint(pos=50, score=1, type="newline"),
    ]
    ast_points = [BreakPoint(pos=10, score=90, type="ast:func")]

    merged = merge_break_points(regex_points, ast_points)

    assert [p.pos for p in merged] == [5, 10, 50]


def test_merge_break_points_keeps_higher_score_on_position_collision():
    regex_points = [BreakPoint(pos=10, score=20, type="blank")]
    ast_points = [BreakPoint(pos=10, score=90, type="ast:func")]

    merged = merge_break_points(regex_points, ast_points)

    assert len(merged) == 1
    assert merged[0].score == 90
    assert merged[0].type == "ast:func"


def test_merge_break_points_empty_ast_points_returns_regex_only():
    regex_points = [BreakPoint(pos=5, score=20, type="blank")]

    merged = merge_break_points(regex_points, [])

    assert merged == regex_points


def test_get_ast_status_parses_samples_through_real_path(monkeypatch):
    # The health check must exercise get_ast_break_points itself, not just
    # grammar loading + query compilation: a boom inside the parse path
    # must surface as "unavailable" rather than a false "active".
    from pyqmd_mlx.store import _ast

    _ast._clear_ast_caches()

    class _BoomParser:
        def __init__(self, *_a, **_kw):
            pass

        def parse(self, *_a, **_kw):
            raise RuntimeError("parse boom")

    monkeypatch.setattr("tree_sitter.Parser", _BoomParser)
    try:
        status = _ast.get_ast_status()
        assert status.available is False
        assert all(lang.available is False for lang in status.languages)
    finally:
        _ast._clear_ast_caches()


def test_get_ast_break_points_memoizes_by_content_hash():
    from pyqmd_mlx.store import _ast

    _ast._clear_ast_caches()
    try:
        content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        first = get_ast_break_points(content, "sample.py")
        assert first
        assert len(_ast._AST_BREAK_POINTS_CACHE) == 1
        second = get_ast_break_points(content, "sample.py")
        assert second == first
        assert len(_ast._AST_BREAK_POINTS_CACHE) == 1  # no second parse
    finally:
        _ast._clear_ast_caches()
