import typer
from typer.testing import CliRunner

from pyqmd_mlx.cli._index_report import echo_embed_hint, echo_index_summary
from pyqmd_mlx.store import Store
from pyqmd_mlx.store._indexing import ReindexResult, scan_and_register_collection

runner = CliRunner()


def _invoke(fn):
    app = typer.Typer()
    app.command()(fn)
    return runner.invoke(app, [])


def test_echo_index_summary_uses_node_wording_with_leading_blank_line():
    result = ReindexResult(indexed=4, updated=1, unchanged=2, removed=3)

    out = _invoke(lambda: echo_index_summary(result))

    assert out.stdout == "\nIndexed: 4 new, 1 updated, 2 unchanged, 3 removed\n"


def test_echo_index_summary_prints_orphan_line_when_nonzero():
    result = ReindexResult(indexed=0, updated=0, unchanged=0, removed=1, orphaned_cleaned=2)

    out = _invoke(lambda: echo_index_summary(result))

    assert out.stdout.endswith("Cleaned up 2 orphaned content hash(es)\n")


def test_echo_embed_hint_uses_node_wording(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha")
    store = Store(":memory:")
    store.add_collection("notes", str(tmp_path))
    scan_and_register_collection(store, str(tmp_path), "**/*.md", "notes")

    out = _invoke(lambda: echo_embed_hint(store))

    assert out.stdout == (
        "\nRun 'pyqmd embed' to update embeddings (1 unique hashes need vectors)\n"
    )


def test_echo_embed_hint_silent_when_nothing_pending():
    out = _invoke(lambda: echo_embed_hint(Store(":memory:")))
    assert out.stdout == ""
