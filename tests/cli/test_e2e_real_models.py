"""One deliberate real-model end-to-end test: collection add -> embed ->
query, using actual pyqmd_mlx.llm models (no fakes), through the real Typer app.
"""

import json

import pytest
from huggingface_hub.utils import disable_progress_bars
from typer.testing import CliRunner

from pyqmd_mlx.cli.app import app
from pyqmd_mlx.store import Store

runner = CliRunner()

# Hugging Face download progress bars write to stderr, but CliRunner's
# capture merges stdout/stderr -- an uncached model download would otherwise
# interleave progress-bar text into the `--format json` output asserted on
# below. Disabling them is a test-harness/output-capture concern, not a
# change to model behavior or a mock of any kind. Must be called (not just
# the env var set) because huggingface_hub reads the env var once at import
# time, and it may already be imported transitively before this test runs.
disable_progress_bars()


@pytest.mark.slow
def test_pyqmd_add_embed_query_end_to_end_with_real_models(tmp_path, monkeypatch):
    (tmp_path / "auth.md").write_text(
        "# Authentication\nHow to configure authentication for your application "
        "using API keys and OAuth tokens."
    )
    (tmp_path / "cooking.md").write_text(
        "# Cooking\nA recipe for making fresh pasta from scratch with flour and eggs."
    )

    store = Store(":memory:")
    for module_name in ("collection", "documents", "search", "embed", "status"):
        monkeypatch.setattr(
            f"pyqmd_mlx.cli.commands.{module_name}.get_store", lambda db_path=None, s=store: s
        )

    add_result = runner.invoke(app, ["collection", "add", str(tmp_path), "--name", "docs"])
    assert add_result.exit_code == 0, add_result.output

    embed_result = runner.invoke(app, ["embed"])
    assert embed_result.exit_code == 0, embed_result.output

    query_result = runner.invoke(
        app, ["query", "how do I set up authentication", "--format", "json"]
    )
    assert query_result.exit_code == 0, query_result.output
    parsed = json.loads(query_result.output)
    assert len(parsed) >= 1
    assert parsed[0]["title"] == "Authentication"
