"""Ordered, stateful CLI scenarios ("flows") for the mutating commands
the single-independent-command CliScenario model in cli_scenarios.py
can't express: collection add/remove/rename/update-cmd, context add/
list/remove, update, embed. Each flow runs sequentially against one
fresh, isolated pyqmd Store (or, on the Node side, one fresh isolated
index via capture_node_snapshots.py's _fresh_isolated_index) plus a
small synthetic fixture corpus (parity/fixtures/mutating/) distinct
from the shared scifact profile -- see docs/specs/
2026-09-17-parity-suite-mutating-commands-design.md.

build_cli_flow_scenarios() and run_pyqmd_flow() are the two entry
points: the former is called by both capture_node_snapshots.py (Node
side) and parity/test_structural.py (pyqmd side, via run_pyqmd_flow) so
both sides run the exact same step sequence.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from typer.testing import CliRunner

from parity._text_normalize import TextSub
from parity.scenarios.cli_scenarios import _collection_list_shape, _collection_show_shape
from pyqmd_mlx.cli.commands import collection as collection_cmd
from pyqmd_mlx.cli.commands import context as context_cmd
from pyqmd_mlx.cli.commands import embed as embed_cmd
from pyqmd_mlx.cli.commands import update as update_cmd
from pyqmd_mlx.store import Store

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "mutating"

_runner = CliRunner()


@dataclass
class CliFlowStep:
    name: str
    args: list[str]
    extract: Callable[[str, int], object]
    expect_failure: bool = False
    # Override when Node's argv genuinely differs from pyqmd's -- e.g.
    # `collection remove` needs --yes on pyqmd (to skip its confirmation
    # prompt) but Node's `collection remove` has no such flag or prompt.
    node_args: list[str] | None = None
    # Runs (with no arguments -- already bound to whatever it needs at
    # scenario-build time) immediately before this step's CLI invocation.
    # Used by update_lifecycle to mutate the fixture corpus on disk
    # between the initial `collection add` and the `update` step.
    before: Callable[[], None] | None = None
    # Declared, deliberate Node/pyqmd output-text differences for this
    # step (see parity/_text_normalize.py and the 2026-09-24 output-text
    # parity spec). Applied to both sides before comparing.
    text_subs: list[TextSub] = field(default_factory=list)
    # When set, test_cli_flow_step_text_matches_node skips this step's
    # text comparison with this reason. Use only when no narrower
    # TextSub can express the difference.
    text_skip_reason: str | None = None


@dataclass
class CliFlowScenario:
    name: str
    steps: list[CliFlowStep]
    # Temp fixture-corpus dirs this flow created, in creation order --
    # random per run, so text comparison maps them to <CORPUS_n>.
    corpus_dirs: list[Path] = field(default_factory=list)

    def placeholders(self) -> dict[str, str]:
        return {str(d): f"<CORPUS_{i}>" for i, d in enumerate(self.corpus_dirs, start=1)}


@dataclass
class FlowStepResult:
    extracted: object
    stdout: str
    stderr: str
    exit_code: int


def _new_fixture_corpus() -> Path:
    """A fresh temp directory containing a copy of every file in
    parity/fixtures/mutating/. Called once per flow that needs a corpus
    (twice for collection_lifecycle, which needs two distinct
    collections) at build_cli_flow_scenarios() call time -- not shared
    across flows, since update_lifecycle mutates its copy on disk."""
    dest = Path(tempfile.mkdtemp(prefix="pyqmd-parity-flow-"))
    for fixture_file in _FIXTURE_DIR.glob("*.md"):
        shutil.copy(fixture_file, dest / fixture_file.name)
    return dest


def _ok_shape(stdout: str, exit_code: int) -> dict:
    """For pure confirmation commands (add/rename/remove/update-cmd,
    context add/remove): a flow's very next step is always a list/show
    that verifies the mutation actually landed, so the confirmation step
    itself only needs to assert success/failure, not parse wording."""
    return {"exit_code": exit_code}


_NODE_COUNTS_RE = re.compile(
    r"Indexed:\s*(\d+)\s*new,\s*(\d+)\s*updated,\s*(\d+)\s*unchanged,\s*(\d+)\s*removed",
    re.IGNORECASE,
)


def _update_counts_shape(stdout: str, exit_code: int) -> dict:
    """Parses 'Indexed: N new, N updated, N unchanged, N removed', which
    both Node (qmd.ts: collection add's indexFiles and update's own) and
    pyqmd (_index_report.echo_index_summary, used by both commands) print.
    Exact counts are a legitimate structural check here (unlike
    cli_scenarios.py's _status_shape, deliberately presence-only for the
    much larger scifact corpus) because this fixture's file set is fully
    controlled by the flow itself."""
    if exit_code != 0:
        return {"exit_code": exit_code}
    match = _NODE_COUNTS_RE.search(stdout)
    if not match:
        return {
            "exit_code": exit_code,
            "indexed": None,
            "updated": None,
            "unchanged": None,
            "removed": None,
        }
    indexed, updated, unchanged, removed = (int(g) for g in match.groups())
    return {
        "exit_code": exit_code,
        "indexed": indexed,
        "updated": updated,
        "unchanged": unchanged,
        "removed": removed,
    }


def _build_collection_lifecycle() -> CliFlowScenario:
    corpus_a = _new_fixture_corpus()
    corpus_b = _new_fixture_corpus()
    return CliFlowScenario(
        "collection_lifecycle",
        [
            CliFlowStep(
                "add_flow_a",
                ["collection", "add", str(corpus_a), "--name", "flow-a"],
                _update_counts_shape,
            ),
            CliFlowStep(
                "add_flow_a_duplicate",
                ["collection", "add", str(corpus_a), "--name", "flow-a"],
                _ok_shape,
                expect_failure=True,
            ),
            CliFlowStep(
                "add_flow_dummy",
                ["collection", "add", str(corpus_b), "--name", "flow-dummy"],
                _update_counts_shape,
            ),
            CliFlowStep("list_after_adds", ["collection", "list"], _collection_list_shape),
            CliFlowStep(
                "set_update_cmd", ["collection", "update-cmd", "flow-a", "echo", "hi"], _ok_shape
            ),
            CliFlowStep(
                "rename_collision",
                ["collection", "rename", "flow-a", "flow-dummy"],
                _ok_shape,
                expect_failure=True,
            ),
            CliFlowStep("rename_real", ["collection", "rename", "flow-a", "flow-b"], _ok_shape),
            CliFlowStep(
                "show_flow_b",
                ["collection", "show", "flow-b"],
                lambda stdout, exit_code: _collection_show_shape(stdout, exit_code, "flow-b"),
                text_subs=[
                    TextSub(
                        r"^  Documents: \d+\n?",
                        "",
                        "pyqmd-only superset: collection show also prints the document count",
                    )
                ],
            ),
            CliFlowStep("exclude_flow_b", ["collection", "exclude", "flow-b"], _ok_shape),
            CliFlowStep("list_shows_excluded_tag", ["collection", "list"], _collection_list_shape),
            CliFlowStep("include_flow_b", ["collection", "include", "flow-b"], _ok_shape),
            CliFlowStep(
                "remove_flow_b",
                ["collection", "remove", "flow-b", "--yes"],
                _ok_shape,
                node_args=["collection", "remove", "flow-b"],
                text_subs=[
                    TextSub(
                        r"^  Deleted \d+ documents$",
                        "  Deleted <N> documents",
                        "unavoidable output difference: Node prints bun:sqlite's `changes` "
                        "for the DELETE, which counts trigger-cascaded FTS/metadata rows "
                        "(24 for this 4-document collection); pyqmd prints the real count",
                    )
                ],
            ),
            CliFlowStep(
                "remove_flow_b_again",
                ["collection", "remove", "flow-b", "--yes"],
                _ok_shape,
                expect_failure=True,
                node_args=["collection", "remove", "flow-b"],
            ),
            CliFlowStep("list_final", ["collection", "list"], _collection_list_shape),
        ],
        corpus_dirs=[corpus_a, corpus_b],
    )


def _context_list_shape(stdout: str, exit_code: int) -> dict:
    """Node's `context list` prints a 'Configured Contexts' header (plus
    a leading blank line) pyqmd's doesn't -- skipped the same way
    cli_scenarios.py's _ls_shape/_collection_list_shape skip Node-only
    headers. Both sides print one unindented collection-name line, then
    an indented path line, then a more-indented context-text line, per
    entry -- parsed structurally by indentation rather than by exact
    wording."""
    if exit_code != 0:
        return {"exit_code": exit_code, "entries": []}
    entries = []
    current_collection = None
    pending_path = None
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        text = raw_line.strip()
        if indent == 0:
            if text.lower() in ("configured contexts", "no contexts configured."):
                continue
            current_collection = text
            pending_path = None
        elif pending_path is None:
            pending_path = text
        else:
            entries.append(
                {"collection": current_collection, "path": pending_path, "context": text}
            )
            pending_path = None
    return {
        "exit_code": exit_code,
        "entries": sorted(entries, key=lambda e: (e["collection"], e["path"])),
    }


def _build_context_lifecycle() -> CliFlowScenario:
    corpus = _new_fixture_corpus()
    return CliFlowScenario(
        "context_lifecycle",
        [
            CliFlowStep(
                "add_collection",
                ["collection", "add", str(corpus), "--name", "flow-ctx"],
                _update_counts_shape,
            ),
            CliFlowStep(
                "context_add_root",
                ["context", "add", "qmd://flow-ctx/", "root context"],
                _ok_shape,
            ),
            CliFlowStep(
                "context_add_sub",
                ["context", "add", "qmd://flow-ctx/sub.md", "file context"],
                _ok_shape,
            ),
            CliFlowStep("context_list_after_adds", ["context", "list"], _context_list_shape),
            CliFlowStep(
                "context_add_bad_path",
                ["context", "add", "./not-a-real-path", "x"],
                _ok_shape,
                expect_failure=True,
            ),
            CliFlowStep("context_remove_root", ["context", "remove", "qmd://flow-ctx/"], _ok_shape),
            CliFlowStep(
                "context_remove_root_again",
                ["context", "remove", "qmd://flow-ctx/"],
                _ok_shape,
                expect_failure=True,
            ),
            CliFlowStep("context_list_final", ["context", "list"], _context_list_shape),
        ],
        corpus_dirs=[corpus],
    )


def _apply_update_mutation(corpus_dir: Path) -> None:
    """Mutates a fixture-corpus copy on disk between update_lifecycle's
    `collection add` and `update` steps: edits b.md (-> Updated), adds a
    new e.md (-> Indexed), deletes d.md (-> Removed). Leaves a.md and
    c.md untouched (-> Unchanged: 2), so `update`'s resulting counts are
    fully predictable from this fixture's known 4-file starting set."""
    (corpus_dir / "b.md").write_text(
        "# Bravo\n\nBravo fixture document, edited during update_lifecycle.\n"
    )
    (corpus_dir / "e.md").write_text(
        "# Echo\n\nEcho fixture document added on disk during update_lifecycle.\n"
    )
    (corpus_dir / "d.md").unlink()


def _build_update_lifecycle() -> CliFlowScenario:
    corpus = _new_fixture_corpus()
    return CliFlowScenario(
        "update_lifecycle",
        [
            CliFlowStep(
                "add_collection",
                ["collection", "add", str(corpus), "--name", "flow-update"],
                _update_counts_shape,
            ),
            CliFlowStep(
                "update_after_mutation",
                ["update"],
                _update_counts_shape,
                before=lambda: _apply_update_mutation(corpus),
                text_subs=[
                    TextSub(
                        r"^Cleaned up \d+ orphaned content hash\(es\)$",
                        "Cleaned up <N> orphaned content hash(es)",
                        "unavoidable output difference (a real behavior difference, recorded "
                        "in COMMAND_STATUS.md, not fixed here): Node's cleanupOrphanedContent "
                        "keeps content still referenced by inactive (soft-deleted) documents, "
                        "pyqmd's drops content no active document references -- so the removed "
                        "d.md's hash is cleaned by pyqmd now (2) but by Node only at cleanup (1)",
                    )
                ],
            ),
        ],
        corpus_dirs=[corpus],
    )


_NODE_EMBED_RE = re.compile(
    r"Embedded\s+(\d+)\s+chunks?\s+from\s+(\d+)\s+documents?", re.IGNORECASE
)


def _embed_counts_shape(stdout: str, exit_code: int) -> dict:
    """Parses 'Embedded N chunks from M documents' from the '✓ Done!' line
    both Node and pyqmd print. The no-op message both print ('✓ All
    content hashes already have embeddings.') doesn't contain it, so a
    no-op run extracts to {chunks_embedded: None, docs_processed: None}
    without needing a separate no-op-detection branch."""
    if exit_code != 0:
        return {"exit_code": exit_code}
    match = _NODE_EMBED_RE.search(stdout)
    if not match:
        return {"exit_code": exit_code, "chunks_embedded": None, "docs_processed": None}
    chunks, docs = (int(g) for g in match.groups())
    return {"exit_code": exit_code, "chunks_embedded": chunks, "docs_processed": docs}


_EMBED_TEXT_SUBS = [
    TextSub(
        r"^Model: .*$",
        "Model: <MODEL>",
        "Node-only machinery: Node embeds with its GGUF model, pyqmd with its MLX model",
    ),
]


def _build_embed_lifecycle() -> CliFlowScenario:
    corpus = _new_fixture_corpus()
    return CliFlowScenario(
        "embed_lifecycle",
        [
            CliFlowStep(
                "add_collection",
                ["collection", "add", str(corpus), "--name", "flow-embed"],
                _update_counts_shape,
            ),
            CliFlowStep("embed_first", ["embed"], _embed_counts_shape, text_subs=_EMBED_TEXT_SUBS),
            CliFlowStep("embed_noop", ["embed"], _embed_counts_shape),
            CliFlowStep(
                "embed_force", ["embed", "--force"], _embed_counts_shape, text_subs=_EMBED_TEXT_SUBS
            ),
            CliFlowStep(
                "embed_bad_collection",
                ["embed", "-c", "does-not-exist"],
                _ok_shape,
                expect_failure=True,
            ),
        ],
        corpus_dirs=[corpus],
    )


_AST_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ast_chunking"


def _new_ast_fixture_corpus() -> Path:
    """A fresh temp directory containing a copy of the AST-chunking
    fixture. Separate from _new_fixture_corpus (which globs *.md from
    _FIXTURE_DIR) because this scenario needs real code content, not
    markdown."""
    dest = Path(tempfile.mkdtemp(prefix="pyqmd-parity-ast-"))
    for fixture_file in _AST_FIXTURE_DIR.glob("*.py"):
        shutil.copy(fixture_file, dest / fixture_file.name)
    return dest


def _build_ast_chunking_embed() -> CliFlowScenario:
    corpus = _new_ast_fixture_corpus()
    return CliFlowScenario(
        "ast_chunking_embed",
        [
            CliFlowStep(
                "add_collection",
                ["collection", "add", str(corpus), "--name", "flow-ast-chunking", "--mask", "*.py"],
                _update_counts_shape,
            ),
            CliFlowStep(
                "embed_auto",
                ["embed", "--chunk-strategy", "auto"],
                _embed_counts_shape,
                text_subs=_EMBED_TEXT_SUBS,
            ),
        ],
        corpus_dirs=[corpus],
    )


def build_cli_flow_scenarios() -> list[CliFlowScenario]:
    return [
        _build_collection_lifecycle(),
        _build_context_lifecycle(),
        _build_update_lifecycle(),
        _build_embed_lifecycle(),
        _build_ast_chunking_embed(),
    ]


# command -> (Typer app, module path to patch get_store on, strip args[0]
# before invoking). "collection" is mounted into the top-level app with
# name="collection" (pyqmd_mlx/cli/app.py), so the "collection" prefix in a
# step's args is added by the PARENT app at mount time -- collection_cmd.app
# itself registers "add"/"list"/"show"/"remove"/"rename"/"update-cmd", not
# "collection". Same reasoning applies to "context" once Task 4 adds it.
_FLOW_APP_FOR_COMMAND: dict[str, tuple[object, str, bool]] = {
    "collection": (collection_cmd.app, "pyqmd_mlx.cli.commands.collection", True),
    "context": (context_cmd.app, "pyqmd_mlx.cli.commands.context", True),
    "update": (update_cmd.app, "pyqmd_mlx.cli.commands.update", False),
    "embed": (embed_cmd.app, "pyqmd_mlx.cli.commands.embed", False),
}


def run_pyqmd_flow_detailed(
    scenario: CliFlowScenario, store: Store, monkeypatch
) -> dict[str, FlowStepResult]:
    """Runs every step of `scenario` in order against `store`, via the
    real Typer CLI apps (not direct Store calls) -- exercising exactly
    the code path a user's terminal would. Returns each step's extracted
    shape plus its raw stdout/stderr/exit code. `monkeypatch` is a pytest
    MonkeyPatch (the fixture, or pytest.MonkeyPatch.context())."""
    for _app, module_path, _strip in _FLOW_APP_FOR_COMMAND.values():
        monkeypatch.setattr(f"{module_path}.get_store", lambda db_path=None: store)

    results: dict[str, FlowStepResult] = {}
    for step in scenario.steps:
        if step.before:
            step.before()
        command = step.args[0]
        app, _module_path, strip = _FLOW_APP_FOR_COMMAND[command]
        invoke_args = step.args[1:] if strip else step.args
        result = _runner.invoke(app, invoke_args)
        if step.expect_failure:
            assert result.exit_code != 0, (
                f"{scenario.name}.{step.name} was expected to fail but exited 0"
            )
        results[step.name] = FlowStepResult(
            extracted=step.extract(result.stdout, result.exit_code),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )
    return results


def run_pyqmd_flow(scenario: CliFlowScenario, store: Store, monkeypatch) -> dict[str, object]:
    """{step_name: extracted} -- the structural-parity view of
    run_pyqmd_flow_detailed."""
    return {
        name: r.extracted
        for name, r in run_pyqmd_flow_detailed(scenario, store, monkeypatch).items()
    }
