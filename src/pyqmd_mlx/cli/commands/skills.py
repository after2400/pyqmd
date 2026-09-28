"""`pyqmd skills list`/`skills get`/`skills path` commands — Node's generic
multi-skill discovery surface (`runSkillsCommand` in Node's
`src/cli/qmd.ts`), ported against the bundled skills under
`src/pyqmd_mlx/skills/`.

Adaptations vs. Node (everything else is verbatim): the skill source is
this package's `_SKILLS_DIR` rather than Node's walk-up-from-package-root
search, with a `PYQMD_SKILLS_DIR` env override mirroring Node's
`QMD_SKILLS_DIR` (renamed per this repo's `PYQMD_*` convention); the
singular `skill show`/`install` group is untouched — this plural group is
purely additive. An unknown subcommand follows Typer's standard `No such
command` handling rather than Node's `Unknown skills subcommand` text.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from pyqmd_mlx.cli.commands.skill import _SKILLS_DIR

app = typer.Typer(help="List and retrieve pyqmd's bundled runtime skills.")

_SKILL_SUBDIRS = ("references", "templates", "scripts")


@dataclass
class SkillInfo:
    name: str
    description: str
    dir: Path
    hidden: bool


def _skills_search_dirs() -> list[Path]:
    """Where to discover skills: `PYQMD_SKILLS_DIR` when set (Node's
    `QMD_SKILLS_DIR`), else the bundled skills directory."""
    override = os.environ.get("PYQMD_SKILLS_DIR")
    if override:
        return [Path(override)]
    return [_SKILLS_DIR] if _SKILLS_DIR.is_dir() else []


def parse_skill_frontmatter(content: str) -> tuple[str, str, bool] | None:
    """Parse a skill's `name:`/`description:`/`hidden:` frontmatter, with
    Node's semantics: multi-line `description:` continuations (indented
    lines) are joined with spaces; `hidden:` accepts `true`/`yes`."""
    trimmed = content.lstrip()
    if not trimmed.startswith("---"):
        return None
    rest = trimmed[3:]
    end = rest.find("\n---")
    if end < 0:
        return None
    frontmatter = rest[:end]
    name = ""
    description = ""
    hidden = False
    lines = frontmatter.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("name:"):
            name = line[len("name:") :].strip()
        elif line.startswith("description:"):
            parts = [line[len("description:") :].strip()]
            while i + 1 < len(lines) and re.match(r"^\s+\S", lines[i + 1]):
                i += 1
                parts.append(lines[i].strip())
            description = " ".join(parts)
        elif line.startswith("hidden:"):
            hidden = line[len("hidden:") :].strip().lower() in ("true", "yes")
        i += 1
    if not name:
        return None
    return (name, description, hidden)


def discover_skills() -> list[SkillInfo]:
    """All discoverable skills (including hidden ones), sorted by name."""
    found: list[SkillInfo] = []
    for search_dir in _skills_search_dirs():
        try:
            entries = list(search_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            skill_path = entry / "SKILL.md"
            if not skill_path.is_file():
                continue
            try:
                content = skill_path.read_text(encoding="utf-8")
            except OSError:
                continue
            parsed = parse_skill_frontmatter(content)
            if parsed is None:
                continue
            name, description, hidden = parsed
            found.append(SkillInfo(name=name, description=description, dir=entry, hidden=hidden))
    return sorted(found, key=lambda skill: skill.name)


def _runtime_skills() -> list[SkillInfo]:
    """Skills shown by `list` and `get --all` — hidden ones excluded."""
    return [skill for skill in discover_skills() if not skill.hidden]


def find_skill(name: str) -> SkillInfo | None:
    """Find a skill by name, including hidden ones (Node's `findSkill`
    applies no hidden filter — hiding only removes a skill from listings)."""
    return next((skill for skill in discover_skills() if skill.name == name), None)


def collect_skill_files(skill: SkillInfo) -> list[tuple[str, str]]:
    """Supplementary `(relative path, content)` files for `get --full`."""
    files: list[tuple[str, str]] = []
    for subdir_name in _SKILL_SUBDIRS:
        subdir = skill.dir / subdir_name
        if not subdir.is_dir():
            continue
        for entry in sorted(subdir.iterdir(), key=lambda p: p.name):
            if not entry.is_file():
                continue
            try:
                files.append((f"{subdir_name}/{entry.name}", entry.read_text(encoding="utf-8")))
            except OSError:
                # Ignore unreadable supplementary files.
                pass
    return files


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload))


def _effective_json(ctx: typer.Context | None, flag: bool) -> bool:
    """A subcommand's `--json` OR the group-level `skills --json` flag —
    Node parses `--json` globally, so both positions must work."""
    if flag:
        return True
    obj = ctx.obj if ctx is not None else None
    return isinstance(obj, dict) and bool(obj.get("json"))


def _write_content(content: str) -> None:
    typer.echo(content if content.endswith("\n") else content + "\n", nl=False)


def _run(json_mode: bool, fn) -> None:
    """Run a skills subcommand, mapping expected failures to Node's
    reporting: a `{"success": false, "error": msg}` envelope on stdout
    under `--json`, else the bare message on stderr — both exit 1.
    Unexpected exceptions propagate with their traceback instead of using
    the error envelope — a deliberate narrowing vs. Node, whose skills
    dispatch catches everything (only expected user-facing failures get
    the clean one-line treatment here).
    """
    try:
        fn()
    except (ValueError, OSError) as exc:
        if json_mode:
            _emit_json({"success": False, "error": str(exc)})
        else:
            typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _read_skill_content(skill: SkillInfo) -> str:
    return (skill.dir / "SKILL.md").read_text(encoding="utf-8")


def _emit_list(json_mode: bool) -> None:
    def _run_list() -> None:
        skills = _runtime_skills()
        if json_mode:
            _emit_json(
                {
                    "success": True,
                    "data": [
                        {"name": skill.name, "description": skill.description} for skill in skills
                    ],
                }
            )
            return
        if not skills:
            typer.echo("No skills found")
            return
        width = max(len(skill.name) for skill in skills)
        for skill in skills:
            typer.echo(f"  {skill.name.ljust(width)}  {skill.description}")

    _run(json_mode, _run_list)


@app.callback(invoke_without_command=True)
def _skills_callback(
    ctx: typer.Context,
    json_mode: bool = typer.Option(False, "--json", help="Print structured JSON."),
) -> None:
    """List and retrieve pyqmd's bundled runtime skills.

    With no subcommand this lists skills, matching Node (whose `skills`
    dispatch defaults to `list`). The callback also keeps the group from
    collapsing into a single top-level command, which would break
    `runner.invoke(app, ["list"])`-style invocation.
    """
    ctx.obj = {"json": json_mode}
    if ctx.invoked_subcommand is None:
        _emit_list(json_mode)


@app.command("list")
def list_skills(
    ctx: typer.Context,
    json_mode: bool = typer.Option(False, "--json", help="Print structured JSON."),
) -> None:
    """List bundled runtime skills."""
    _emit_list(_effective_json(ctx, json_mode))


@app.command("get")
def get_skill(
    ctx: typer.Context,
    names: Optional[list[str]] = typer.Argument(None, help="Skill name(s) (see `skills list`)."),
    full: bool = typer.Option(False, "--full", help="Include references/templates/scripts."),
    all_: bool = typer.Option(False, "--all", help="Print all bundled runtime skills."),
    json_mode: bool = typer.Option(False, "--json", help="Print structured JSON."),
) -> None:
    """Print a bundled runtime skill."""
    names = names or []
    json_mode = _effective_json(ctx, json_mode)

    def _run_get() -> None:
        targets = _runtime_skills() if all_ else [_require_skill(name) for name in names]
        if not targets:
            raise ValueError("No skill name provided. Usage: pyqmd skills get <name>")
        if json_mode:
            _emit_json(
                {
                    "success": True,
                    "data": [
                        {
                            "name": skill.name,
                            "content": _read_skill_content(skill),
                            **(
                                {
                                    "files": [
                                        {"path": path, "content": content}
                                        for path, content in collect_skill_files(skill)
                                    ]
                                }
                                if full
                                else {}
                            ),
                        }
                        for skill in targets
                    ],
                }
            )
            return
        for index, skill in enumerate(targets):
            if index > 0:
                typer.echo("\n---\n")
            _write_content(_read_skill_content(skill))
            if full:
                for path, content in collect_skill_files(skill):
                    typer.echo(f"\n--- {path} ---\n")
                    _write_content(content)

    _run(json_mode, _run_get)


def _require_skill(name: str) -> SkillInfo:
    skill = find_skill(name)
    if skill is None:
        raise ValueError(f"Skill not found: {name}")
    return skill


@app.command("path")
def skill_path(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Skill name (see `skills list`)."),
    json_mode: bool = typer.Option(False, "--json", help="Print structured JSON."),
) -> None:
    """Print runtime skill directory path(s)."""
    json_mode = _effective_json(ctx, json_mode)

    def _run_path() -> None:
        if name is None:
            paths = [str(search_dir) for search_dir in _skills_search_dirs()]
            if json_mode:
                _emit_json({"success": True, "data": {"paths": paths}})
            else:
                for path in paths:
                    typer.echo(path)
            return
        skill = _require_skill(name)
        if json_mode:
            _emit_json({"success": True, "data": {"name": skill.name, "path": str(skill.dir)}})
        else:
            typer.echo(str(skill.dir))

    _run(json_mode, _run_path)


@app.command("help")
def skills_help() -> None:
    """Print `skills` usage."""
    typer.echo("Usage: pyqmd skills <list|get|path> [options]")
    typer.echo("")
    typer.echo("Commands:")
    typer.echo("  list                 List bundled runtime skills")
    typer.echo("  get <name>           Print a bundled runtime skill")
    typer.echo("  get <name> --full    Include references/templates/scripts")
    typer.echo("  get --all            Print all bundled runtime skills")
    typer.echo("  path [name]          Print runtime skill directory path(s)")
    typer.echo("")
    typer.echo("Options:")
    typer.echo("  --json               Print structured JSON")
