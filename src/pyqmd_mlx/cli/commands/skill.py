"""`pyqmd skill list`/`skill show`/`skill install` commands."""

import os
import shutil
import sys
from pathlib import Path

import typer

from pyqmd_mlx.cli._errors import run_or_exit
from pyqmd_mlx.cli._theme import cyan, green

app = typer.Typer(help="List, show, or install pyqmd's bundled agent skills.")

_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


def _available_skills() -> list[str]:
    return sorted(p.name for p in _SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


def _skill_source_dir(name: str) -> Path:
    return _SKILLS_DIR / name


@app.callback()
def _skill_callback() -> None:
    """List, show, or install pyqmd's bundled agent skills.

    A no-op callback: Typer collapses a Typer() app with only one
    @app.command() registered and no callback into a single top-level
    command, which breaks `runner.invoke(app, ["show"])`-style
    invocation ("show" gets consumed as a stray positional argument
    instead of a subcommand name). Kept so the group stays a group.
    """


def get_skill_install_dir(global_: bool, name: str = "pyqmd") -> Path:
    base = Path.home() if global_ else Path.cwd()
    return base / ".agents" / "skills" / name


def get_claude_skill_link_path(global_: bool, name: str = "pyqmd") -> Path:
    base = Path.home() if global_ else Path.cwd()
    return base / ".claude" / "skills" / name


@app.command("show")
def show(
    skill: str = typer.Argument("pyqmd", help="Skill name (see `skill list`)."),
) -> None:
    """Print a bundled skill document."""
    available = _available_skills()

    def _run_show():
        if skill not in available:
            raise ValueError(f"Unknown skill: {skill} (available: {', '.join(available)})")
        typer.echo((_skill_source_dir(skill) / "SKILL.md").read_text(encoding="utf-8"))

    run_or_exit(_run_show)


@app.command("list")
def list_skills() -> None:
    """List bundled skill names (see `skill show <name>` for content)."""
    for name in _available_skills():
        typer.echo(name)


def _skill_description(name: str) -> str:
    """The `description:` frontmatter line of a bundled SKILL.md -- the
    installed stub must carry the real one, since agents pick a skill by
    its description."""
    text = (_skill_source_dir(name) / "SKILL.md").read_text(encoding="utf-8")
    for line in text.split("---", 2)[1].splitlines():
        if line.startswith("description:"):
            return line[len("description:") :].strip()
    raise ValueError(f"Bundled skill {name} has no description in its SKILL.md frontmatter")


def _stub_content(name: str) -> str:
    return f"""---
name: {name}
description: {_skill_description(name)}
---

# {name}

This is a pointer file, not the real skill content -- it can drift out
of sync with whatever pyqmd version is actually installed. Run
`pyqmd skill show {name}` to print the current, version-matched instructions.
"""


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


@app.command("install")
def install(
    skill: str = typer.Argument("pyqmd", help="Skill name (see `skill list`)."),
    global_: bool = typer.Option(
        False,
        "--global",
        help="Install to ~/.agents/skills/<name> instead of ./.agents/skills/<name>.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing install."),
    yes: bool = typer.Option(
        False, "--yes", help="Also create the Claude Code symlink without prompting."
    ),
) -> None:
    """Install a bundled skill doc into .agents/skills/<name> (or ~/.agents/skills/<name> with --global)."""
    install_dir = get_skill_install_dir(global_, skill)

    def _run_install():
        available = _available_skills()
        if skill not in available:
            raise ValueError(f"Unknown skill: {skill} (available: {', '.join(available)})")
        if install_dir.exists() or install_dir.is_symlink():
            if not force:
                raise ValueError(f"Skill already exists: {install_dir} (use --force to replace it)")
            _remove_existing(install_dir)
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_skill_source_dir(skill), install_dir)
        (install_dir / "SKILL.md").write_text(_stub_content(skill), encoding="utf-8")

    run_or_exit(_run_install)
    typer.echo(f"{green('✓')} Installed {skill} skill to {cyan(str(install_dir))}")
    run_or_exit(lambda: _offer_claude_symlink(install_dir, global_, force, yes, skill))


def _offer_claude_symlink(
    install_dir: Path, global_: bool, force: bool, yes: bool, name: str
) -> None:
    link_path = get_claude_skill_link_path(global_, name)

    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == install_dir.resolve():
            typer.echo(f"Claude already sees the skill via {link_path}")
            return
        if not force:
            raise ValueError(
                f"Claude skill path already exists: {link_path} (use --force to replace it)"
            )
        _remove_existing(link_path)

    if yes:
        create = True
    elif not _is_interactive():
        typer.echo(
            f"Tip: run 'pyqmd skill install {name} --force --yes' to also create a "
            f"Claude Code symlink at {link_path}"
        )
        return
    else:
        create = typer.confirm(f"Create a symlink in {link_path}?", default=False)

    if not create:
        return

    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(install_dir, link_path.parent)
    link_path.symlink_to(relative_target, target_is_directory=True)
    typer.echo(f"{green('✓')} Created Claude Code symlink: {cyan(str(link_path))}")
