set shell := ["bash", "-uc"]

# Just runs the first target if no arguments are provided, so we set this to list all targets for safety.
_default:
    @just --list

# Install/upgrade dev dependencies and activate pre-commit hooks.
[group('development')]
develop:
    uv sync --dev
    uv tool install pre-commit
    pre-commit install

[group('quality')]
test *args:
    uv run pytest {{args}} -v

[group('quality')]
test-fast:
    uv run pytest -v -m "not slow"

[group('quality')]
test-slow:
    uv run pytest -v -m slow

[group('quality')]
test-parity *args:
    uv run pytest parity/ {{args}}

# Performance comparison against a Node qmd checkout at ../qmd (override with e.g. `just bench-scifact --qmd-repo-root /path/to/qmd`).
# Uses the built-in SciFact profile by default (override with `--dataset-config path/to/profile.yaml`,
# same as `test-parity`) -- its corpus lives in data/scifact/, gitignored and not shipped with the repo;
# generate it first with `uv run scripts/prepare_scifact_corpus.py`.
[group('quality')]
bench-scifact *args:
    uv run python -m parity.benchmark --qmd-repo-root ../qmd {{args}}

[group('quality')]
lint:
    uv run ruff check .
    uv run ruff format --check .

# Install pyqmd as a real `pyqmd` command on PATH, editable (source changes apply without reinstalling).
[group('install')]
install:
    uv tool install --editable .

[group('install')]
uninstall:
    uv tool uninstall pyqmd-mlx
