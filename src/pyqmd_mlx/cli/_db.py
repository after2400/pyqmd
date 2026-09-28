"""DB path resolution and Store lifecycle for the pyqmd CLI.

Uses its own cache directory (~/.cache/pyqmd/) distinct from the live Node
CLI's ~/.cache/qmd/, so the two tools can coexist during the transition
without any risk of one clobbering the other's database.
"""

import os
from pathlib import Path

from pyqmd_mlx.store import Store

DEFAULT_DB_PATH = Path.home() / ".cache" / "pyqmd" / "index.sqlite"

# Lets a user point the CLI at a project-local or test index without any
# command needing a --db option: every command already calls get_store()
# with no arguments, so this env var is checked transparently here.
DB_PATH_ENV_VAR = "PYQMD_DB"


def get_store(db_path: str | None = None) -> Store:
    """Open (creating parent directories if needed) the pyqmd database and
    return a Store. Resolution order: explicit `db_path` argument, then the
    PYQMD_DB environment variable, then DEFAULT_DB_PATH."""
    path = Path(db_path or os.environ.get(DB_PATH_ENV_VAR) or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return Store(str(path))
