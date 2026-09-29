"""Single source of truth for the pyqmd release version.

Updated automatically by python-semantic-release on each release (see
pyproject.toml's [tool.semantic_release] version_variables) — never
bump by hand. Read at runtime by the CLI's --version flag and the MCP
server's _package_version(); read at build time by hatchling (see
[tool.hatch.version] path in pyproject.toml).
"""

__version__ = "0.6.6"
