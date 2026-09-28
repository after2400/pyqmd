export default {
  extends: ["@commitlint/config-conventional"],
  rules: {
    // 0 = off, 1 = warn, 2 = error  |  "always" / "never"
    "scope-enum": [
      2,
      "always",
      [
        "store", // src/pyqmd_mlx/store/ -- schema, Store class, search/query
        "cli", // src/pyqmd_mlx/cli/ -- Typer commands, output formatting
        "mcp", // src/pyqmd_mlx/mcp/ -- MCP server (stdio/HTTP)
        "llm", // src/pyqmd_mlx/llm/ -- MLX embed/rerank/query-expansion layer
        "bench", // src/pyqmd_mlx/bench/ and the bench command
        "skills", // src/pyqmd_mlx/skills/ and the skill/skills commands
        "parity", // parity/ -- Node-vs-pyqmd golden-snapshot suite
        "scripts", // scripts/ -- dev utility scripts
        "config", // pyproject.toml, .pre-commit-config.yaml, Justfile, .gitignore
        "docs", // README/CLAUDE.md/COMMAND_STATUS.md and any docs/ that aren't specs
        "specs", // docs/specs/ -- published design/spec docs
        "tests", // test-only changes
      ],
    ],
    "scope-empty": [0, "never"], // off - some commits legitimately have no scope
    "type-enum": [
      2,
      "always",
      [
        // conventional commits standard types
        "feat",
        "fix",
        "docs",
        "style",
        "refactor",
        "perf",
        "test",
        "build",
        "ci",
        "chore",
        "revert",
        // escape hatch — never bumps version, never appears in changelog
        "doh",
      ],
    ],
  },
};
