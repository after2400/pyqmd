# Model sets, a config file, and `pyqmd models` — design

**Date:** 2026-11-04
**Status:** Implemented. The default model set is unchanged; flipping the default to Granite is
explicitly out of scope until it has been benchmarked on a real notes corpus.
**Related:** `2026-11-04-us-expansion-model-design.md` (the Granite expansion model),
`2026-11-04-granite-reranker-design.md` (the Granite reranker),
`2026-09-24-expand-model-hf-default-design.md` (`PYQMD_EXPAND_MODEL`, `ExpansionModelError`).

## Why

pyqmd now ships two model sets: the default (a Qwen3-based expansion model and Qwen3-Reranker, both
Alibaba bases) and an opt-in set on IBM Granite. Today choosing between them means two environment
variables, one of which (`PYQMD_EXPAND_MODEL`) has no command-line flag, and nothing persists: an MCP
server only sees what its launch config sets (the README already warns about this for breadcrumbs).

Concrete scenario: on a machine behind a TLS-inspecting proxy that rejects some Hugging Face
repositories, the first `pyqmd query` fails because it tries to download the default Qwen models.
The user wants to run one command, choose the Granite set, and have every later run use Granite only,
without ever trying to download a Qwen model; and a failed first attempt must not leave anything that
makes the retry fail.

## Goals

1. A persistent choice: a config file read by the CLI and the MCP server alike.
2. `pyqmd models use granite` selects the Granite set without downloading or touching any default
   (Qwen) model; `--download` additionally fetches only the selected set.
3. `pyqmd pull` becomes a real download of the active models (today a no-op stub).
4. A model download blocked by the network yields one clean, actionable error (no traceback) that
   names the way out; rerunning with a flag, an env var or `models use granite` succeeds.
5. Docs: README "Models" section, `SKILL.md`, `CLAUDE.md`, `COMMAND_STATUS.md`.

## Non-goals

- Changing the default set (a later, separate change).
- Making the embedding model configurable: changing it re-indexes, and no alternative exists yet.
  The embedding model (Google's embeddinggemma) is shared by both sets, so it needs no choice.
- Other settings in the config file (database path, breadcrumbs). The format leaves room; no keys
  are added now.
- Interactive prompts. A first-run question would break MCP servers and scripts.
- Any workaround for a blocked network beyond a clear message (no proxy configuration, no mirror).

## Design

### 1. Model sets

Two named sets in `llm/_constants.py`; each role's model id lives in one place:

| Set       | Expansion                                                | Rerank                                              | Notes                               |
| --------- | -------------------------------------------------------- | --------------------------------------------------- | ----------------------------------- |
| `default` | `after2400/qmd-query-expansion-1.7B-mlx-mixed-4-6`       | `mlx-community/Qwen3-Reranker-0.6B-mxfp8`           | Alibaba bases; parity with Node qmd |
| `granite` | `after2400/qmd-query-expansion-granite-1b-mlx-mixed-4-6` | `ibm-granite/granite-embedding-reranker-english-r2` | IBM Granite; US-made; English only  |

`DEFAULT_EXPAND_MODEL` / `DEFAULT_RERANK_MODEL` stay as the `default` set's entries.
`GRANITE_EXPAND_MODEL` is new (`GRANITE_RERANK_MODEL` exists). Embedding is not part of a set.

### 2. Resolution order (per role)

First match wins:

1. command-line flag: `--rerank-model` (exists), `--expand-model` (new, on `query` and `bench`);
2. `PYQMD_RERANK_MODEL` / `PYQMD_EXPAND_MODEL`;
3. `PYQMD_MODEL_SET` (a set name; new);
4. the config file's explicit `expand` / `rerank` key;
5. the config file's `set` key;
6. the built-in `default` set.

An unknown set name (env or file) is a `ConfigError` naming the valid sets. Empty strings are
treated as unset, as for the existing variables. `resolve_expand_model()` and
`resolve_rerank_model()` implement 2–6, so `Store`, the MCP server, `bench` and `status` need no
other change; flags keep going through the existing `Store(rerank_model=...)` / `expand_model=...`
arguments.

### 3. The config file

- Location: `$PYQMD_CONFIG` if set, else `$XDG_CONFIG_HOME/pyqmd/config.toml`, else
  `~/.config/pyqmd/config.toml`. A missing file is normal (nothing configured).
- Format: TOML, read with `tomllib` (Python ≥ 3.11 is already required):

  ```toml
  [models]
  set = "granite"        # "default" or "granite"
  # expand = "org/repo-or-local-dir"   # explicit overrides, beat `set`
  # rerank = "org/repo-or-local-dir"
  ```

- Reading: invalid TOML, a wrong type, or an unknown key under `[models]` raises `ConfigError`
  (a `ValueError`, so the CLI's existing handler prints `Error: <file>: <what>` and exits 1, and the
  MCP wrapper returns `is_error`). Unknown top-level tables are ignored (forward compatibility).
- Writing (only `models use`): load, change `[models]`, write the whole file back atomically
  (temp file + rename) with a small serializer for string-valued tables. Comments in a file pyqmd
  rewrites are lost; the `models use` output says so when the file existed. The directory is created
  if missing.

### 4. Commands

`pyqmd models` is a Typer sub-app; bare `pyqmd models` runs `show`.

- `pyqmd models show` — for each role (embedding, reranking, generation): the model id, **where
  it came from** (flag is per-run so not shown; `env PYQMD_..`, `config <path>`, `default`), and
  whether it is already in the Hugging Face cache (`cached` / `not downloaded`; a local directory is
  `local`). Never touches the network.
- `pyqmd models list` — the sets, with role models, makers and one-line notes.
- `pyqmd models use <set> [--download]` — writes `set = "<set>"` and removes explicit `expand` /
  `rerank` keys from `[models]` (otherwise they would silently still win; it prints what it
  removed). Prints a note for any `PYQMD_*MODEL*` env var currently set, since that still wins.
  **It never downloads or loads anything unless `--download` is given, and then only the models of
  the chosen set** (plus the shared embedding model). An unknown set exits 1 listing the valid ones.
- `pyqmd pull` — downloads the active models: resolves the three roles exactly as a query would,
  skips local directories and cached repos, and `huggingface_hub.snapshot_download`s the rest.
  `--refresh` (accepted today, ignored) is kept hidden and ignored. Per model it prints one line
  (`downloaded`, `already cached`, or the failure); the exit status is 1 if any failed, and every
  failure uses the same message as a failed load (§5). It downloads files only; it does not load a
  model into memory.
- `pyqmd models use granite --download` is `use` followed by `pull`. If the download fails, the
  selection stays written (the choice is the config's job, the download is retryable) and the exit
  status is 1.

`status` already prints the three model ids; it gains nothing new except that, being driven by
`resolve_*`, it reflects the config.

### 5. Blocked or failed downloads

Shared helper `llm/_load_errors.py` turns an exception from a model load or download into one
message, used by the expansion and rerank loaders (replacing their private wording), by a new
`EmbeddingModelError` (the embedding loader had no wrapping, so a blocked embedding model gave a
raw traceback), and by `pull`.

Message shape: `Could not load <role> model '<id>': <short reason>. <remedy>`

- **Reason**, classified from the exception chain (checked by type name and message, because the
  `huggingface_hub` / `httpx` / `requests` exception classes differ by version):
  - HTTP 401/403 or gated repo → "access was denied (a network filter blocks this repository, or it
    needs a login)";
  - TLS verification failure (`CERTIFICATE_VERIFY_FAILED`, `SSLError`) → "the connection failed TLS
    verification (a corporate proxy that inspects traffic needs its CA certificate: point
    `SSL_CERT_FILE` at your company's CA bundle)";
  - connection / timeout / offline → "Hugging Face could not be reached";
  - anything else → the original message, trimmed to one line.
- **Remedy**:
  - when the failing model is a built-in `default`-set model: "Use the IBM Granite set instead:
    `pyqmd models use granite`, or `PYQMD_MODEL_SET=granite`, or `--rerank-model` /
    `--expand-model`";
  - otherwise (an explicit choice): "Set `<VAR>` to another Hugging Face repo id or a local MLX
    model directory."
- Nothing is cached on failure (`get_or_load` caches only successes, as before), so the next run
  starts clean. No automatic fallback to another model: results would change silently.

**Guarantee (and its test):** with the Granite set selected — by file, env var or flags — no code
path loads, downloads or even checks the cache for a `default`-set model. `Store` resolves ids once;
the reranker backend check (`is_cross_encoder`) reads the _selected_ model's `config.json` only;
`pull` and `models use --download` take their ids from the resolved roles. A test replaces the hub
and the MLX loaders with fakes that raise if a default-set id is ever requested, then runs `query`,
`embed`, `status`, `pull` and `models use granite --download` with the Granite set selected.

### 6. MCP server

No code change: `Store` resolves models through the same functions, so the config file reaches the
MCP server without environment variables. The README's MCP section says so. Load failures are
already returned as `is_error` results by `tool_result_or_error`; `EmbeddingModelError` is added to
the expected exceptions of both the CLI and MCP handlers.

### 7. Docs (in the same change)

- **README:** a "Models" section — the role/default/alternative table, sizes and licences, the
  ways to choose (`pyqmd models use`, config file, env vars, flags) with the precedence order, the
  blocked-network note, English-only for the Granite reranker, and the headline benchmark numbers.
- **`SKILL.md`:** fix "~934 MB" (the expansion model is ~1.0 GB), document `models`, `pull`,
  `--expand-model`, the config file and `PYQMD_MODEL_SET`. It ships in the wheel, so this part is
  `fix`/`feat`, not `docs`.
- **`CLAUDE.md`:** command list (keep-in-sync rule: `pull`, `models`, `--expand-model`) and the
  Architecture paragraph.
- **`COMMAND_STATUS.md`:** `pull` (now a real download, closer to Node's), `models` (pyqmd-only),
  `query` / `bench` flags.

## Testing

All fast (no weights, no network):

- Resolution: a table over every precedence level and both roles; unknown set; empty strings.
- Config: missing file; valid; invalid TOML, wrong types, unknown `[models]` key; `$PYQMD_CONFIG`
  and XDG location; write round-trip (creates directory, keeps unrelated tables, atomic).
- Commands (CliRunner, `PYQMD_CONFIG` in a temp dir): `models show` sources and cache status
  (fake `try_to_load_from_cache`); `models list`; `models use granite` writes only config, removes
  explicit keys, notes env overrides, rejects unknown sets, and without `--download` makes zero
  hub calls; `--download` fetches exactly the three Granite-set ids; `pull` per-model lines and exit
  status; bare `models` shows.
- Errors: each classification from synthetic exception chains (403, TLS, connection); remedy text
  differs for default-set vs explicit models; `EmbeddingModelError` end to end through the CLI and
  MCP; no traceback.
- The guarantee test in §5.
- A manual end-to-end check (not in the suite): a Hugging Face cache directory that contains only
  the Granite and embedding repos with `HF_HUB_OFFLINE=1` — default set fails cleanly naming
  `pyqmd models use granite`; after running it, `pyqmd query` succeeds.

## Risks and open points

- **A TLS-inspecting proxy's behaviour is not reproduced here.** The classification keys on exception text, and the
  exact errors a TLS-inspecting proxy produces depend on its policy (it may return 403, drop the
  connection, or present an untrusted certificate). The manual check above proves the _offline_
  path, not the proxy's. The message always includes the original reason, so an unclassified case is
  still diagnosable. To confirm on the real network, run `pyqmd pull` there once.
- **Comments in `config.toml` are lost on rewrite.** Accepted: the file is small and pyqmd-owned;
  hand-edits still work, they just do not survive `models use`.
- **`PYQMD_MODEL_SET` adds a third selection mechanism.** Kept because one line in an MCP client's
  `env` block is the smallest way to select a set there; the config file is the persistent route.
- **Hugging Face download semantics.** `snapshot_download` fetches every file in a repo; both
  Granite repos are single-model repos, so that is what a query would fetch anyway.
- The default flip stays gated on the real-notes bench run (see the expansion-model spec).

## Out of scope

Flipping the default set; a configurable embedding model; other config keys; interactive prompts;
proxy or mirror configuration.
