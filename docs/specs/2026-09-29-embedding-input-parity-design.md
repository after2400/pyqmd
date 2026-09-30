# Embedding input parity — design

**Status:** Implemented (2026-09-29)

**Related:** the roadmap's "Feature ideas from the PyPI name/collision
spike" entry (per-chunk heading breadcrumbs). This spec is the baseline that
breadcrumbs gets measured against; see "Follow-up: breadcrumbs" below.

## Problem

pyqmd embeds each chunk differently from Node `qmd`, and nothing records
that as a choice. For the same document and the same chunk, the text pyqmd
hands the embedding model differs from Node's in two ways, and pyqmd has no
way to notice that stored vectors came from an older format.

Compared against the pinned Node reference (`8262698`):

| #   | Gap                                                       | Node                                                                                                                                                                                                   | pyqmd today                                                                                                                                                                   |
| --- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **No title in embeddings**                                | `title: <extractTitle(body, MIN(path))> \| text: <chunk>` (`store.ts:2125`, `llm.ts:112`)                                                                                                              | `title: none \| text: <chunk>` for every chunk; `index_content` never passes a title                                                                                          |
| 2   | **Embeddings cut at 512 tokens**                          | 2048-token embedding context (`EMBED_CONTEXT_SIZE`, capped at the model's trained context)                                                                                                             | `batch_encode_plus(..., truncation=True, max_length=512)` in `llm/embed.py`, since the first MLX commit                                                                       |
| 3   | **Title extraction ignores file type**                    | per extension: `.md` heading rule, `.org` `#+TITLE`/`*` heading, else the file name                                                                                                                    | the markdown heading rule runs on every file; no `.org` rule                                                                                                                  |
| 4   | **`update` never refreshes a title on unchanged content** | same hash, new title → `updateDocumentTitle` (title, `modified_at`, FTS), counted as updated                                                                                                           | same hash → unchanged; the old title stays                                                                                                                                    |
| 5   | **No embedding fingerprint**                              | each vector row stores `getEmbeddingFingerprint(model)`; a mismatch counts as pending                                                                                                                  | `content_vectors.embed_fingerprint` exists but is always `''`; only moved chunk positions count as stale                                                                      |
| 6   | **No index-health warning on `vsearch`/`query`**          | `checkIndexHealth` prints a stderr warning or tip for pending embeddings and a stale index                                                                                                             | nothing                                                                                                                                                                       |
| 7   | **Embedding chunks are the wrong size**                   | embed-time chunking estimates 3 chars/token: 2700-char chunks, 405 overlap, 600 window (`chunkDocumentByTokensWithLlm`, `store.ts:3283`)                                                               | 4 chars/token for everything: 3600 / 540 / 800                                                                                                                                |
| 8   | **Chunk boundaries differ around code fences**            | a cut can land inside a fence when no break point outside it is in the window; an unclosed fence runs to end of file; when overlap would stall, the next chunk starts at the previous end (no overlap) | cuts and overlap starts are pushed forward past any fence (`_extend_past_fence`); an unclosed fence is ignored; a stalled overlap restarts halfway through the previous chunk |

None of gaps 1–7 is recorded anywhere as a deliberate choice. Of gap 8,
only the unclosed-fence rule is (in `find_code_fences`' docstring); the
other two behaviours are explained in code comments but never named as
differences from Node.

Gap 2 is the largest in practice. pyqmd's full chunks are ~900 tokens
(3600 chars) and Node's 675–900 (2700 chars); either way well over 512,
so pyqmd's vectors see only the start of every full chunk and the rest is
invisible to vector search. Gap 7 compounds it: pyqmd's embed
chunks are a third longer than Node's, so different text lands in each
vector. Gap 1 drops the one line of context Node gives every chunk. Gap 3
gives non-markdown files odd titles (a `.py` file whose first line is
`# setup helpers` is titled "setup helpers"; Node titles it by file name),
which shows in results and weighs 4× in BM25.

Gaps 1–3, 7 and 8 need a re-embed. Gap 5 is what makes that re-embed
happen without `embed --force`, and gap 6 is how a user finds out it's
needed. Gap 4 is what makes gap 3's fix reach documents that didn't change.

The original scope was gaps 1, 3 and 5. Gaps 2, 4 and 6 turned up while
reading the code for them. Gaps 7 and 8 came from an audit of Node's whole
index → embed path against pyqmd's (see "Audit" below). All are included
because each one is either the same re-embed (2, 7, 8) or needed for
another gap's fix to reach existing users (4, 6).

Query-time chunking (picking each result's best chunk for the reranker and
the snippet) is not affected: Node uses 3600/540/800 there
(`hybridQuery` calls `chunkDocumentAsync` with its defaults), and so does
pyqmd. Node uses two chunk sizes, and pyqmd will too.

## Goal

For every chunk, the text pyqmd embeds is the text Node embeds, apart from
the documented differences below (token-verified chunking, and pyqmd's
code fence rules, section 8). Stored vectors carry a fingerprint, so any
future change to the
embedding input marks them stale on its own, and `status`, `embed`,
`vsearch` and `query` all say so.

Not a goal: bit-identical vectors. pyqmd runs an MLX conversion of
embeddinggemma and Node runs a GGUF build, so vectors differ numerically
regardless; the parity suite compares results, not vectors.

## Design

### 1. Titles in document embeddings

- `pyqmd_mlx.llm.embed()` gains `title: str | None = None`. With
  `kind="document"`, every text in the batch is formatted with that title
  (`format_doc_for_embedding(text, model, title)`, which already implements
  both Node formats: `title: <t> | text: <c>`, or `<t>\n<c>` for Qwen
  embedding models). With `kind="query"`, `title` is ignored.
- One title per call is enough: `index_content` embeds one document's
  chunks per call, and they all share the document's title.
- `Store.index_content` computes the title with the fixed
  `extract_title(content, filepath)` (section 3), where `filepath` is the
  `MIN(path)` it already receives from `get_indexable_content`, matching
  Node's `extractTitle(doc.body, doc.path)`. A caller that passes no
  `filepath` (the parity suite's index fixture and two scripts do today)
  gets that `MIN(path)` looked up from `documents`, so no caller can
  silently embed with `title: none`; only content with no active document
  at all (bare unit tests) embeds untitled. It does not use the stored
  `documents.title`: content is shared across paths by hash, and Node
  derives the embedding title the same way.
- The `embed_fn` contract changes accordingly: Store calls
  `embed_fn(texts, model, kind="document", title=...)`. Every test fake
  (32 of them today, `(texts, model, kind="query")`) gains a
  `title=None` parameter.

### 2. Embedding token limit: 512 → 2048

- `embed()` truncates at `min(2048, tokenizer.model_max_length)`, mirroring
  Node's `resolveEmbedTokenLimit()` (`min(EMBED_CONTEXT_SIZE,
trainContextSize)`). For the default model both are 2048
  (`max_position_embeddings` and `model_max_length` in its config).
- A named constant `EMBED_CONTEXT_TOKENS = 2048` in `llm/_constants.py`.
- Applies to queries too; queries are short, so nothing changes for them.
- Cost: up to ~1.8× more tokens per full chunk, so `embed` gets slower on
  long documents. Measured in "Verification" below.

### 3. Title extraction by file type

`_extract_title` in `store/_indexing.py` becomes a port of Node's
`titleExtractors` + `extractTitle` (`store.ts:2926-2956`), and moves to its
own small module (`store/_title.py`) so both indexing and `index_content`
can use it:

- The extension is taken from the path and lower-cased (`.MD` counts as
  `.md`).
- `.md`: first `^##?\s+(.+)$` heading; if it is `📝 Notes` or `Notes`, the
  first `^##\s+(.+)$` heading instead. (Same as today's rule, now scoped
  to `.md`.)
- `.org`: `^#\+TITLE:\s*(.+)$` (case-insensitive), else the first
  `^\*+\s+(.+)$` heading.
- An extension with no extractor, or an extractor that finds nothing: the
  file name without its last extension. This is a mechanical port of
  Node's `filename.replace(/\.[^.]+$/, "").split("/").pop() || filename`
  on the relative path, quirks included: `.env` stays `.env`, `README`
  stays `README`, and `dir.v2/README` becomes `dir` (the regex strips from
  the last `.` in the whole path). pyqmd's `Path(...).stem` gives `README`
  for that last case today; Node's answer wins.

`.markdown`/`.mdx` files get the file-name fallback, as in Node.

Two regex details, checked against Node's own function in bun
(2026-09-29, while implementing):

- `\s` matches newlines in both languages, so `#+TITLE:` followed by a
  line break captures the next line: `#+TITLE:   \n* Heading` is titled
  `* Heading`. Only a `#+TITLE:` with nothing after it at all (end of
  file) comes out empty and falls back to the file name. (An earlier
  draft of this spec said an empty `#+TITLE` always falls back; wrong.)
- JS's `.`, and multiline `^`/`$`, treat `\r`, U+2028 and U+2029 as line
  breaks; Python's only `\n`. The port uses explicit JS line classes, so
  `# a<U+2028>b` is titled `a` as in Node.

### 4. `update` refreshes changed titles

In `scan_and_register_collection`, when a document's hash is unchanged but
the freshly extracted title differs from the stored one:

- A new `Store.update_document_title(document_id, title, modified_at)`
  sets `title` and `modified_at` (to the scan's `now`, as Node does) and
  re-syncs that document's FTS row.
- It counts as `updated`, not `unchanged`, so the summary line matches
  Node's.

After upgrading, the first `pyqmd update` (or `collection add` re-scan)
fixes every title gap 3 affects. `update` never re-embeds; the fingerprint
below is what marks those vectors stale.

### 5. Embedding fingerprint

- A port of Node's `getEmbeddingFingerprint(model)` (`store.ts:141`): the
  first 6 hex chars of SHA-256 over these lines, joined with `\n`:
  - `model:<model id>`
  - `query:<format_query_for_embedding(PROBE_QUERY, model)>`
  - `doc:<format_doc_for_embedding(PROBE_DOC, model, PROBE_TITLE)>`
  - `chunk_tokens:900`
  - `chunk_overlap_tokens:135`

  with Node's probe strings (`__qmd_embedding_query_probe__`,
  `__qmd_embedding_title_probe__`, `__qmd_embedding_document_probe__`).
  It lives in `store/_fingerprint.py`: it combines the `llm` formatters
  with the store's chunk constants, and `llm` importing `store` would be
  circular (`store` already imports `llm`).

- The values differ from Node's (the model ids differ), which doesn't
  matter: pyqmd has its own index.
- The 512 → 2048 change is not an input to the fingerprint, as in Node.
  It's caught anyway because every existing pyqmd vector has an empty
  fingerprint.
- **Writes:** `index_content` stores the current fingerprint on every
  `content_vectors` row it inserts.
- **Already-embedded check** (`index_content`): unchanged only if the
  stored positions match a fresh chunking pass (as today) **and** every
  stored row for this hash and model has the current fingerprint.
- **Pending counts**: `count_pending_embed` and `get_status_counts`'
  `pending_embed` count a hash as pending when it has no `content_vectors`
  row for this model **with the current fingerprint** (Node's
  `getHashesNeedingEmbedding`). The existing auto-chunk-strategy stale scan
  in `count_pending_embed` stays as it is.
- Empty-fingerprint rows (every vector in every existing pyqmd index) are
  therefore stale. `embed` (no `--force`) re-embeds them; `status` counts
  them as pending; the MCP instructions' existing "N documents need
  embedding" note picks this up with no change.

### 6. Index-health warning on `vsearch` and `query`

A port of Node's `checkIndexHealth` (`qmd.ts:361-378`), called at the start
of the `vsearch` and `query` commands, as in Node, writing to stderr:

- pending ≥ 10% of active documents:
  `Warning: N documents (P%) need embeddings. Run 'pyqmd embed' for better results.`
  (yellow)
- pending > 0 and < 10%:
  `Tip: N documents need embeddings. Run 'pyqmd embed' to index them.`
  (dim)
- the most recent document `modified_at` is 14 or more days old:
  `Tip: Index last updated N days ago. Run 'pyqmd update' to refresh.`
  (dim)

`qmd` becomes `pyqmd` in the wording, as in every other pyqmd message. The
MCP `query` tool doesn't warn (Node's doesn't either); its instructions
already carry the pending note.

### 7. Embed-time chunk size

- New constants in `store/_chunking.py`, next to the existing ones:
  `EMBED_CHARS_PER_TOKEN = 3`, and `EMBED_CHUNK_SIZE_CHARS` (2700),
  `EMBED_CHUNK_OVERLAP_CHARS` (405), `EMBED_CHUNK_WINDOW_CHARS` (600)
  derived from the same token constants.
- One function builds a document's embedding inputs,
  `embedding_chunks(content, filepath, chunk_strategy)`, calling
  `chunk_document` with the embed sizes. Every embed-side caller goes
  through it: `index_content` (chunking and the already-embedded
  position check) and `count_pending_embed`'s auto-strategy stale scan.
  Query-time callers (`Store.query`'s best-chunk selection) keep
  `chunk_document`'s defaults, as Node does.
- Stored positions change for every multi-chunk document, so the existing
  position check would flag them on its own; the empty fingerprint already
  does.

### 8. Chunk boundaries around code fences: keep pyqmd's rules

**Decided (2026-09-29): keep pyqmd's rules and document them as
deliberate differences from Node.** Three rules differ (table, gap 8).
They only matter where a code fence sits near a cut point or an overlap
start, or a fence is left unclosed; fence-free text chunks identically.

They exist for a reason: never splitting a code block keeps code examples
whole in one vector and one snippet, and ignoring a lone ``` stops one
stray marker from collapsing a whole file into one chunk. The cost is that
code-heavy documents chunk differently from Node's.

- `_chunking.py`'s module docstring gets a short "Differences from Node"
  list naming all three rules, and `COMMAND_STATUS.md`'s `embed` row
  mentions them, so the next audit finds them recorded.
- The model-input tests (Testing) use fence-free fixtures only; pyqmd's
  existing fence tests keep covering the fence rules.

Rejected: **match Node exactly** (remove `_extend_past_fence`, extend an
unclosed fence to end of file, restart a stalled overlap at the previous
end). Exact parity on code-heavy documents, at the cost of the problems
those rules were written to avoid.

### 9. Padding mask in batched embedding (found during implementation)

**Found 2026-09-29 by the new slow test
`test_padding_in_a_batch_does_not_change_a_vector`; fixed on this
branch.** mlx-embeddings 0.1.0 (the latest release) builds
embeddinggemma's additive padding mask and then casts it to
`embed_tokens.weight.dtype`. In the 8-bit model that weight is packed
`uint32`, so `-inf` becomes `0` and padding is never masked. Pooling
already skips padded positions, so the damage is in attention: every text
in a padded batch except the longest attends to padding tokens. A short
text embedded alone vs. batched with a long one: cosine **0.924** (0.9241
on one machine's GPU, 0.9245 on another's CPU). With the mask built in
float16: 0.999999.

In pyqmd, `index_content` embeds all of a document's chunks in one batch,
so in every multi-chunk document the shorter chunks (usually the last)
were embedded against padding. Queries are embedded one at a time and
were unaffected. Pre-existing since the MLX layer; this branch's slow
test exposed it.

Fix: `embed()` runs gemma3_text models through its own copy of
mlx-embeddings' forward pass (`_gemma3_text_embeds`), identical except
that the mask is built in the activation dtype (the quantized embedding's
`scales` dtype). An unpadded text gives the same vector as upstream's own
path (cosine 1.0). Other model types keep upstream's `__call__`.
Because that copy relies on 0.1.0's internals, mlx-embeddings is pinned
to exactly `==0.1.0` (it was `>=0.1.0`): moving the pin means checking the
new release against the copy, and deleting the copy if upstream fixed the
mask. The slow test guards the copy either way. mlx-lm, whose Gemma3
transformer block the copy also runs, stays floored (`>=0.31.3`); it also
drives expansion and reranking, so pinning it is a separate decision.

Two more problems in the same upstream code, **not fixed here, to be
measured first** (a follow-up):

- **Embedding scale.** The same cast truncates `hidden_size ** 0.5`
  (27.71) to 27 before scaling token embeddings: every vector, batched or
  not, is computed with a 2.6% smaller input scale than the reference
  model. Confirmed the cast; not yet measured what it does to vectors.
- **Sliding window.** When a mask is passed (pyqmd always passes one),
  every layer gets it, so the sliding-window layers (window 512) attend
  over the whole input. That only matters beyond about 512 tokens, which
  inputs can now reach with the 2048-token limit. What Node's llama.cpp
  does for embeddinggemma's bidirectional sliding window isn't confirmed
  yet.

The evidence for both should be Node's own vectors: a one-off bun script
that embeds the fixture strings from `node_expected.json` with Node's
model, compared against pyqmd's vectors with and without each change.
Cross-backend quantization (Q8_0 GGUF vs. MLX 8-bit) keeps them from
matching exactly, so a change is kept only if it moves pyqmd measurably
closer.

## What upgrading looks like

1. Upgrade to the release with this fix.
2. `pyqmd vsearch …`/`query …` print
   `Warning: N documents (100%) need embeddings. Run 'pyqmd embed' …`, and
   `pyqmd status` shows them as pending. Search still works on the old
   vectors meanwhile.
3. `pyqmd update` corrects any titles gap 3 affected.
4. `pyqmd embed` (no `--force`) re-embeds everything once.

The changelog entries for these commits say to run `pyqmd update` and
`pyqmd embed` after upgrading.

## Still different from Node (documented, unchanged)

- **Token-verified chunking** (`chunkDocumentByTokensWithLlm`): Node
  re-splits any char-estimated chunk that turns out longer than 900 real
  tokens. Deferred since the storage-layer spec, and still deferred. The
  2048 limit makes it matter less: a chunk would need more than twice its
  estimated token count before any of it went unembedded.
- **`QMD_EMBED_CONTEXT_SIZE`**: Node's env override for the embedding
  context isn't ported; nothing needs it.
- **Code fence chunking rules** (section 8).

## Audit: Node's index → embed → query path against pyqmd's

Read-only comparison against the pinned Node reference, 2026-09-29, done
to catch anything else that changes stored vectors before this release
(so indexes re-embed once, not twice).

**Changes stored vectors — included above:** gaps 1, 2, 3, 7, 8.

**Checked, same as Node:** break-point patterns and scores; the
best-cutoff decay; the `.md` title regex; embedding query format; the
empty-content skip; batch pooling (mlx_embeddings mean-pools with the
attention mask, then applies EmbeddingGemma's dense layers and
normalizes, as SentenceTransformers does); FTS BM25 weights
(1.5 / 4.0 / 1.0); query-time best-chunk sizes; skip-rerank scoring;
intent prepended to the rerank query.

**Not checked:** whether Node's GGUF embeddinggemma build applies the same
dense layers. That's a backend difference with no text-level test; only a
side-by-side vector comparison with a Node run would show it.

**Search-behaviour gaps found, out of scope here** (only the CRLF row
changes stored vectors, and only for CRLF files; each needs its own
decision, recorded in the roadmap):

| Gap                                                 | Node                                                                                                                            | pyqmd                                                                                                                                                                   | Recorded before this audit?                                                               |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| FTS path column                                     | indexes `collection/path` as a searchable column (BM25 weight 1.5)                                                              | stores the bare path, `UNINDEXED`: file and folder names never match `search`                                                                                           | No                                                                                        |
| `vsearch` pipeline                                  | expands the query, searches the original plus each `vec:`/`hyde:` line, keeps each file's best score, default `--min-score` 0.3 | one vector search of the original query, default `--min-score` 0                                                                                                        | No                                                                                        |
| Vector score                                        | `1 - distance`                                                                                                                  | `1 - distance / 2`: higher, compressed scores, so the same `--min-score` means something different                                                                      | No (COMMAND_STATUS attributes score differences to the backend only)                      |
| Default `-n`                                        | 5 for cli/md/csv/xml, 20 for json/files, every search command                                                                   | 20 for all three, any format (`DEFAULT_SEARCH_LIMIT`; CLAUDE.md's "10 for query" is stale)                                                                              | No                                                                                        |
| CRLF files (found while planning)                   | `readFileSync(path, "utf-8")` keeps `\r\n`                                                                                      | `Path.read_text` turns `\r\n` into `\n`: content, hashes, positions and vectors differ for CRLF files. Changes vectors, but fixing it later re-indexes only those files | No                                                                                        |
| Best-chunk term count                               | a query term repeated in the query counts once per repeat                                                                       | counted once (a set)                                                                                                                                                    | No (minor)                                                                                |
| Rerank blend, per-list depth (20 vs 40), list order | —                                                                                                                               | —                                                                                                                                                                       | Yes: roadmap, 2026-09-24 expansion spike (the blend is kept on purpose: it scores better) |

Rerank prompt and expansion prompt inputs were compared earlier (the
rerank fixtures, and the seeded-sampling spec), so they weren't re-audited.

## Second parity corpus: ConditionalQA

### Why scifact isn't enough

Scifact can barely see this spec's biggest gaps. Its documents are short
(median ~1.4 KB, ~360 tokens, one chunk; ~13% over ~2 KB), so the
512-token cut (gap 2) and the embed chunk size (gap 7) touch only its
long tail. Every document is a title line plus one abstract, with no
sections, so it never exercises heading-scored cuts or (later)
breadcrumbs. And `OVERLAP_THRESHOLD` in `test_quality.py` has been a
guess since the suite was written, for lack of a second real dataset to
tune it against.

### The dataset

[ConditionalQA](https://github.com/haitian-sun/ConditionalQA) (Sun et
al., 2021), `v1_0/`:

- **Corpus** (`documents.json`): 652 UK government guidance pages from
  gov.uk (e.g. "Child Tax Credit", "Apply to become a special guardian"),
  each `{title, url, contents}`, where `contents` is a list of one-element
  HTML strings: `<h1>` 3,456, `<h2>` 7,260, `<h3>` 2,354, `<h4>` 2, `<p>`
  30,511, `<li>` 23,113, `<tr>` 2,503 (cells already joined with `|`).
  645 of 652 pages have sections. Median page ~6.5 KB (2–3 embed chunks),
  90th percentile ~15 KB.
- **Questions**: each is a short personal `scenario` plus a `question`,
  with the `url` of the page that answers it, `answers`, `evidences`
  (the supporting sentences) and `not_answerable`. dev: 285 questions over
  59 pages; train: 2,338 over 436 pages. Every dev URL is in the corpus.
  URL slugs (`https://www.gov.uk/<slug>`) are unique and flat.
- **Licence**: the page text is gov.uk content under the Open Government
  Licence v3.0 (attribution required). The dataset repo ships a BSD-2
  licence file, but its README says the data is for NLP research use
  only. So, as with scifact: downloaded at prep time into gitignored
  `data/`, never committed. What does get committed is the Node snapshots
  (below), which quote short excerpts of pages and a few sample
  questions; `parity/README.md` credits the dataset and gov.uk (OGL).

### How it plugs in

Everything reuses the existing dataset-profile machinery; no suite code
assumes scifact beyond its defaults.

- **Prep script** `scripts/prepare_conditionalqa_corpus.py`, modelled on
  `prepare_scifact_corpus.py`: downloads `documents.json` and `dev.json`
  from the repo at a pinned commit, and writes `data/conditionalqa/`:
  - `corpus/<slug>.md` per page: `# <title>`, then each content item in
    order: `<h1>`→`##`, `<h2>`→`###`, `<h3>`→`####`, `<h4>`→`#####`
    (gov.uk uses `<h1>` for a guide's part titles, below the page title),
    `<p>` → a paragraph, consecutive `<li>` → a `- ` list, consecutive
    `<tr>` → one line per row, as given. Inline tags are stripped and
    HTML entities unescaped.
  - `queries.json`: one question per dev page, the first in file order,
    so 59 queries spread over 59 pages. Each entry is `{query_id,
query: <question>, intent: <scenario>}`.
  - `qrels.tsv`: `query-id`, `corpus-id` (the page slug), `score` 1:
    page-level relevance, the same format as scifact's.
- **Profile** `parity/datasets/conditionalqa.yaml` pointing at those
  three files.
- **Intent in the parity suite**: `load_queries` keeps an optional
  `intent` field; the quality test passes it to `Store.query(intent=…)`
  and the Node quality capture passes `--intent`. The scenario is what
  makes these questions answerable ("How long will it be before I hear
  back from the court?" means nothing without it), and scifact never
  exercises `--intent`. Profiles without intents behave as today.
- **Node baseline**: the user runs `capture_node_snapshots.py --phase
all` with this profile, which writes `parity/node_ref/conditionalqa/`.
  `.gitignore` ignores every `parity/node_ref/*` profile except scifact,
  so it gains `!parity/node_ref/conditionalqa/`. The user runs the
  model-input capture (`--phase inputs`, default profile; Testing) in the
  same sitting, so there is one manual capture step, not two.
- **Running it**: `just test-parity --dataset-config
parity/datasets/conditionalqa.yaml`. `just test-parity` alone still
  runs scifact.

### Rejected alternatives

- **MLDR, English** (MIT): long Wikipedia articles, but plain text with
  no headings, and a very large corpus that would need subsampling.
- **LongEmbed**: its synthetic needle/passkey sets test truncation
  directly, but they aren't realistic documents, and licences vary by
  component dataset.
- **DAPR**: includes ConditionalQA already, but strips the HTML, so the
  headings are lost.
- **Chunk-level relevance** from ConditionalQA's evidence sentences:
  possible later (which chunk should win), but the quality test scores
  documents, so page-level qrels are enough here.
- **All 285 dev questions**: many pages have several near-identical
  questions, and each query costs a full `query` run per pass in the Node
  capture. One per page gives 59 independent queries.

## Rejected alternatives

- **Port Node's legacy-fingerprint adoption**
  (`maybeAdoptLegacyEmbeddingFingerprint`): it embeds one sample chunk with
  the current format and, if the stored vector matches, stamps every
  empty-fingerprint row as current. Every existing pyqmd vector was
  embedded with `title: none` and a 512-token cut, so the check could never
  pass. It's also exported but not called anywhere in Node's CLI.
- **Re-embed automatically after upgrading**: writes to the user's index
  without being asked. The warning and `status` tell the user; they run
  `pyqmd embed`.
- **Pass one title per text to `embed()`**: more general, but every current
  caller embeds one document at a time. One `title` per call is simpler.
- **Use the stored `documents.title` for embeddings**: content is embedded
  once per hash, shared by every path with that content; Node derives the
  title from `MIN(path)` at embed time, and so does this.
- **Ship breadcrumbs in the same change**: the fix is a bug fix whatever
  happens with breadcrumbs, and breadcrumbs needs this as its baseline to
  be measured against.

## Testing

Fast unit tests (`-m "not slow"`):

- **Titles** (`tests/store/test_title.py`): `.md` heading, `.md` "Notes"
  special case, `.MD` upper case, `.org` `#+TITLE`, `.org` `*` heading,
  `.py` with a leading `# comment` → file name, `.markdown` → file name,
  no extension, `.env`.
- **`update` title refresh**: re-scan after changing only the extraction
  outcome (e.g. a `.py` file indexed under the old rule) → title updated,
  counted as updated, FTS finds the new title and not the old one.
- **Title reaches the embedder**: a recording fake `embed_fn` sees
  `kind="document"` and the extracted title for each `index_content` call.
- **`embed()` formatting and limit**: with a stub tokenizer/model, the
  document texts are formatted with the title and truncated at
  `min(2048, model_max_length)`.
- **Fingerprint**: the value is stable for a given model and changes when
  the formatted probe changes; `index_content` writes it; an
  empty-fingerprint row with unchanged positions is re-embedded; pending
  counts (`count_pending_embed`, `get_status_counts`) include
  empty-fingerprint hashes and exclude current ones.
- **Health warning**: `vsearch`/`query` CLI tests for the ≥10% warning, the
  <10% tip, the 14-day tip, and silence when nothing is pending or stale.
- **Embed-time chunk size**: `embedding_chunks` uses 2700/405/600;
  `Store.query`'s best-chunk selection still uses 3600/540/800.

### Model-input parity tests (new test class)

The bugs in this spec were all in what pyqmd feeds the model, and every
existing test compared outputs. These tests compare inputs, as exact
strings, against Node's.

- **Fixtures** (`parity/fixtures/embedding_inputs/`), small and synthetic,
  each chosen to hit one rule:
  - a long `.md` with `#`/`##`/`###` sections, well over 2700 chars
    (several chunks, heading-scored cuts);
  - a `.md` whose first heading is `Notes`;
  - a `.md` with no heading (file-name title);
  - a `.org` with `#+TITLE`, and one with only `*` headings;
  - a `.py` whose first line is `# comment`.

  All fence-free (section 8), and all kept short enough per chunk
  (plain English prose) that Node's token re-split never fires, so the
  capture needs no model.

- **Node capture**: a new `--phase inputs` in
  `parity/capture_node_snapshots.py` runs a small bun script against the
  pinned Node checkout. For each fixture it records `extractTitle`, the
  `chunkDocumentAsync` chunks at 2700/405/600 (text and pos), and each
  chunk's `formatDocForEmbedding(text, title, model)` string; plus
  `formatQueryForEmbedding` for a few queries, and
  `getEmbeddingFingerprint` for Node's default model (so the fingerprint
  port is checked against Node's own value). Written, with the Node commit
  it came from, to `parity/fixtures/embedding_inputs/node_expected.json`,
  next to the fixtures it describes (not under `node_ref/<profile>/`: it
  doesn't depend on a dataset profile). A test checks that commit matches
  scifact's `COMMIT.txt`, so the pinned Node reference stays single.
  Capture stays manual: the user runs it.
- **pyqmd side**: a test builds the same records through the real code
  path: `extract_title`, `embedding_chunks`, and the formatter `embed()`
  applies, factored out as `format_embedding_inputs(texts, model, kind,
title)` so the test and `embed()` share it. The test asserts exact
  equality per fixture, per chunk.
- **Call-site test**: a recording fake `embed_fn` on a real `Store`
  confirms `index_content` hands the embedder exactly the
  `embedding_chunks` texts and the `extract_title` title. That's the link
  that broke for gap 1: the formatter was right, the caller never passed
  a title.

Slow tests (`-m slow`, real model):

- Two texts that differ only after token ~600 embed to different vectors
  (they don't today: both are cut at 512).
- Padding doesn't change a vector: a short text embedded alone and in a
  batch with a long one gives the same vector (cosine ≥ 0.9999).

Parity suite:

- `just test-parity` stays green, on scifact and on ConditionalQA. The
  quality test builds its own isolated index, so it re-embeds with the
  new input. The CLI-flow snapshots include Node's stderr, health lines
  included, and `test_cli_flow_step_text_matches_node` compares both
  stdout and stderr (after path normalization); it passes on both corpora
  with the change.

## Verification

- The quality test measures `query` only. Before and after the change,
  run it on both corpora and record the pyqmd metrics it prints (Node's
  are the pinned baselines). The "before" run is on `main` plus the new
  corpus and may fail the ConditionalQA bar; that's a finding, recorded,
  not something to work around. The "after" run must pass on both.
- Scifact mostly measures the title change: every document starts with a
  `# Title` line, and only its long tail is affected by gaps 2 and 7.
  ConditionalQA measures gaps 2 and 7 (multi-chunk pages, text well past
  token 512) and the `.md` title rule on real headings.
- Record the `query` results on ConditionalQA against the overlap and
  quality numbers, and note in the roadmap whether `OVERLAP_THRESHOLD`
  (0.7, a guess) still looks right with a second dataset.
- Record `embed` wall time on both corpora before and after, for the cost
  of gap 2.
- No command in this work writes to the user's own index. Re-embedding it
  is the user's call after the release.

## Results

Measured with `uv run pytest parity/ -s --durations=5`. The hashes are
commits on the PR branch. "Before" is 5433605 (`main` plus the new corpus and the capture tooling, no fixes);
"after" is d39e837 (every fix, including section 9). Both runs of a
corpus are on the same machine, one after the other, so the times compare.

### SciFact (30 queries, qrels mode)

Both runs: 62 passed, 1 skipped (the agreement-mode test, which only runs
without qrels).

| Metric    | Before | After  | Change | Node mean | Margin |
| --------- | ------ | ------ | ------ | --------- | ------ |
| MRR       | 0.7880 | 0.8034 | +0.015 | 0.7579    | 0.0623 |
| nDCG@10   | 0.7995 | 0.8210 | +0.022 | 0.7781    | 0.0523 |
| Recall@10 | 0.8667 | 0.9000 | +0.033 | 0.8722    | 0.0461 |

Recall@10 moves from below Node's mean to above it, but on 30 queries
+0.033 is one more query in the top 10 (27 of 30, was 26), and the rank
metrics moved by about one Node sigma: no regression, slightly better, not
a proven win. Index + embed (the quality test's setup) took 105.5 s before
and 112.4 s after (+7%); SciFact's abstracts are short, so few of them
gain chunks from the smaller embed-time chunk size.

### ConditionalQA (59 queries, qrels mode)

"After" ran at e2e58d1, which has the same source as d39e837 (the commits
in between add only snapshots, tests and docs). Both runs: 62 passed,
1 skipped.

| Metric    | Before | After  | Change | Node mean | Margin |
| --------- | ------ | ------ | ------ | --------- | ------ |
| MRR       | 0.7278 | 0.7374 | +0.010 | 0.5703    | 0.0501 |
| nDCG@10   | 0.7654 | 0.7686 | +0.003 | 0.6368    | 0.0439 |
| Recall@10 | 0.8814 | 0.8644 | −0.017 | 0.8373    | 0.0392 |

Mean top-10 overlap with Node's pass-1 rankings: 0.583 before, 0.614
after.

The rank metrics moved by less than one Node sigma. Recall@10's −0.017 is
exactly one query (51 of 59 in the top 10, was 52), and it stays above
Node's mean. So this is flat: no measurable quality change either way.
pyqmd was already well above Node on this corpus before any fix (MRR
+0.16, about three margins). That gap predates this branch, so it isn't
from embedding inputs; query expansion or reranking are the likelier
places to look.

The overlap rose by 0.031, the one clear sign that pyqmd now ranks more
like Node here. Index + embed (the quality test's setup) took 32.0 s
before and 131.0 s after (about 4×). ConditionalQA's gov.uk pages are
long, so both changes bite here: chunks that used to be cut at 512 tokens
are now embedded up to 2048, and the smaller embed-time chunk size makes
more chunks. How the 4× splits between the two wasn't measured.

**`OVERLAP_THRESHOLD` (0.7).** It applies only in agreement mode (no
qrels), so neither run was gated on it. Had it been, ConditionalQA would
have failed both before (0.583) and after (0.614), even though pyqmd
scores well above Node on every qrels metric. The threshold compares
against a single Node pass, and Node's own passes vary (MRR 0.53–0.60
across the 30), so its pass-to-pass overlap is also below 1 and hasn't
been measured. The threshold is left unchanged on this branch. Before
relying on agreement mode for a real corpus, measure Node's own overlap
between passes and set the threshold relative to that.

## Commits and release

Order matters: the new corpus and the capture tooling land first, the
user runs the Node captures once, and the "before" metrics are recorded,
all before any fix commit.

Released as a patch (`fix`), one changelog entry per commit, e.g.:

- `test(parity): add ConditionalQA as a second parity corpus, with intents`
- `test(parity): capture Node's embedding inputs for fixture documents`
  (the `--phase inputs` tooling and fixtures)
- `test(parity): pin Node's ConditionalQA and embedding-input snapshots`
  (after the user's capture)
- `fix(store): extract titles by file type, as Node does`
- `fix(store): refresh stored titles when only the title changes`
- `fix(llm): embed chunks with their document title and up to 2048 tokens`
- `fix(store): chunk for embedding at Node's embed-time size`
- `test(parity): compare embedding inputs with Node's, string for string`
- `fix(store): fingerprint embeddings so format changes re-embed`
- `fix(cli): warn on vsearch/query when embeddings are pending or stale`
- `fix(llm): mask padding when embedding a batch of chunks` (section 9)
- `fix(config): pin mlx-embeddings to exactly 0.1.0` (section 9)
- `docs(parity): credit SciFact's sources and licences` (both built-in
  corpora are credited alike; their snapshots quote excerpts, the corpora
  themselves are never committed)
- `docs(specs): publish the embedding input parity spec`

`COMMAND_STATUS.md` rows for `update`, `embed`, `status`, `vsearch` and
`query` get updated in the same branch.

## Follow-up: breadcrumbs

With this in place, breadcrumbs becomes an experiment, measured against
this baseline: embed each chunk as `title: <doc title> > <h2> > <h3> |
text: …`, bench it against the Node-matching format on ConditionalQA
(scifact has no sections, so breadcrumbs would change nothing there), and
ship it only if it clearly wins. The fingerprint above means shipping it
would re-embed indexes on its own. Either way, the result goes in the
roadmap.
