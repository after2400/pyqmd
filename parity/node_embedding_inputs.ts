// Records Node qmd's embedding inputs for pyqmd's model-input parity test
// (tests/parity/test_embedding_inputs.py): per fixture, extractTitle, the
// embed-time chunks (chunkDocumentByTokensWithLlm's character pass:
// chunkDocumentAsync at 3 chars/token), and each chunk's
// formatDocForEmbedding string; plus formatQueryForEmbedding for
// queries.json and getEmbeddingFingerprint for Node's default model.
// Run only by `python -m parity.capture_node_snapshots --phase inputs`
// (manual). Usage: bun parity/node_embedding_inputs.ts <qmd-repo-root> <fixtures-dir>
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";

const [qmdRoot, fixturesDir] = process.argv.slice(2);
if (!qmdRoot || !fixturesDir) {
  console.error("usage: bun node_embedding_inputs.ts <qmd-repo-root> <fixtures-dir>");
  process.exit(2);
}
const store = await import(join(qmdRoot, "src", "store.ts"));
const llm = await import(join(qmdRoot, "src", "llm.ts"));

const model: string = llm.DEFAULT_EMBED_MODEL_URI;
const CHARS_PER_TOKEN = 3; // chunkDocumentByTokensWithLlm's avgCharsPerToken
const SKIP = new Set(["node_expected.json", "queries.json"]);

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) return walk(full);
    return SKIP.has(name) ? [] : [full];
  });
}

const documents: Record<string, unknown> = {};
for (const full of walk(fixturesDir).sort()) {
  const relPath = relative(fixturesDir, full).split(sep).join("/");
  const body = readFileSync(full, "utf-8");
  const title = store.extractTitle(body, relPath);
  const chunks: { text: string; pos: number }[] = await store.chunkDocumentAsync(
    body,
    store.CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN,
    store.CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN,
    store.CHUNK_WINDOW_TOKENS * CHARS_PER_TOKEN,
    relPath,
    "regex",
  );
  documents[relPath] = {
    title,
    chunks: chunks.map((c) => ({
      pos: c.pos,
      text: c.text,
      formatted: store.formatDocForEmbedding(c.text, title, model),
    })),
  };
}

const queryList: string[] = JSON.parse(readFileSync(join(fixturesDir, "queries.json"), "utf-8"));
const queries = Object.fromEntries(
  queryList.map((q) => [q, store.formatQueryForEmbedding(q, model)]),
);

console.log(
  JSON.stringify(
    { model, fingerprint: store.getEmbeddingFingerprint(model), documents, queries },
    null,
    2,
  ),
);
