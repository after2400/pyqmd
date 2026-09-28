"""Download BEIR SciFact and convert its corpus into markdown files for `qmd collection add`.

Usage: cd python && uv run scripts/prepare_scifact_corpus.py [--output data/scifact]
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def corpus_doc_to_markdown(doc: dict) -> str:
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    if title:
        return f"# {title}\n\n{text}\n"
    return f"{text}\n"


def write_corpus_markdown(corpus_jsonl_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            markdown = corpus_doc_to_markdown(doc)
            (output_dir / f"{doc['_id']}.md").write_text(markdown, encoding="utf-8")
            count += 1
    return count


def download_and_extract(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "scifact.zip"
    if not zip_path.exists():
        urlretrieve(SCIFACT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return dest_dir / "scifact"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BEIR SciFact as a qmd collection")
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    scifact_dir = download_and_extract(output_dir / "raw")

    corpus_dir = output_dir / "corpus"
    count = write_corpus_markdown(scifact_dir / "corpus.jsonl", corpus_dir)
    print(f"Wrote {count} markdown documents to {corpus_dir}")

    shutil.copy(scifact_dir / "queries.jsonl", output_dir / "queries.jsonl")
    shutil.copy(scifact_dir / "qrels" / "test.tsv", output_dir / "qrels-test.tsv")
    print(f"Copied queries.jsonl and qrels-test.tsv to {output_dir}")
    print()
    print("Next (run manually -- qmd never indexes automatically):")
    print(f"  qmd collection add {corpus_dir} --name scifact")
    print("  qmd embed --collection scifact")


if __name__ == "__main__":
    main()
