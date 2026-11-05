"""Download a BEIR retrieval dataset (MTEB copy on the Hugging Face Hub) and convert its
corpus into markdown files that `pyqmd collection add` can index.

Writes the same layout as prepare_scifact_corpus.py: <output>/corpus/<id>.md,
<output>/queries.jsonl and <output>/qrels-test.tsv (query-id, corpus-id, score).
Keep the output outside Dropbox: CQADupStack's unix forum alone is 47,000 files.

Usage: uv run scripts/prepare_beir_corpus.py nfcorpus --output ~/tmp/beir/nfcorpus
       uv run scripts/prepare_beir_corpus.py cqadupstack-unix --output ~/tmp/beir/cqadupstack-unix

Licences (check before redistributing; this script only downloads for local use):
NFCorpus and CQADupStack are CC BY-SA 4.0 in BEIR.
"""

import argparse
import shutil
from pathlib import Path

from prepare_scifact_corpus import write_corpus_markdown

DATASETS = {
    "nfcorpus": "mteb/nfcorpus",
    "cqadupstack-unix": "mteb/cqadupstack-unix",
    "cqadupstack-programmers": "mteb/cqadupstack-programmers",
    "cqadupstack-webmasters": "mteb/cqadupstack-webmasters",
}


def prepare(name: str, output: Path) -> int:
    from huggingface_hub import hf_hub_download

    repo = DATASETS[name]

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(repo, filename, repo_type="dataset"))

    count = write_corpus_markdown(fetch("corpus.jsonl"), output / "corpus")
    shutil.copyfile(fetch("queries.jsonl"), output / "queries.jsonl")
    shutil.copyfile(fetch("qrels/test.tsv"), output / "qrels-test.tsv")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(args.dataset, args.output)
    print(f"{args.dataset}: wrote {count} documents to {args.output / 'corpus'}")


if __name__ == "__main__":
    main()
