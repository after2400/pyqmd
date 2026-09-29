"""Download ConditionalQA (v1_0) and convert it into the files the
`conditionalqa` parity profile reads: one markdown file per gov.uk page,
a queries file (question + scenario as intent, one per dev page) and
page-level qrels.

Usage: uv run scripts/prepare_conditionalqa_corpus.py [--output data/conditionalqa]

The page text is gov.uk content under the Open Government Licence v3.0;
the dataset's README restricts it to research use. It is downloaded into
gitignored data/ and never committed. See
docs/specs/2026-09-29-embedding-input-parity-design.md.
"""

import argparse
import csv
import html
import json
import re
import shutil
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "conditionalqa"

# Pinned so a re-run always produces the same corpus.
DATASET_COMMIT = "77bd295952daf415548b3244db10880d3d55cfe0"
BASE_URL = f"https://raw.githubusercontent.com/haitian-sun/ConditionalQA/{DATASET_COMMIT}/v1_0"
PAGE_URL_PREFIX = "https://www.gov.uk/"

_ITEM_RE = re.compile(r"^<(h[1-4]|p|li|tr)>(.*)</\1>$", re.DOTALL)
# gov.uk uses <h1> for a guide's part titles, below the page title (#).
_HEADING_MARKS = {"h1": "##", "h2": "###", "h3": "####", "h4": "#####"}


def page_slug(url: str) -> str:
    """'https://www.gov.uk/child-tax-credit' -> 'child-tax-credit'."""
    slug = url[len(PAGE_URL_PREFIX) :] if url.startswith(PAGE_URL_PREFIX) else ""
    if not slug or "/" in slug:
        raise ValueError(f"not a flat gov.uk page URL: {url!r}")
    return slug


def _parse_item(item: str) -> tuple[str, str]:
    match = _ITEM_RE.match(item.strip())
    if match is None:
        raise ValueError(f"unexpected content item: {item[:80]!r}")
    tag, inner = match.groups()
    return tag, " ".join(html.unescape(inner).split())


def page_to_markdown(page: dict) -> str:
    blocks: list[str] = [f"# {page['title'].strip()}"]
    run_tag: str | None = None
    run: list[str] = []

    def flush() -> None:
        nonlocal run_tag
        if run:
            prefix = "- " if run_tag == "li" else ""
            blocks.append("\n".join(prefix + line for line in run))
            run.clear()
        run_tag = None

    for item in page["contents"]:
        tag, text = _parse_item(item)
        if not text:
            continue
        if tag in ("li", "tr"):
            if tag != run_tag:
                flush()
                run_tag = tag
            run.append(text)
            continue
        flush()
        blocks.append(f"{_HEADING_MARKS[tag]} {text}" if tag in _HEADING_MARKS else text)
    flush()
    return "\n\n".join(blocks) + "\n"


def select_queries(questions: list[dict]) -> list[dict]:
    """One question per page, the first in file order."""
    seen: set[str] = set()
    selected: list[dict] = []
    for question in questions:
        slug = page_slug(question["url"])
        if slug in seen:
            continue
        seen.add(slug)
        entry = {"query_id": question["id"], "query": question["question"].strip()}
        scenario = question["scenario"].strip()
        if scenario:
            entry["intent"] = scenario
        entry["page"] = slug
        selected.append(entry)
    return selected


def write_dataset(
    documents: list[dict], questions: list[dict], output_dir: Path
) -> tuple[int, int]:
    """Write corpus/, queries.json and qrels.tsv under output_dir. Returns
    (page count, query count)."""
    corpus_dir = output_dir / "corpus"
    shutil.rmtree(corpus_dir, ignore_errors=True)
    corpus_dir.mkdir(parents=True)
    slugs: set[str] = set()
    for page in documents:
        slug = page_slug(page["url"])
        slugs.add(slug)
        (corpus_dir / f"{slug}.md").write_text(page_to_markdown(page), encoding="utf-8")

    selected = select_queries(questions)
    missing = [q["page"] for q in selected if q["page"] not in slugs]
    if missing:
        raise ValueError(f"questions reference pages missing from the corpus: {missing}")

    queries = [{k: v for k, v in q.items() if k != "page"} for q in selected]
    (output_dir / "queries.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "qrels.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for q in selected:
            writer.writerow([q["query_id"], q["page"], 1])
    return len(slugs), len(selected)


def _download(name: str, raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    if not path.exists():
        urlretrieve(f"{BASE_URL}/{name}", path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ConditionalQA as a parity profile")
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR), help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    documents = _download("documents.json", output_dir / "raw")
    questions = _download("dev.json", output_dir / "raw")
    pages, queries = write_dataset(documents, questions, output_dir)
    print(f"Wrote {pages} markdown pages to {output_dir / 'corpus'}")
    print(f"Wrote {queries} queries (one per dev page) and page-level qrels to {output_dir}")
    print("Page text: gov.uk, Open Government Licence v3.0. Dataset: research use only.")


if __name__ == "__main__":
    main()
