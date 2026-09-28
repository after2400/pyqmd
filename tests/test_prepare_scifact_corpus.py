import json

from prepare_scifact_corpus import corpus_doc_to_markdown, write_corpus_markdown


def test_corpus_doc_to_markdown_includes_title_and_text():
    doc = {"_id": "4983", "title": "Cancer Biology", "text": "Some passage about cancer."}

    markdown = corpus_doc_to_markdown(doc)

    assert markdown.startswith("# Cancer Biology\n\n")
    assert "Some passage about cancer." in markdown


def test_corpus_doc_to_markdown_handles_missing_title():
    doc = {"_id": "1", "title": "", "text": "No title here."}

    markdown = corpus_doc_to_markdown(doc)

    assert markdown == "No title here.\n"


def test_write_corpus_markdown_creates_one_file_per_doc(tmp_path):
    jsonl_path = tmp_path / "corpus.jsonl"
    jsonl_path.write_text(
        json.dumps({"_id": "1", "title": "A", "text": "text a"})
        + "\n"
        + json.dumps({"_id": "2", "title": "B", "text": "text b"})
        + "\n"
    )
    output_dir = tmp_path / "out"

    count = write_corpus_markdown(jsonl_path, output_dir)

    assert count == 2
    assert (output_dir / "1.md").exists()
    assert (output_dir / "2.md").read_text().startswith("# B")
