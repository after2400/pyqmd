import csv
import json

import pytest
from prepare_conditionalqa_corpus import page_slug, page_to_markdown, select_queries, write_dataset


def test_page_slug_strips_the_gov_uk_prefix():
    assert page_slug("https://www.gov.uk/child-tax-credit") == "child-tax-credit"


@pytest.mark.parametrize(
    "url", ["https://example.com/x", "https://www.gov.uk/a/b", "https://www.gov.uk/"]
)
def test_page_slug_rejects_anything_but_a_flat_gov_uk_page(url):
    with pytest.raises(ValueError):
        page_slug(url)


def test_page_to_markdown_maps_headings_lists_rows_and_paragraphs():
    page = {
        "title": "Child Tax Credit",
        "url": "https://www.gov.uk/child-tax-credit",
        "contents": [
            "<h1>Overview</h1>",
            "<p>You can claim.</p>",
            "<li>one</li>",
            "<li>two</li>",
            "<h2>What you’ll get</h2>",
            "<tr>Element | Amount</tr>",
            "<tr>Basic | £545</tr>",
            "<h3>Rates</h3>",
            "<p>Line one\nline two</p>",
            "<h4>Small</h4>",
            "<p>A &amp; B</p>",
        ],
    }

    assert page_to_markdown(page) == (
        "# Child Tax Credit\n\n"
        "## Overview\n\n"
        "You can claim.\n\n"
        "- one\n- two\n\n"
        "### What you’ll get\n\n"
        "Element | Amount\nBasic | £545\n\n"
        "#### Rates\n\n"
        "Line one line two\n\n"
        "##### Small\n\n"
        "A & B\n"
    )


def test_page_to_markdown_rejects_an_unknown_item():
    with pytest.raises(ValueError, match="unexpected content item"):
        page_to_markdown({"title": "T", "url": "u", "contents": ["<div>x</div>"]})


def test_select_queries_keeps_the_first_question_per_page():
    questions = [
        {"id": "dev-0", "url": "https://www.gov.uk/a", "question": " Q1? ", "scenario": " S1 "},
        {"id": "dev-1", "url": "https://www.gov.uk/a", "question": "Q2?", "scenario": "S2"},
        {"id": "dev-2", "url": "https://www.gov.uk/b", "question": "Q3?", "scenario": ""},
    ]

    assert select_queries(questions) == [
        {"query_id": "dev-0", "query": "Q1?", "intent": "S1", "page": "a"},
        {"query_id": "dev-2", "query": "Q3?", "page": "b"},
    ]


def test_write_dataset_writes_corpus_queries_and_qrels(tmp_path):
    documents = [
        {"title": "A page", "url": "https://www.gov.uk/a", "contents": ["<p>alpha</p>"]},
        {"title": "B page", "url": "https://www.gov.uk/b", "contents": ["<p>beta</p>"]},
    ]
    questions = [
        {"id": "dev-0", "url": "https://www.gov.uk/b", "question": "Q?", "scenario": "S"},
    ]
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "stale.md").write_text("left over")

    counts = write_dataset(documents, questions, tmp_path)

    assert counts == (2, 1)
    assert sorted(p.name for p in (tmp_path / "corpus").iterdir()) == ["a.md", "b.md"]
    assert (tmp_path / "corpus" / "b.md").read_text() == "# B page\n\nbeta\n"
    assert json.loads((tmp_path / "queries.json").read_text()) == [
        {"query_id": "dev-0", "query": "Q?", "intent": "S"}
    ]
    with (tmp_path / "qrels.tsv").open(newline="") as f:
        assert list(csv.DictReader(f, delimiter="\t")) == [
            {"query-id": "dev-0", "corpus-id": "b", "score": "1"}
        ]


def test_write_dataset_rejects_a_question_whose_page_is_missing(tmp_path):
    documents = [{"title": "A", "url": "https://www.gov.uk/a", "contents": ["<p>x</p>"]}]
    questions = [{"id": "dev-0", "url": "https://www.gov.uk/zz", "question": "Q", "scenario": ""}]

    with pytest.raises(ValueError, match="zz"):
        write_dataset(documents, questions, tmp_path)
