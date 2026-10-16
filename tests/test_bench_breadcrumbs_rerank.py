import json
from types import SimpleNamespace

import pytest
from bench_breadcrumbs_rerank import (
    Logger,
    format_progress,
    load_checkpoint,
    rerank_verdict,
    reranked_pages,
    run_questions,
)


def fake_store(pages_by_query, calls):
    def query(q, **kwargs):
        calls.append((q, kwargs))
        return [SimpleNamespace(file=f"{p}.md") for p in pages_by_query[q]]

    return SimpleNamespace(query=query)


def questions(*pairs):
    return [SimpleNamespace(query=q, page=p) for q, p in pairs]


HEADER = {"model": "m", "limit": 40, "pages": 3}


def test_reranked_pages_asks_the_store_for_the_full_pipeline_with_reranking_on():
    calls = []
    store = fake_store({"how to": ["guide", "notes"]}, calls)
    assert reranked_pages(store, "how to") == ["guide", "notes"]
    assert calls == [("how to", {"limit": 40})]


def test_run_questions_scores_both_arms_and_writes_a_checkpoint(tmp_path):
    off_calls, on_calls = [], []
    stores = {
        "off": fake_store({"q1": ["a", "gold"], "q2": ["gold"]}, off_calls),
        "on": fake_store({"q1": ["gold", "a"], "q2": ["gold"]}, on_calls),
    }
    qs = questions(("q1", "gold"), ("q2", "gold"))
    path = tmp_path / "ck.jsonl"
    results = run_questions(stores, qs, path, HEADER, Logger(None), log_every=1)
    assert list(results["off"]["page_mrr"]) == [0.5, 1.0]
    assert list(results["on"]["page_mrr"]) == [1.0, 1.0]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == HEADER
    assert len(lines) == 3


def test_run_questions_resumes_and_does_not_ask_finished_questions_again(tmp_path):
    qs = questions(("q1", "gold"), ("q2", "gold"))
    path = tmp_path / "ck.jsonl"
    answers = {"q1": ["gold"], "q2": ["x", "gold"]}
    first = {"off": fake_store(answers, []), "on": fake_store(answers, [])}
    run_questions(first, qs[:1], path, HEADER, Logger(None), log_every=1)

    off_calls, on_calls = [], []
    second = {"off": fake_store(answers, off_calls), "on": fake_store(answers, on_calls)}
    results = run_questions(second, qs, path, HEADER, Logger(None), log_every=1)
    assert [c[0] for c in off_calls] == ["q2"] and [c[0] for c in on_calls] == ["q2"]
    assert list(results["off"]["page_mrr"]) == [1.0, 0.5]


def test_a_checkpoint_from_a_different_setup_is_refused(tmp_path):
    path = tmp_path / "ck.jsonl"
    path.write_text(json.dumps({**HEADER, "model": "other"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path, HEADER)


def test_a_checkpoint_for_a_different_question_is_refused(tmp_path):
    path = tmp_path / "ck.jsonl"
    answers = {"q1": ["gold"], "q2": ["gold"]}
    stores = {"off": fake_store(answers, []), "on": fake_store(answers, [])}
    run_questions(stores, questions(("q1", "gold")), path, HEADER, Logger(None), log_every=1)
    with pytest.raises(ValueError, match="question"):
        run_questions(stores, questions(("q2", "gold")), path, HEADER, Logger(None), log_every=1)


def test_a_torn_last_line_is_dropped_on_resume(tmp_path):
    path = tmp_path / "ck.jsonl"
    answers = {"q1": ["gold"]}
    stores = {"off": fake_store(answers, []), "on": fake_store(answers, [])}
    run_questions(stores, questions(("q1", "gold")), path, HEADER, Logger(None), log_every=1)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"i": 1, "q": "q2", "off": ["go')  # killed mid-write
    done = load_checkpoint(path, HEADER)
    assert list(done) == [0]


def test_logger_prints_and_appends_to_the_file(tmp_path, capsys):
    log = Logger(tmp_path / "run.log")
    log.say("first")
    log.say("second")
    assert capsys.readouterr().out == "first\nsecond\n"
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == "first\nsecond\n"


def test_format_progress_shows_rate_and_time_left():
    text = format_progress(done=100, total=400, elapsed=200.0, resumed=0)
    assert "100/400" in text and "2.0 s/question" in text and "10 min left" in text


def test_format_progress_rates_only_the_questions_done_this_session():
    text = format_progress(done=300, total=400, elapsed=100.0, resumed=200)
    assert "1.0 s/question" in text and "2 min left" in text


def test_rerank_verdict_never_flips_the_default_by_itself():
    ci = {"page_ndcg10": (0.02, 0.01, 0.03), "page_mrr": (0.0, -0.01, 0.01)}
    code, text = rerank_verdict(ci)
    assert code == 1 and "default question" in text


def test_rerank_verdict_two_and_three():
    flat = {"page_ndcg10": (0.0, -0.01, 0.01), "page_mrr": (0.0, -0.01, 0.01)}
    assert rerank_verdict(flat)[0] == 2
    worse = {"page_ndcg10": (0.0, -0.01, 0.01), "page_mrr": (-0.02, -0.03, -0.01)}
    assert rerank_verdict(worse)[0] == 3
