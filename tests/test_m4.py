"""Tests for Module 4: Evaluation."""
import sys, os
import json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report, EvalResult


@pytest.fixture
def mock_ragas(monkeypatch):
    """Use real RAGAS Result/Dataset conversion without making paid API calls."""
    import ragas
    from ragas.evaluation import Result
    from datasets import Dataset

    calls = []

    def evaluate(dataset, metrics):
        calls.append((dataset, metrics))
        scores = Dataset.from_dict({
            "faithfulness": [0.8, 0.4][:len(dataset)],
            "answer_relevancy": [0.6, 0.2][:len(dataset)],
            "context_precision": [0.9, 0.5][:len(dataset)],
            "context_recall": [0.7, 0.3][:len(dataset)],
        })
        return Result(scores=scores, dataset=dataset)

    monkeypatch.setattr(ragas, "evaluate", evaluate)
    return calls

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics(mock_ragas):
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


def test_evaluate_aggregates_and_preserves_rows(mock_ragas):
    result = evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1", "c2"], ["c3"]], ["gt1", "gt2"])
    assert "error" not in result
    assert result["faithfulness"] == pytest.approx(0.6)
    assert result["answer_relevancy"] == pytest.approx(0.4)
    assert result["context_precision"] == pytest.approx(0.7)
    assert result["context_recall"] == pytest.approx(0.5)
    rows = result["per_question"]
    assert len(rows) == 2 and all(isinstance(row, EvalResult) for row in rows)
    assert rows[0].question == "q1" and rows[0].answer == "a1"
    assert rows[0].contexts == ["c1", "c2"] and rows[0].ground_truth == "gt1"
    assert rows[1].faithfulness == pytest.approx(0.4)
    dataset, metrics = mock_ragas[0]
    assert dataset.column_names == ["question", "answer", "contexts", "ground_truth"]
    assert {metric.name for metric in metrics} == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    }


def test_evaluate_empty_input_skips_ragas(mock_ragas):
    result = evaluate_ragas([], [], [], [])
    assert result["per_question"] == [] and result["faithfulness"] == 0.0
    assert "error" not in result
    assert mock_ragas == []


def test_evaluate_rejects_misaligned_inputs(mock_ragas):
    result = evaluate_ragas(["q"], [], [["c"]], ["gt"])
    assert result["error"] == "ValueError" and result["per_question"] == []
    assert mock_ragas == []


def test_evaluate_api_failure_is_explicit(monkeypatch):
    import ragas

    def fail(*args, **kwargs):
        raise RuntimeError("Evaluation service unavailable")

    monkeypatch.setattr(ragas, "evaluate", fail)
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["error"] == "RuntimeError"
    assert result["per_question"] == []
    assert all(result[name] == 0.0 for name in (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ))


def test_evaluate_undefined_metric_is_not_a_valid_score(monkeypatch):
    import ragas
    import pandas as pd
    from types import SimpleNamespace

    frame = pd.DataFrame([{
        "question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt",
        "faithfulness": float("nan"), "answer_relevancy": 0.8,
        "context_precision": 0.9, "context_recall": 0.7,
    }])
    monkeypatch.setattr(ragas, "evaluate", lambda *args, **kwargs: SimpleNamespace(to_pandas=lambda: frame))
    result = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["error"] == "ValueError" and result["per_question"] == []


@pytest.mark.parametrize("metric,fix", [
    ("faithfulness", "temperature"),
    ("context_recall", "BM25"),
    ("context_precision", "reranking"),
    ("answer_relevancy", "prompt"),
])
def test_each_diagnostic_branch(metric, fix):
    scores = dict.fromkeys(("faithfulness", "answer_relevancy", "context_precision", "context_recall"), 0.9)
    scores[metric] = 0.1
    result = EvalResult("q", "a", ["c"], "gt", **scores)
    failure = failure_analysis([result])[0]
    assert failure["worst_metric"] == metric
    assert failure["score"] == pytest.approx(0.7)
    assert failure["diagnosis"]
    assert fix in failure["suggested_fix"]


def test_failures_sorted_by_average_and_limited():
    results = [
        EvalResult("good", "a", [], "gt", 0.9, 0.9, 0.9, 0.9),
        EvalResult("worst", "a", [], "gt", 0.1, 0.2, 0.3, 0.4),
        EvalResult("middle", "a", [], "gt", 0.5, 0.6, 0.7, 0.8),
    ]
    failures = failure_analysis(results, bottom_n=2)
    assert [item["question"] for item in failures] == ["worst", "middle"]
    assert [item["score"] for item in failures] == pytest.approx([0.25, 0.65])
    assert failure_analysis(results, bottom_n=0) == []
    assert failure_analysis(results, bottom_n=-1) == []
    assert failure_analysis([]) == []


def test_failure_analysis_rejects_undefined_scores():
    result = EvalResult("q", "a", [], "gt", float("nan"), 0.2, 0.3, 0.4)
    with pytest.raises(ValueError, match="finite"):
        failure_analysis([result])


def test_save_report_contains_aggregates_and_failures(mock_ragas, tmp_path):
    results = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    failures = failure_analysis(results["per_question"])
    path = tmp_path / "reports" / "evaluation.json"
    save_report(results, failures, path=str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["aggregate"]["faithfulness"] == pytest.approx(0.8)
    assert "per_question" not in report["aggregate"]
    assert report["failures"] == failures
    assert report["per_question"][0]["question"] == "q"
    assert report["per_question"][0]["answer"] == "a"
    assert report["per_question"][0]["contexts"] == ["c"]
    assert report["per_question"][0]["faithfulness"] == pytest.approx(0.8)
