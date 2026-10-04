from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict
from math import isfinite

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    empty_result = {**dict.fromkeys(metric_names, 0.0), "per_question": []}
    try:
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("Evaluation inputs must have matching lengths")
        if not questions:
            return empty_result

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                           context_precision, context_recall])
        frame = result.to_pandas()
        if len(frame) != len(questions):
            raise ValueError("RAGAS must return one result per question")
        per_question = []
        for _, row in frame.iterrows():
            scores = {name: float(row[name]) for name in metric_names}
            if not all(isfinite(score) for score in scores.values()):
                raise ValueError("RAGAS returned an undefined metric score")
            per_question.append(EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"], **scores,
            ))
        return {**{name: sum(getattr(item, name) for item in per_question) / len(per_question)
                   for name in metric_names}, "per_question": per_question}
    except Exception as error:
        # Zero placeholders are not measured scores; expose the failure explicitly.
        error_type = type(error).__name__
        print(f"  ⚠️  RAGAS evaluation failed ({error_type}).")
        return {**empty_result, "error": error_type}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "LLM có thể bịa thông tin ngoài tài liệu",
            "Thắt chặt system prompt, giảm temperature về 0",
        ),
        "context_recall": (
            "Hệ thống tìm kiếm có thể bỏ sót đoạn văn liên quan",
            "Cải thiện chunking hoặc bổ sung từ khóa BM25",
        ),
        "context_precision": (
            "Đoạn văn không liên quan có thể bị xếp lên đầu",
            "Bổ sung Cross-Encoder reranking hoặc lọc theo metadata",
        ),
        "answer_relevancy": (
            "Câu trả lời có thể lệch trọng tâm câu hỏi",
            "Viết lại prompt để hướng dẫn trả lời trực tiếp hơn",
        ),
    }
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name)) for name in diagnostic_tree}
        if not all(isfinite(score) for score in scores.values()):
            raise ValueError("Failure analysis requires finite metric scores")
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": result.question, "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(item) for item in results.get("per_question", [])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
