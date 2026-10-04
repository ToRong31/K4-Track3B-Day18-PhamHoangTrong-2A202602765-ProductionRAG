"""Tests for Module 3: Reranking."""
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker, RerankResult

Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
DOCS = [
    {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
    {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
]

def test_rerank_returns():
    r = CrossEncoderReranker().rerank(Q, DOCS, top_k=2)
    assert len(r) > 0 and len(r) <= 2

def test_rerank_type():
    assert all(isinstance(x, RerankResult) for x in CrossEncoderReranker().rerank(Q, DOCS))

def test_rerank_sorted():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if len(r) >= 2:
        assert r[0].rerank_score >= r[1].rerank_score

def test_rerank_relevant_first():
    r = CrossEncoderReranker().rerank(Q, DOCS)
    if r:
        assert "nghỉ" in r[0].text.lower() or "12" in r[0].text

def test_benchmark_stats():
    stats = benchmark_reranker(CrossEncoderReranker(), Q, DOCS, n_runs=2)
    assert "avg_ms" in stats and "min_ms" in stats and "max_ms" in stats


def test_empty_and_zero_top_k_do_not_load_model(monkeypatch):
    reranker = CrossEncoderReranker()

    def unexpected_load():
        raise AssertionError("No model needed for empty results")

    monkeypatch.setattr(reranker, "_load_model", unexpected_load)
    assert reranker.rerank(Q, []) == []
    assert reranker.rerank(Q, DOCS, top_k=0) == []
    assert reranker.rerank(Q, DOCS, top_k=-1) == []


def test_pairs_ranking_and_original_metadata():
    import numpy as np

    class Model:
        def predict(self, pairs, **kwargs):
            assert pairs == [(Q, doc["text"]) for doc in DOCS]
            return np.array([0.2, 0.9, 0.5])

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    results = reranker.rerank(Q, DOCS, top_k=2)
    assert [r.text for r in results] == [DOCS[1]["text"], DOCS[2]["text"]]
    assert [r.rank for r in results] == [0, 1]
    assert [r.rerank_score for r in results] == pytest.approx([0.9, 0.5])
    assert [r.original_score for r in results] == [0.7, 0.6]
    assert DOCS[0]["score"] == 0.8


@pytest.mark.parametrize("score", [0.75, "numpy_scalar", "numpy_array"])
def test_single_document_scalar_scores(score):
    import numpy as np

    if score == "numpy_scalar":
        score = np.float32(0.75)
    elif score == "numpy_array":
        score = np.array(0.75)

    class Model:
        def predict(self, pairs, **kwargs):
            return score

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    results = reranker.rerank(Q, [{"text": "answer", "metadata": {"source": "policy"}}])
    assert len(results) == 1
    assert results[0].rank == 0
    assert results[0].rerank_score == pytest.approx(0.75)
    assert results[0].original_score == 0.0
    assert results[0].metadata == {"source": "policy"}


def test_load_model_once(monkeypatch):
    import sentence_transformers

    calls = []
    model = object()

    def load(name, **kwargs):
        calls.append(name)
        return model

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", load)
    reranker = CrossEncoderReranker()
    assert reranker._load_model() is model
    assert reranker._load_model() is model
    assert calls == ["BAAI/bge-reranker-v2-m3"]


def test_rejects_missing_scores():
    class Model:
        def predict(self, pairs, **kwargs):
            return [0.5]

    reranker = CrossEncoderReranker()
    reranker._model = Model()
    with pytest.raises(ValueError, match="one score per document"):
        reranker.rerank(Q, DOCS)
