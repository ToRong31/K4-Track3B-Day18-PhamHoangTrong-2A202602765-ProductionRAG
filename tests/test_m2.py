"""Tests for Module 2: Hybrid Search."""
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m2_search import segment_vietnamese, BM25Search, reciprocal_rank_fusion, SearchResult
from src.m2_search import DenseSearch, HybridSearch

CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
    {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
    {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
]

def test_segment_returns_string():
    assert isinstance(segment_vietnamese("nghỉ phép năm"), str)

def test_bm25_search():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép", top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"

def test_bm25_relevant_first():
    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search("nghỉ phép năm", top_k=2)
    if results:
        assert "nghỉ" in results[0].text.lower() or "12" in results[0].text

def test_rrf_merges():
    a = [SearchResult("doc1", 0.9, {}, "bm25"), SearchResult("doc2", 0.8, {}, "bm25")]
    b = [SearchResult("doc2", 0.95, {}, "dense"), SearchResult("doc3", 0.85, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and "doc2" in [r.text for r in merged]

def test_rrf_method():
    a = [SearchResult("d1", 0.9, {}, "bm25")]
    b = [SearchResult("d1", 0.8, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], top_k=1)
    if merged:
        assert merged[0].method == "hybrid"


def test_segment_normalizes_compound_words(monkeypatch):
    import underthesea

    def tokenize(text, format):
        assert text == "nghỉ phép năm" and format == "text"
        return "nghỉ_phép năm"

    monkeypatch.setattr(underthesea, "word_tokenize", tokenize)
    assert segment_vietnamese("nghỉ phép năm") == "nghỉ phép năm"


def test_bm25_empty_and_unmatched_queries():
    search = BM25Search()
    assert search.search("nghỉ phép") == []
    search.index(CHUNKS)
    assert search.search("xyzunknown") == []
    assert search.search("   ") == []
    assert search.search("nghỉ phép", top_k=0) == []
    assert search.search("NGHỈ PHÉP")[0].metadata == {"source": "policy"}
    search.index([])
    assert search.search("nghỉ phép") == []
    search.index([{"text": "", "metadata": {}}])
    assert search.search("nghỉ phép") == []


def test_rrf_exact_scores_and_top_k():
    a = [SearchResult("doc1", 1000, {"source": "a"}, "bm25"),
         SearchResult("doc2", 500, {"source": "b"}, "bm25")]
    b = [SearchResult("doc2", 0.01, {"source": "b"}, "dense"),
         SearchResult("doc3", 0.001, {}, "dense")]
    merged = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert [r.text for r in merged] == ["doc2", "doc1"]
    assert merged[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert merged[1].score == pytest.approx(1 / 61)
    assert merged[0].metadata == {"source": "b"}
    assert all(r.method == "hybrid" for r in merged)
    assert a[1].score == 500 and a[1].method == "bm25"


def test_rrf_empty_and_invalid_parameters():
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([[SearchResult("d", 1, {}, "bm25")]], top_k=0) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=-1)


@pytest.fixture
def dense_search():
    """Exercise real Qdrant indexing/querying with deterministic 1024-D embeddings."""
    import numpy as np
    from qdrant_client import QdrantClient
    from config import EMBEDDING_DIM

    class Encoder:
        def encode(self, texts, **kwargs):
            def vector(text):
                result = np.zeros(EMBEDDING_DIM)
                result[0 if "nghỉ" in text.lower() else 1] = 1.0
                return result

            return vector(texts) if isinstance(texts, str) else np.array([vector(t) for t in texts])

    search = DenseSearch.__new__(DenseSearch)
    search.client = QdrantClient(":memory:")
    search._encoder = Encoder()
    yield search
    search.client.close()


def test_dense_index_query_and_metadata(dense_search):
    dense_search.index(CHUNKS, collection="test_m2")
    results = dense_search.search("nghỉ phép", top_k=2, collection="test_m2")
    assert len(results) == 2
    assert results[0].text == CHUNKS[0]["text"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == {"source": "policy"}
    assert all(r.method == "dense" for r in results)


def test_dense_reindex_replaces_old_points(dense_search):
    dense_search.index(CHUNKS, collection="test_m2")
    dense_search.index(CHUNKS[:1], collection="test_m2")
    results = dense_search.search("nghỉ phép", top_k=10, collection="test_m2")
    assert len(results) == 1
    dense_search.index([], collection="test_m2")
    assert dense_search.search("nghỉ phép", collection="test_m2") == []


def test_dense_missing_collection_and_empty_queries(dense_search):
    assert dense_search.search("nghỉ phép", collection="missing") == []
    dense_search.index(CHUNKS, collection="test_m2")
    assert dense_search.search(" ", collection="test_m2") == []
    assert dense_search.search("nghỉ phép", top_k=0, collection="test_m2") == []


def test_hybrid_combines_real_bm25_and_qdrant(dense_search):
    search = HybridSearch.__new__(HybridSearch)
    search.bm25 = BM25Search()
    search.dense = dense_search
    search.index(CHUNKS)
    results = search.search("nghỉ phép", top_k=2)
    assert len(results) == 2
    assert results[0].text == CHUNKS[0]["text"]
    assert all(r.method == "hybrid" for r in results)
