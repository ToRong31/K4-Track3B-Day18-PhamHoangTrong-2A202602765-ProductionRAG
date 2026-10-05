import sys
from types import SimpleNamespace

from src.m3_rerank import FlashrankReranker


def test_flashrank_empty_input_does_not_load_model():
    reranker = FlashrankReranker()
    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [{"text": "doc"}], top_k=0) == []
    assert reranker._model is None


def test_flashrank_maps_sorted_scores_to_original_documents(monkeypatch):
    loads = []

    class FakeRanker:
        def __init__(self):
            loads.append(True)

        def rerank(self, request):
            assert request.query == "query"
            assert request.passages == [{"id": 0, "text": "A"}, {"id": 1, "text": "B"}]
            return [{"id": 0, "score": .2}, {"id": 1, "score": .9}]

    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(
        Ranker=FakeRanker, RerankRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    reranker = FlashrankReranker()
    docs = [{"text": "A", "score": .8, "metadata": {"source": "a"}},
            {"text": "B", "score": .4, "metadata": {"source": "b"}}]
    result = reranker.rerank("query", docs, top_k=1)
    assert len(result) == 1
    assert (result[0].text, result[0].original_score, result[0].rerank_score, result[0].rank) == ("B", .4, .9, 0)
    assert result[0].metadata == {"source": "b"}
    assert [item.text for item in reranker.rerank("query", docs)] == ["B", "A"]
    assert len(loads) == 1
