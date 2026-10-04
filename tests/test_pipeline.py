from types import SimpleNamespace
import json
import pytest
from src.pipeline import prepare_chunks, retrieve_contexts, run_query, enrich_children, _index_matches
from src.m3_rerank import RerankResult


def test_pipeline_uses_bounded_m1_hierarchy_and_links_original_parents():
    text = "# Handbook\n\n" + "Policy text " * 500
    parents, children = prepare_chunks([{"text": text, "metadata": {"source": "a.md"}}])
    assert len(parents) > 1
    parent_map = {parent["metadata"]["parent_id"]: parent["text"] for parent in parents}
    assert all(len(parent["text"]) <= 2048 for parent in parents)
    assert all(len(child["text"]) <= 256 for child in children)
    assert all(child["metadata"]["parent_text"] == parent_map[child["metadata"]["parent_id"]]
               for child in children)
    assert all(child["text"] in child["metadata"]["parent_text"] for child in children)
    assert all(child["metadata"]["chunk_type"] == "child" for child in children)


def test_parent_ids_are_unique_across_documents():
    parents, _ = prepare_chunks([
        {"text": "# A\n\nOne", "metadata": {"source": "a.md"}},
        {"text": "# B\n\nTwo", "metadata": {"source": "b.md"}},
    ])
    assert len({item["metadata"]["parent_id"] for item in parents}) == 2


def test_retrieval_deduplicates_parents_and_uses_raw_evidence():
    hits = [SimpleNamespace(text="Generated misleading text", score=1., metadata={
        "source": source, "parent_id": pid, "parent_text": raw, "raw_text": raw,
    }) for source, pid, raw in [("a", "p1", "Original A"), ("a", "p1", "Original A"), ("b", "p2", "Original B")]]
    search = SimpleNamespace(search=lambda query: hits)

    def rerank(query, docs, top_k):
        assert len(docs) == 3
        assert [d["text"] for d in docs] == ["Original A", "Original A", "Original B"]
        assert top_k == 3
        return [RerankResult(d["text"], d["score"], 1., d["metadata"], i) for i, d in enumerate(docs[:top_k])]

    contexts = retrieve_contexts("question", search, SimpleNamespace(rerank=rerank))
    assert contexts == ["Original A", "Original B"]
    assert not any("Generated" in context for context in contexts)


def test_empty_retrieval_skips_generation():
    search = SimpleNamespace(search=lambda query: [])
    assert run_query("q", search, object()) == ("Không tìm thấy thông tin.", [])


def test_enrichment_cache_avoids_duplicate_calls(tmp_path, monkeypatch):
    import config
    import src.pipeline as pipeline
    calls = []
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test")

    def enrich(parents):
        calls.append(parents)
        return [SimpleNamespace(enriched_text="generated", auto_metadata={})]

    monkeypatch.setattr(pipeline, "enrich_chunks", enrich)
    parents = [{"text": "original", "metadata": {"source": "a", "parent_text": "original"}}]
    path = str(tmp_path / "cache.json")
    first = enrich_children(parents, path)
    second = enrich_children(parents, path)
    assert len(calls) == 1 and first == second
    assert second[0]["metadata"]["raw_text"] == "original"


def test_generation_prompt_uses_original_context_and_default_temperature(monkeypatch):
    import config
    import openai
    import src.pipeline as pipeline
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(pipeline, "retrieve_contexts", lambda *args: ["Original evidence"])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert run_query("q", object(), object()) == ("Answer", ["Original evidence"])
    assert "temperature" not in requests[0]
    assert "Original evidence" in requests[0]["messages"][1]["content"]


def test_retrieval_reranks_children_before_expanding_parents():
    query = "vacation days for Senior"
    queries = []
    hits = [SimpleNamespace(text=f"child {i}", score=1., metadata={
        "source": "a.md", "parent_id": f"p{i}", "parent_text": f"parent {i}",
    }) for i in range(20)]

    def search(text):
        queries.append(text)
        return hits

    def rerank(text, docs, top_k):
        assert text == query
        assert len(docs) == 20 and top_k == 20
        assert all(d["text"].startswith("child") for d in docs)
        chosen = [docs[i] for i in (19, 8, 3)]
        return [RerankResult(d["text"], d["score"], 1., d["metadata"], i)
                for i, d in enumerate(chosen)]

    contexts = retrieve_contexts(query, SimpleNamespace(search=search), SimpleNamespace(rerank=rerank))
    assert queries == [query]
    assert contexts == ["parent 19", "parent 8", "parent 3"]


@pytest.mark.parametrize("query, expected_count", [
    ("What is the password policy?", 3),
    ("What was the policy in 2022?", 3),
    ("Compare old versions", 3),
])
def test_current_policy_order_preserves_history(query, expected_count):
    def hit(title, year, source):
        parent = f"# {title}\n> Ngày hiệu lực: 01/01/{year} | Phòng ban: IT\n\nOriginal policy"
        return SimpleNamespace(text="Generated query bait", score=1., metadata={
            "source": source, "parent_id": source, "parent_text": parent, "raw_text": "Original policy",
        })

    hits = [hit("Passwords (Phiên bản cũ)", 2022, "old"),
            hit("Passwords (Phiên bản hiện hành)", 2024, "new"),
            hit("Device policy", 2021, "device")]

    def rerank(query, docs, top_k):
        assert len(docs) == expected_count
        assert all("Generated" not in doc["text"] for doc in docs)
        return [RerankResult(d["text"], 1., 1., d["metadata"], i) for i, d in enumerate(docs)]

    contexts = retrieve_contexts(query, SimpleNamespace(search=lambda query: hits), SimpleNamespace(rerank=rerank))
    assert len(contexts) == expected_count
    if query == "What is the password policy?":
        assert "2024" in contexts[0] and "2022" in contexts[1]
    if "2022" in query:
        assert "2022" in contexts[0]


def test_duplicate_children_do_not_consume_all_context_slots():
    hits = [SimpleNamespace(text=f"child {i}", score=1., metadata={
        "source": "a", "parent_id": pid, "parent_text": f"parent {pid}", "raw_text": f"child {i}",
    }) for i, pid in enumerate(["p1", "p1", "p1", "p2", "p3"])]

    def rerank(query, docs, top_k):
        return [RerankResult(d["text"], 1., 1., d["metadata"], i) for i, d in enumerate(docs[:top_k])]

    contexts = retrieve_contexts("query", SimpleNamespace(search=lambda query: hits), SimpleNamespace(rerank=rerank))
    assert contexts == ["parent p1", "parent p2", "parent p3"]


def test_multi_intent_adds_missing_evidence_without_displacing_current_policy():
    def hit(source, parent):
        return SimpleNamespace(text="generated", score=1., metadata={
            "source": source, "parent_id": source, "parent_text": parent, "raw_text": source,
        })

    old = hit("old", "# Leave (Phiên bản cũ)\n> Ngày hiệu lực: 01/01/2023\n\nOld leave")
    new = hit("new", "# Leave (Phiên bản hiện hành)\n> Ngày hiệu lực: 01/01/2024\n\nCurrent leave")
    salary = hit("salary", "Salary table")
    irrelevant = hit("unrelated", "Unrelated policy")
    query = "annual leave entitlement and monthly salary range"
    queries = []

    def search(text):
        queries.append(text)
        return [salary] if text == "monthly salary range" else [old, new, irrelevant]

    def rerank(text, docs, top_k):
        return [RerankResult(d["text"], 1., 1., d["metadata"], i) for i, d in enumerate(docs[:top_k])]

    contexts = retrieve_contexts(query, SimpleNamespace(search=search), SimpleNamespace(rerank=rerank))
    assert queries == [query, "annual leave entitlement", "monthly salary range"]
    assert contexts == [new.metadata["parent_text"], "Salary table", old.metadata["parent_text"]]


def test_build_indexes_only_enriched_children(monkeypatch):
    import src.pipeline as pipeline
    calls = []
    monkeypatch.setattr(pipeline, "load_documents", lambda: [{"text": "Policy " * 400, "metadata": {"source": "a"}}])

    def enrich(children):
        assert all(len(child["text"]) <= 256 for child in children)
        return [{"text": "enriched", "metadata": child["metadata"]} for child in children]

    search = SimpleNamespace(index=lambda chunks: calls.append(chunks))
    monkeypatch.setattr(pipeline, "enrich_children", enrich)
    monkeypatch.setattr(pipeline, "HybridSearch", lambda: search)
    monkeypatch.setattr(pipeline, "CrossEncoderReranker", lambda: object())
    pipeline.build_pipeline()
    assert len(calls) == 1
    assert all(child["text"] == "enriched" for child in calls[0])
    assert all(child["metadata"]["chunk_type"] == "child" for child in calls[0])


def test_index_reuse_requires_matching_text_and_metadata():
    payload = {"text": "Original", "source": "a", "parent_id": "p1"}
    client = SimpleNamespace(collection_exists=lambda name: True,
                             scroll=lambda *args, **kwargs: ([SimpleNamespace(payload=payload)], None))
    search = SimpleNamespace(dense=SimpleNamespace(client=client))
    chunks = [{"text": "Original", "metadata": {"source": "a", "parent_id": "p1"}}]
    assert _index_matches(search, chunks)
    assert not _index_matches(search, [{"text": "Modified", "metadata": chunks[0]["metadata"]}])
    assert not _index_matches(search, [{"text": "Original", "metadata": {"source": "b", "parent_id": "p1"}}])


def test_missing_index_is_not_reused():
    search = SimpleNamespace(dense=SimpleNamespace(client=SimpleNamespace(collection_exists=lambda name: False)))
    assert not _index_matches(search, [])


@pytest.mark.parametrize("matching_revision", [True, False])
def test_resume_only_reuses_answers_from_current_revision(tmp_path, monkeypatch, matching_revision):
    import src.pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    items = [{"question": "q1", "ground_truth": "g1"}, {"question": "q2", "ground_truth": "g2"}]
    monkeypatch.setattr(pipeline, "load_test_set", lambda: items)
    reports = tmp_path / "reports"
    reports.mkdir()
    checkpoint = reports / "query_traces.json"
    checkpoint.write_text(json.dumps([{
        "pipeline_revision": pipeline.PIPELINE_REVISION if matching_revision else "obsolete",
        **items[0], "answer": "cached", "contexts": ["cached evidence"],
    }]), encoding="utf-8")
    calls, evaluated = [], []

    def query(question, *args):
        calls.append(question)
        return "fresh " + question, ["evidence " + question]

    def evaluate(questions, answers, contexts, truths):
        evaluated.append((questions, answers, contexts, truths))
        return {"per_question": []}

    monkeypatch.setattr(pipeline, "run_query", query)
    monkeypatch.setattr(pipeline, "evaluate_ragas", evaluate)
    monkeypatch.setattr(pipeline, "save_report", lambda *args: None)
    pipeline.evaluate_pipeline(object(), object(), resume=True)
    assert calls == (["q2"] if matching_revision else ["q1", "q2"])
    assert evaluated[0][0] == ["q1", "q2"]
    assert evaluated[0][1] == (["cached", "fresh q2"] if matching_revision else ["fresh q1", "fresh q2"])
    traces = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [row["question"] for row in traces] == ["q1", "q2"]
    assert all(row["pipeline_revision"] == pipeline.PIPELINE_REVISION for row in traces)
