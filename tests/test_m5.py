"""Tests for Module 5: Enrichment Pipeline."""
import sys, os
import json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m5_enrichment import (
    summarize_chunk, generate_hypothesis_questions,
    contextual_prepend, extract_metadata, enrich_chunks, EnrichedChunk,
)
import src.m5_enrichment as enrichment


@pytest.fixture(autouse=True)
def offline_enrichment(monkeypatch):
    """Tests never send live API requests; API behavior is tested explicitly below."""
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "")


@pytest.fixture
def mock_api(monkeypatch):
    import openai
    from types import SimpleNamespace

    state = {"calls": [], "error": None, "content": json.dumps({
        "summary": "Nhân viên có 12 ngày phép năm.",
        "questions": ["Nhân viên có bao nhiêu ngày phép năm?", "Ai được nghỉ phép năm?"],
        "context": "Đoạn văn thuộc chính sách nghỉ phép trong policy.md.",
        "metadata": {"topic": "nghỉ phép", "entities": [], "category": "hr", "language": "vi"},
    }, ensure_ascii=False)}

    def create(**kwargs):
        state["calls"].append(kwargs)
        if state["error"]:
            raise state["error"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=state["content"]))])

    def client(**kwargs):
        assert kwargs["max_retries"] == 0 and kwargs["timeout"] == 30.0
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "test-key")
    return state

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_chunks_type():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert result[0].original_text == SAMPLE


def test_combined_uses_one_call_and_indexes_all_fields(mock_api):
    result = enrich_chunks(CHUNKS)
    assert len(mock_api["calls"]) == len(CHUNKS)
    assert all(item.method == "combined" for item in result)
    first = result[0]
    assert first.original_text == SAMPLE
    assert first.enriched_text.startswith("Đoạn văn thuộc chính sách nghỉ phép")
    assert SAMPLE in first.enriched_text
    assert first.summary in first.enriched_text
    assert all(q in first.enriched_text for q in first.hypothesis_questions)
    assert first.auto_metadata["category"] == "hr"
    request = mock_api["calls"][0]
    assert request["model"] == "gpt-4o-mini"
    assert "temperature" not in request
    assert request["response_format"] == {"type": "json_object"}
    assert json.loads(request["messages"][1]["content"]) == {"source": "policy.md", "text": SAMPLE}


def test_combined_offline_has_complete_fallback():
    result = enrichment._enrich_single_call(SAMPLE, "policy.md")
    assert set(result) == {"summary", "questions", "context", "metadata"}
    assert result["summary"] == SAMPLE
    assert result["questions"] and all(q.endswith("?") for q in result["questions"])
    assert "policy.md" in result["context"]
    assert result["metadata"]["entities"] == []
    assert result["metadata"]["category"] == "general"


def test_extractive_summary_keeps_first_two_sentences():
    text = "Câu đầu. Câu thứ hai! Câu thứ ba?"
    assert summarize_chunk(text) == "Câu đầu. Câu thứ hai!"


def test_offline_context_preserves_exact_original():
    text = "  Dòng một.\nDòng hai.  "
    assert contextual_prepend(text, "Sổ tay") == "Trích từ tài liệu Sổ tay.\n\n" + text
    assert contextual_prepend(text) == text


def test_empty_and_nonpositive_question_count_do_not_call_api(mock_api):
    assert summarize_chunk("  ") == ""
    assert generate_hypothesis_questions("") == []
    assert generate_hypothesis_questions(SAMPLE, n_questions=0) == []
    assert generate_hypothesis_questions(SAMPLE, n_questions=-1) == []
    assert contextual_prepend("", "policy.md") == ""
    assert enrich_chunks([]) == []
    assert mock_api["calls"] == []


def test_api_question_limit_and_metadata(mock_api):
    assert generate_hypothesis_questions(SAMPLE, n_questions=1) == ["Nhân viên có bao nhiêu ngày phép năm?"]
    assert extract_metadata(SAMPLE)["topic"] == "nghỉ phép"
    assert len(mock_api["calls"]) == 2


@pytest.mark.parametrize("content", [
    "invalid JSON", "[]", '{"summary": "x"}',
    '{"summary":"x","questions":"wrong","context":"c","metadata":{}}',
    '{"summary":"x","questions":[],"context":"c","metadata":{"topic":"t","category":"hr","language":"vi","entities":"wrong"}}',
])
def test_invalid_api_output_falls_back_without_retry(mock_api, content):
    mock_api["content"] = content
    result = enrichment._enrich_single_call(SAMPLE, "policy.md")
    assert result == enrichment._fallback_enrichment(SAMPLE, "policy.md")
    assert len(mock_api["calls"]) == 1


def test_api_failure_falls_back_without_extra_calls(mock_api):
    mock_api["error"] = RuntimeError("API unavailable")
    result = enrich_chunks(CHUNKS)
    assert len(mock_api["calls"]) == len(CHUNKS)
    assert all(item.original_text in item.enriched_text for item in result)
    assert result[0].summary == SAMPLE


def test_preserves_trusted_source_metadata(mock_api):
    content = json.loads(mock_api["content"])
    content["metadata"].update({"source": "invented.md", "parent_id": "invented"})
    mock_api["content"] = json.dumps(content)
    metadata = {"source": "policy.md", "parent_id": "parent_0", "category": "policy"}
    chunk = {"text": SAMPLE, "metadata": metadata}
    result = enrich_chunks([chunk])[0]
    assert result.auto_metadata["source"] == "policy.md"
    assert result.auto_metadata["parent_id"] == "parent_0"
    assert result.auto_metadata["category"] == "policy"
    assert chunk["metadata"] == metadata


def test_explicit_methods_enrich_text_without_api():
    result = enrich_chunks(CHUNKS[:1], methods=["summary", "hyqa", "contextual", "metadata"])[0]
    assert result.summary and result.hypothesis_questions
    assert result.enriched_text.startswith("Trích từ tài liệu policy.md.")
    assert result.summary in result.enriched_text
    assert result.auto_metadata["source"] == "policy.md"


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        enrich_chunks(CHUNKS, methods=["typo"])
