"""Tests for Module 1: Advanced Chunking."""
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m1_chunking import (chunk_basic, chunk_semantic, chunk_hierarchical,
                              chunk_structure_aware, compare_strategies, load_documents, Chunk)

TEXT = """# Nghỉ phép

## Nghỉ phép năm

Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên.

## Nghỉ phép không lương

Nhân viên có thể xin nghỉ phép không lương tối đa 30 ngày mỗi năm.
Đơn xin nghỉ phải được Giám đốc bộ phận phê duyệt.

## Nghỉ ốm

Cần nộp giấy xác nhận y tế trong vòng 3 ngày làm việc."""


# --- Baseline (đã implement sẵn) ---

def test_basic_returns_chunks():
    assert len(chunk_basic(TEXT)) > 0

def test_basic_type():
    assert all(isinstance(c, Chunk) for c in chunk_basic(TEXT))


# --- Semantic Chunking ---

def test_semantic_returns_chunks():
    result = chunk_semantic(TEXT, threshold=0.5)
    assert len(result) > 0, "Semantic chunking should return chunks"

def test_semantic_type():
    assert all(isinstance(c, Chunk) for c in chunk_semantic(TEXT, 0.5))

def test_semantic_groups_by_topic():
    """Semantic should produce fewer chunks than basic (groups related sentences)."""
    basic = chunk_basic(TEXT, chunk_size=100)
    semantic = chunk_semantic(TEXT, threshold=0.5)
    assert len(semantic) <= len(basic) + 2  # Allow some tolerance


# --- Hierarchical Chunking ---

def test_hierarchical_returns_both():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    assert len(parents) > 0, "Should return parents"
    assert len(children) > 0, "Should return children"

def test_hierarchical_children_have_parent_id():
    _, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    for c in children:
        assert c.parent_id is not None, "Each child must have parent_id"

def test_hierarchical_valid_parent_ids():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, f"Child parent_id '{c.parent_id}' not in parents"

def test_hierarchical_children_smaller():
    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    avg_p = sum(len(p.text) for p in parents) / max(len(parents), 1)
    avg_c = sum(len(c.text) for c in children) / max(len(children), 1)
    assert avg_c < avg_p, "Children should be smaller than parents"


# --- Structure-Aware Chunking ---

def test_structure_returns_chunks():
    result = chunk_structure_aware(TEXT)
    assert len(result) > 0, "Structure-aware should return chunks"

def test_structure_preserves_headers():
    result = chunk_structure_aware(TEXT)
    texts = " ".join(c.text for c in result)
    assert "Nghỉ phép năm" in texts, "Should preserve section headers"

def test_structure_has_section_metadata():
    result = chunk_structure_aware(TEXT)
    if result:
        assert any("section" in c.metadata for c in result), "Should have section in metadata"


# --- Compare ---

def test_compare_all_strategies():
    docs = load_documents()
    if docs:
        r = compare_strategies(docs)
        for key in ["basic", "semantic", "hierarchical", "structure"]:
            assert key in r, f"Missing strategy: {key}"


def test_semantic_splits_at_similarity_drop(monkeypatch):
    import numpy as np
    import src.m1_chunking as chunking

    class Model:
        def encode(self, sentences):
            assert sentences == ["First sentence.", "Related sentence!", "New topic?"]
            return np.array([[1., 0.], [1., 0.], [0., 1.]])

    monkeypatch.setattr(chunking, "_semantic_model", lambda: Model())
    metadata = {"source": "example"}
    chunks = chunk_semantic("First sentence. Related sentence!\n\nNew topic?",
                            threshold=0.85, metadata=metadata)
    assert [c.text for c in chunks] == ["First sentence. Related sentence!", "New topic?"]
    assert all(c.metadata == {"source": "example", "strategy": "semantic"}
               for c in chunks)
    assert metadata == {"source": "example"}


def test_empty_input():
    assert chunk_semantic(" \n\n ") == []
    assert chunk_hierarchical(" \n\n ") == ([], [])
    assert chunk_structure_aware(" \n\n ") == []


def test_hierarchical_limits_long_paragraphs():
    text = "x" * 501
    parents, children = chunk_hierarchical(text, parent_size=200, child_size=80,
                                          metadata={"source": "example"})
    assert "".join(p.text for p in parents) == text
    assert all(0 < len(p.text) <= 200 for p in parents)
    assert all(0 < len(c.text) <= 80 for c in children)
    for parent in parents:
        matching = [c for c in children if c.parent_id == parent.parent_id]
        assert "".join(c.text for c in matching) == parent.text
        assert all(c.metadata["parent_id"] == parent.parent_id for c in matching)
    assert all(c.metadata["source"] == "example" for c in parents + children)


@pytest.mark.parametrize("parent_size,child_size", [(0, 80), (200, 0), (-1, 80)])
def test_hierarchical_rejects_invalid_sizes(parent_size, child_size):
    with pytest.raises(ValueError):
        chunk_hierarchical(TEXT, parent_size=parent_size, child_size=child_size)


def test_structure_preserves_tables_lists_and_code():
    body = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n- item\n\n```markdown\n## In code\n```"
    chunks = chunk_structure_aware("Introduction\n\n# Section\n" + body + "\n\n### Next\nEnd")
    assert [c.metadata["section"] for c in chunks] == ["", "# Section", "### Next"]
    assert chunks[1].text == "# Section\n" + body


def test_load_documents_uses_ocr_sidecar_once(tmp_path, monkeypatch):
    import src.m1_chunking as chunking

    (tmp_path / "scan.pdf").write_bytes(b"scanned PDF placeholder")
    (tmp_path / "scan.ocr.md").write_text("# Trang 1\n\nVăn bản OCR.", encoding="utf-8")

    def unexpected_extract(path):
        raise AssertionError("PDF already has an OCR sidecar")

    monkeypatch.setattr(chunking, "_extract_pdf_text", unexpected_extract)
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["text"] == "# Trang 1\n\nVăn bản OCR."
    assert docs[0]["metadata"] == {"source": "scan.pdf", "extraction_method": "ocr"}


def test_empty_ocr_sidecar_does_not_hide_pdf_text(tmp_path, monkeypatch):
    import src.m1_chunking as chunking

    (tmp_path / "scan.pdf").write_bytes(b"PDF placeholder")
    (tmp_path / "scan.ocr.md").write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(chunking, "_extract_pdf_text", lambda path: "Original text layer")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["text"] == "Original text layer"
    assert docs[0]["metadata"] == {"source": "scan.pdf"}
