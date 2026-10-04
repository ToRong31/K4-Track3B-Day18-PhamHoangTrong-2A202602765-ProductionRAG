from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .ocr.md: bản OCR của PDF cùng tên; giữ source PDF và không nạp trùng.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    ocr_sources = set()
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(fp)
        metadata = {"source": source}
        if source.endswith(".ocr.md"):
            if not text.strip():
                continue
            source = source[:-len(".ocr.md")] + ".pdf"
            metadata = {"source": source, "extraction_method": "ocr"}
            ocr_sources.add(source)
        docs.append({"text": text, "metadata": metadata})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        if os.path.basename(fp) in ocr_sources:
            continue
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


@lru_cache(maxsize=1)
def _semantic_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text)
                 if s.strip()]
    if not sentences:
        return []

    from numpy import dot
    from numpy.linalg import norm

    embeddings = _semantic_model().encode(sentences)
    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        similarity = dot(a, b) / (norm(a) * norm(b) + 1e-9)
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentences[i])
    return [Chunk(text=" ".join(group),
                  metadata={**(metadata or {}), "strategy": "semantic"})
            for group in groups]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_bounded(text: str, size: int) -> list[str]:
    """Prefer paragraph/line/word boundaries, with a hard limit for long tokens."""
    parts = []
    remaining = text.strip()
    while len(remaining) > size:
        window = remaining[:size + 1]
        boundary = window.rfind("\n\n", 1, size + 1)
        if boundary < 1:
            boundary = window.rfind("\n", 1, size + 1)
        if boundary < 1:
            boundary = window.rfind(" ", 1, size + 1)
        end = boundary if boundary > 0 else size
        part = remaining[:end].strip()
        if part:
            parts.append(part)
        remaining = remaining[end:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    metadata = metadata or {}
    parents, children = [], []
    for parent_text in _split_bounded(text, parent_size):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        ))
        for child_text in _split_bounded(parent_text, child_size):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    chunks = []
    section = ""
    lines = []
    fence_char, fence_length = None, 0

    def flush():
        content = "".join(lines).strip()
        if content:
            chunks.append(Chunk(text=content, metadata={
                **(metadata or {}), "section": section, "strategy": "structure",
            }))

    for line in text.splitlines(keepends=True):
        fence = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line.rstrip())
        if fence:
            marker, suffix = fence.groups()
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not suffix.strip():
                fence_char, fence_length = None, 0
            lines.append(line)
            continue
        header = re.match(r'^#{1,3}[ \t]+\S.*$', line.rstrip())
        if header and fence_char is None:
            flush()
            lines = []
            section = header.group(0)
        lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
