"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
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


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Split text into sentences (by punctuation or double newline)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences using a fast sentence-transformer model
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # 3. Helper: cosine similarity between two vectors
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # 4. Group consecutive sentences by similarity
    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Similarity dropped — flush current group as a chunk
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])

    # 5. Don't forget last group
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}

    # 1. Split text into paragraphs, then accumulate into parent chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    parents: list[Chunk] = []
    children: list[Chunk] = []
    current_text = ""
    p_index = 0

    for para in paragraphs:
        # If adding this paragraph would exceed parent_size, flush the current parent
        if current_text and len(current_text) + len(para) + 2 > parent_size:
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_index},
            ))
            # Split this parent into children using a sliding window
            _split_parent_into_children(current_text.strip(), pid, child_size, metadata, children)
            p_index += 1
            current_text = ""
        current_text += para + "\n\n"

    # Flush the last parent
    if current_text.strip():
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_index},
        ))
        _split_parent_into_children(current_text.strip(), pid, child_size, metadata, children)

    return parents, children


def _split_parent_into_children(
    parent_text: str, parent_id: str, child_size: int,
    metadata: dict, children: list[Chunk],
) -> None:
    """Split a parent chunk into children using a sliding window.

    Production pattern:
      - Index CHILDREN vào vector DB (nhỏ → embedding chính xác)
      - Khi retrieve child → lookup parent_id → trả parent cho LLM (đủ context)
    """
    # Try to split on sentence boundaries first for cleaner children
    sentences = re.split(r'(?<=[.!?])\s+', parent_text)
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > child_size:
            children.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
                parent_id=parent_id,
            ))
            current = ""
        current += sentence + " "
    if current.strip():
        children.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "child", "chunk_index": len(children)},
            parent_id=parent_id,
        ))


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # 1. Split by markdown headers (h1-h3)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    # 2. Pair headers with their content
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Flush the previous section if it has content
            if current_header or current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "section": current_header,
                            "strategy": "structure",
                            "chunk_index": len(chunks),
                        },
                    ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # 3. Don't forget last section
    if current_header or current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "section": current_header,
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            ))

    # Ưu điểm: giữ nguyên tables, lists, code blocks
    # Dùng khi: corpus có structured documents (docs, API refs, manuals)
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    # Collect all chunks per strategy across all documents
    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(parents)
        all_children.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    def _stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    results = {
        "basic": _stats(all_basic),
        "semantic": _stats(all_semantic),
        "hierarchical": {
            "parents": _stats(all_parents),
            "children": _stats(all_children),
            "num_chunks": f"{len(all_parents)}p/{len(all_children)}c",
        },
        "structure": _stats(all_structure),
    }

    # Print comparison table
    print(f"{'Strategy':<18} | {'Chunks':>8} | {'Avg Len':>8} | {'Min':>5} | {'Max':>5}")
    print("-" * 56)
    for name in ["basic", "semantic", "structure"]:
        s = results[name]
        print(f"{name:<18} | {s['num_chunks']:>8} | {s['avg_length']:>8} | {s['min_length']:>5} | {s['max_length']:>5}")
    # Hierarchical: special format
    h = results["hierarchical"]
    h_children = h["children"]
    print(f"{'hierarchical':<18} | {h['num_chunks']:>8} | {h_children['avg_length']:>8} | {h_children['min_length']:>5} | {h_children['max_length']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
