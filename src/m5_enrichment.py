"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import json
import re

from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_openai_client():
    """Khởi tạo OpenAI client nếu có API key."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).

    Strategy:
        - Nếu có OPENAI_API_KEY: dùng gpt-4o-mini để tóm tắt bằng LLM.
        - Fallback (extractive): lấy 2 câu đầu tiên của chunk.
    """
    client = _get_openai_client()

    if client:
        # Option A: LLM-based summarization
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn. "
                            "Giữ lại thông tin quan trọng nhất. Trả lời bằng ngôn ngữ gốc của văn bản."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass  # Fallback to extractive

    # Option B: Extractive fallback — không cần API key
    # Tách câu bằng dấu chấm câu phổ biến
    sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    # Lấy tối đa 2 câu đầu
    summary_sentences = sentences[:2]
    summary = " ".join(summary_sentences)
    # Đảm bảo kết thúc bằng dấu câu
    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.

    Why HyQA:
        User hỏi "nghỉ phép bao nhiêu ngày?" nhưng doc viết
        "12 ngày làm việc mỗi năm" → vocabulary gap. HyQA bridge gap này
        bằng cách index câu hỏi "Nhân viên được nghỉ bao nhiêu ngày?" cùng chunk.
    """
    client = _get_openai_client()

    if client:
        # Option A: LLM-based question generation
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số. "
                            "Câu hỏi phải dùng ngôn ngữ tự nhiên như người dùng thực sự hỏi."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip()
            questions = raw.split("\n")
            # Làm sạch: bỏ số thứ tự, dấu - ở đầu
            questions = [
                re.sub(r"^[\d\.\-\)\s]+", "", q).strip()
                for q in questions
                if q.strip()
            ]
            return [q for q in questions if q][:n_questions]
        except Exception:
            pass  # Fallback to rule-based

    # Option B: Rule-based extractive fallback — không cần API
    # Tạo câu hỏi từ nội dung bằng cách nhận diện thực thể số và khái niệm chính
    questions = []
    text_lower = text.lower()

    # Pattern 1: Phát hiện số ngày → hỏi số lượng
    day_match = re.search(r'(\d+)\s*ngày', text)
    if day_match and len(questions) < n_questions:
        questions.append("Có bao nhiêu ngày được quy định?")

    # Pattern 2: Phát hiện chính sách / quy định → hỏi về quy định
    if any(kw in text_lower for kw in ["quy định", "chính sách", "được phép", "phải"]):
        if len(questions) < n_questions:
            questions.append("Quy định cụ thể về vấn đề này là gì?")

    # Pattern 3: Phát hiện mật khẩu / bảo mật
    if any(kw in text_lower for kw in ["mật khẩu", "password", "bảo mật", "security"]):
        if len(questions) < n_questions:
            questions.append("Yêu cầu về bảo mật mật khẩu là gì?")

    # Pattern 4: Hỏi chung về chủ đề chính
    first_sentence = re.split(r'[.!?]', text)[0].strip()
    if first_sentence and len(questions) < n_questions:
        questions.append(f"Thông tin về {first_sentence[:30].lower()}... là gì?")

    # Đảm bảo trả về đúng số câu hỏi yêu cầu
    while len(questions) < n_questions:
        questions.append(f"Nội dung chính của đoạn văn này là gì?")

    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.

    Example output:
        "Trích từ chính sách nhân sự, nói về quyền nghỉ phép năm của nhân viên.

        Nhân viên chính thức được nghỉ phép năm 12 ngày..."
    """
    client = _get_openai_client()

    if client:
        # Option A: LLM-based context generation
        try:
            user_content = text
            if document_title:
                user_content = f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì. Chỉ trả về 1 câu, không giải thích thêm."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception:
            pass  # Fallback to rule-based

    # Option B: Rule-based fallback — không cần API
    # Sinh context đơn giản dựa trên tên file và nội dung
    context_parts = []

    if document_title:
        # Làm đẹp tên file: bỏ extension, thay _ bằng khoảng trắng
        clean_title = re.sub(r'\.[a-z]+$', '', document_title)
        clean_title = clean_title.replace("_", " ").replace("-", " ")
        context_parts.append(f"Trích từ tài liệu '{clean_title}'")
    else:
        context_parts.append("Trích từ tài liệu nội bộ")

    # Nhận diện chủ đề từ từ khóa
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["nghỉ phép", "nghỉ lễ", "ngày nghỉ"]):
        context_parts.append("về chính sách nghỉ phép")
    elif any(kw in text_lower for kw in ["mật khẩu", "bảo mật", "vpn", "it"]):
        context_parts.append("về chính sách bảo mật IT")
    elif any(kw in text_lower for kw in ["lương", "thưởng", "phụ cấp"]):
        context_parts.append("về chính sách lương thưởng")
    elif any(kw in text_lower for kw in ["tuyển dụng", "nhân viên mới", "onboarding"]):
        context_parts.append("về quy trình tuyển dụng và hội nhập")
    else:
        context_parts.append("về quy định và chính sách công ty")

    context = " ".join(context_parts) + "."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields:
        {
            "topic": str,
            "entities": list[str],
            "category": str,  # "policy" | "hr" | "it" | "finance"
            "language": str,  # "vi" | "en"
        }

    Use case:
        Metadata gắn vào chunk → enable rich filtering khi search.
        VD: filter category="policy" + topic="nghỉ phép" → precision tăng.
    """
    client = _get_openai_client()

    if client:
        # Option A: LLM-based metadata extraction (JSON output)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trích xuất metadata từ đoạn văn. '
                            'Trả về JSON hợp lệ với các trường: '
                            '{"topic": "chủ đề chính", "entities": ["danh sách thực thể"], '
                            '"category": "policy|hr|it|finance|general", "language": "vi|en"}'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            # Xử lý trường hợp LLM trả về markdown code block
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip("` \n")
            return json.loads(raw)
        except Exception:
            pass  # Fallback to rule-based

    # Option B: Rule-based extractive fallback — không cần API
    text_lower = text.lower()

    # Phát hiện ngôn ngữ đơn giản
    vietnamese_chars = len(re.findall(r'[àáảãạăắặẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text))
    language = "vi" if vietnamese_chars > 2 else "en"

    # Phát hiện category
    if any(kw in text_lower for kw in ["nghỉ phép", "nghỉ lễ", "ngày nghỉ", "chính sách", "quy định"]):
        category = "policy"
    elif any(kw in text_lower for kw in ["mật khẩu", "vpn", "bảo mật", "it", "hệ thống"]):
        category = "it"
    elif any(kw in text_lower for kw in ["lương", "thưởng", "phụ cấp", "tài chính", "ngân sách"]):
        category = "finance"
    elif any(kw in text_lower for kw in ["nhân viên", "tuyển dụng", "hr", "nhân sự"]):
        category = "hr"
    else:
        category = "general"

    # Trích xuất topic đơn giản từ câu đầu tiên
    first_sentence = re.split(r'[.!?]', text)[0].strip()
    topic = first_sentence[:60] if first_sentence else "Không xác định"

    # Trích xuất entities: tìm số + đơn vị, tên viết hoa
    entities = []
    # Số + đơn vị
    number_units = re.findall(r'\d+\s*(?:ngày|tháng|năm|ngày làm việc|%|giờ)', text)
    entities.extend(number_units)
    # Từ viết hoa (tên riêng) — chỉ cho tiếng Anh
    if language == "en":
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized[:3])

    return {
        "topic": topic,
        "entities": list(set(entities))[:5],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.

    Note:
        Enrichment = one-time cost (offline). Dùng model rẻ (gpt-4o-mini).
        ROI cao vì cải thiện MỌI query sau đó.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    use_summary = "summary" in methods or "full" in methods
    use_hyqa = "hyqa" in methods or "full" in methods
    use_contextual = "contextual" in methods or "full" in methods
    use_metadata = "metadata" in methods or "full" in methods

    enriched = []

    for chunk in chunks:
        text: str = chunk["text"]
        meta: dict = chunk.get("metadata", {})

        # Step 1: Summarize chunk
        summary = summarize_chunk(text) if use_summary else ""

        # Step 2: Generate hypothesis questions (HyQA)
        questions = generate_hypothesis_questions(text) if use_hyqa else []

        # Step 3: Contextual prepend
        document_title = meta.get("source", "")
        enriched_text = (
            contextual_prepend(text, document_title) if use_contextual else text
        )

        # Step 4: Auto metadata extraction
        auto_meta = extract_metadata(text) if use_metadata else {}

        # Gộp metadata gốc + metadata tự động (auto_meta có thể override)
        combined_meta = {**meta, **auto_meta}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata=combined_meta,
                method="+".join(methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}\n")

    # Demo full pipeline
    print("=== Full Pipeline Demo ===\n")
    chunks = [
        {"text": sample, "metadata": {"source": "policy.md"}},
        {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
    ]
    results = enrich_chunks(chunks, methods=["contextual", "hyqa", "metadata"])
    for i, ec in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"  original_text:        {ec.original_text}")
        print(f"  enriched_text:        {ec.enriched_text}")
        print(f"  hypothesis_questions: {ec.hypothesis_questions}")
        print(f"  auto_metadata:        {ec.auto_metadata}")
        print(f"  method:               {ec.method}\n")
