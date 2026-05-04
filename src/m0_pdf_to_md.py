"""
Module 0: PDF → Markdown Preprocessing
========================================
Aligned with ingestion_pipeline.md flow:
  - BCTC.pdf      → Journey B (OCR via Claude VLM) — bảng tài chính phức tạp
  - Nghi_dinh.pdf  → A3 Path (Text extraction via MarkItDown) — text-heavy

Usage: python src/m0_pdf_to_md.py
"""

import os
import sys
import base64
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


# ─── A3 Path: Text Extraction → MD (MarkItDown) ─────────────

def convert_a3_text_path(pdf_path: str, output_path: str) -> str:
    """
    A3 Path: Text Extraction for 2-layer PDFs.
    Maps to: ingestion_pipeline → A3_EXTRACT → A3_PRODUCE_HTML → A3_HTML2MD

    Uses MarkItDown which handles the full pipeline internally:
      Text Extraction → Produce HTML → Unrolling Table → HTML → MD

    Auto-fallback: If extracted text < 100 chars, assumes scanned PDF
    and falls back to Tesseract OCR.
    """
    from markitdown import MarkItDown

    print(f"  📝 Running MarkItDown text extraction...")
    md = MarkItDown()
    result = md.convert(pdf_path)

    content = result.text_content.strip()

    if len(content) < 100:
        print(f"  ⚠️  MarkItDown extracted only {len(content)} chars — likely scanned PDF")
        print(f"  🔄 Auto-fallback to Tesseract OCR...")
        return convert_ocr_tesseract_path(pdf_path, output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ A3 Path complete: {os.path.basename(output_path)}")
    print(f"     Characters: {len(content):,}")
    return content


# ─── OCR Path: Tesseract + pdf2image (scanned text-heavy PDFs) ─────

def convert_ocr_tesseract_path(pdf_path: str, output_path: str) -> str:
    """
    OCR Path for scanned/image-only PDFs using Tesseract.
    Maps to: ingestion_pipeline → OCR_NODE (local Tesseract instead of Claude VLM)

    Pipeline: PDF → Images (pdf2image/poppler) → OCR (Tesseract) → Markdown
    Suitable for text-heavy scanned documents (e.g. legal decrees).
    """
    from pdf2image import convert_from_path
    import pytesseract

    print(f"  📝 Converting PDF pages to images...")
    images = convert_from_path(pdf_path, dpi=300)
    print(f"     {len(images)} pages detected")

    all_text = []
    for i, img in enumerate(images, 1):
        print(f"  🔍 OCR page {i}/{len(images)}...", end="\r")
        text = pytesseract.image_to_string(img, lang="vie+eng")
        if text.strip():
            all_text.append(f"<!-- Page {i} -->\n\n{text.strip()}")

    print(f"  🔍 OCR complete: {len(images)} pages processed")

    content = "\n\n---\n\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ Tesseract OCR Path complete: {os.path.basename(output_path)}")
    print(f"     Characters: {len(content):,}")
    return content


# ─── Journey B: OCR via Claude VLM → MD ────────────────────

def convert_b_ocr_path(pdf_path: str, output_path: str) -> str:
    """
    Journey B: OCR via Claude VLM for complex PDFs (tables, charts).
    Maps to: ingestion_pipeline → OCR_NODE (Claude VLM) → B_BROKEN_TABLE → B_UNROLL → B_HTML2MD

    Claude API supports native PDF input — no need for pdf2image.
    Simplified: Claude VLM → MD directly (prompt handles table formatting).
    """
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Read PDF as base64
    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"  📄 PDF size: {file_size_mb:.1f} MB")
    print(f"  🤖 Sending to Claude VLM for OCR...")

    # Prompt covers pipeline steps: OCR + Parse Broken Table + Unroll Table + HTML→MD
    prompt = """Hãy chuyển đổi TOÀN BỘ nội dung tài liệu PDF này sang định dạng Markdown.

QUY TẮC QUAN TRỌNG:
1. **Bảng biểu**: Dùng Markdown table syntax chính xác (| col1 | col2 |). Nếu có merged cells, tách thành các ô riêng biệt.
2. **Số liệu**: Giữ CHÍNH XÁC tất cả con số, đơn vị tiền tệ, phần trăm. KHÔNG làm tròn, KHÔNG bỏ sót.
3. **Heading**: Dùng #, ##, ### phù hợp với cấu trúc tài liệu gốc.
4. **Tiếng Việt**: Giữ nguyên toàn bộ, KHÔNG dịch sang tiếng Anh.
5. **Chỉ trả về nội dung Markdown**, KHÔNG thêm giải thích, commentary, hay ghi chú của bạn.
6. **Trang trống / chỉ có logo**: Bỏ qua hoàn toàn.
7. **Ghi chú cuối trang (footnotes)**: Giữ lại nếu có nội dung quan trọng.
8. Đánh dấu mỗi trang bằng comment <!-- Page X --> để dễ trace."""

    start_time = time.time()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    elapsed = time.time() - start_time
    content = message.content[0].text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ Journey B complete: {os.path.basename(output_path)}")
    print(f"     Characters: {len(content):,}")
    print(f"     Time: {elapsed:.1f}s")
    print(f"     Tokens: input={message.usage.input_tokens:,}, output={message.usage.output_tokens:,}")
    return content


# ─── Main: File Type Router → PDF Layer Detection → Path ────

def main():
    """
    Full preprocessing pipeline, aligned with ingestion_pipeline.md:

    1. File Type Detection → .pdf
    2. PDF Layer Detection → 2-layer / scanned / complex
    3. Route to A3 (text extraction) or Journey B (VLM/OCR)
    4. Output → data/*.md (ready for m1_chunking.py → load_documents())
    """
    print("=" * 60)
    print("MODULE 0: PDF → MARKDOWN PREPROCESSING")
    print("Aligned with ingestion_pipeline.md")
    print("=" * 60)

    pdf_configs = [
        {
            "filename": "BCTC.pdf",
            "output": "BCTC.md",
            "description": "Báo cáo tài chính — bảng số liệu phức tạp",
            "path": "journey_b",  # OCR via Claude VLM
        },
        {
            "filename": "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf",
            "output": "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan.md",
            "description": "Nghị định pháp luật — scanned text-heavy",
            "path": "ocr_tesseract",  # Scanned PDF → Tesseract OCR
        },
    ]

    results = {}

    for cfg in pdf_configs:
        pdf_path = os.path.join(DATA_DIR, cfg["filename"])
        md_path = os.path.join(DATA_DIR, cfg["output"])

        if not os.path.exists(pdf_path):
            print(f"\n⚠️  Skip: {cfg['filename']} not found")
            continue

        if os.path.exists(md_path):
            print(f"\n⏭️  Skip: {cfg['output']} already exists")
            results[cfg["filename"]] = "skipped (already exists)"
            continue

        print(f"\n{'─' * 50}")
        print(f"📄 Processing: {cfg['filename']}")
        print(f"   {cfg['description']}")
        print(f"   Route: {'Journey B (Claude VLM)' if cfg['path'] == 'journey_b' else 'A3 Path (MarkItDown)'}")

        try:
            if cfg["path"] == "journey_b":
                content = convert_b_ocr_path(pdf_path, md_path)
            elif cfg["path"] == "ocr_tesseract":
                content = convert_ocr_tesseract_path(pdf_path, md_path)
            else:
                content = convert_a3_text_path(pdf_path, md_path)

            results[cfg["filename"]] = f"✅ {len(content):,} chars → {cfg['output']}"
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[cfg["filename"]] = f"❌ {e}"

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 SUMMARY")
    print(f"{'=' * 60}")
    for filename, status in results.items():
        print(f"  {filename}: {status}")

    md_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".md")]
    print(f"\n📁 data/ → {len(md_files)} markdown files: {md_files}")
    print("✅ Ready for: m1_chunking.py → load_documents()")


if __name__ == "__main__":
    main()
