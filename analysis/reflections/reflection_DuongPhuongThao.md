# Individual Reflection — Lab 18

**Tên:** Dương Phương Thảo  
**Module phụ trách:** M4 (RAGAS Evaluation) + M5 (Enrichment Pipeline)

---

## 1. Đóng góp kỹ thuật

- **Module M4 — RAGAS Evaluation:**
  - Implement hàm `evaluate_ragas()` tích hợp RAGAS v0.4.3 API với `EvaluationDataset` + `SingleTurnSample`, đánh giá 4 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
  - Implement `failure_analysis()` với Diagnostic Tree tự động xác định root cause (LLM hallucinating, missing chunks, irrelevant chunks, answer mismatch) và suggest fix cho bottom-N câu hỏi tệ nhất.
  - Implement `save_report()` lưu kết quả evaluation dạng JSON chuẩn.
  - Xử lý mapping cột RAGAS (tên cột `llm_context_precision_with_reference`, `llm_context_recall`) sang tên metric chuẩn.

- **Module M5 — Enrichment Pipeline:**
  - Implement 4 kỹ thuật enrichment với dual strategy (LLM-based + rule-based fallback):
    1. `summarize_chunk()`: Tóm tắt chunk bằng LLM hoặc extractive (2 câu đầu).
    2. `generate_hypothesis_questions()`: Tạo HyQA câu hỏi giả định để bridge vocabulary gap, fallback dùng regex patterns nhận diện số liệu, chính sách, bảo mật.
    3. `contextual_prepend()`: Thêm context mô tả chunk theo phong cách Anthropic (giảm 49% retrieval failure), fallback dùng keyword matching phân loại chủ đề.
    4. `extract_metadata()`: LLM extract topic, entities, category, language; fallback dùng regex + keyword heuristic.
  - Implement hàm `enrich_chunks()` gộp pipeline, hỗ trợ chọn methods linh hoạt.

- **Các class/hàm chính đã viết:**
  - `EvalResult` (dataclass), `evaluate_ragas()`, `failure_analysis()`, `save_report()` (M4)
  - `EnrichedChunk` (dataclass), `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `enrich_chunks()` (M5)

- **Số tests pass:** 37/37 (toàn bộ test suite, bao gồm M4: 4/4, M5: 15/15)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** RAGAS framework và cách đánh giá RAG pipeline theo 4 chiều: Faithfulness (LLM có hallucinate không), Answer Relevancy (câu trả lời có đúng trọng tâm không), Context Precision (context retrieve có chính xác không), Context Recall (có bỏ sót thông tin cần thiết không). Hiểu được rằng 4 metrics này phản ánh 4 failure mode khác nhau trong RAG pipeline.

- **Điều bất ngờ nhất:** Faithfulness giảm mạnh (0.83 → 0.66) khi thêm LLM generation so với naive baseline. Naive baseline trả context trực tiếp nên "faithful" tuyệt đối, còn LLM dù được instruct "chỉ dùng context" vẫn bổ sung kiến thức domain (đặc biệt với BCTC). Điều này cho thấy prompt engineering cực kỳ quan trọng — phải dùng extractive prompt cứng thay vì generative mềm.

- **Kết nối với bài giảng:**
  - **Slide 6 (Gap giữa Naive và Production RAG):** Bảng so sánh Naive vs Production giúp set target rõ ràng cho từng metric (Faithfulness ≥0.85, Context Recall ≥0.75, Context Precision ≥0.75, Answer Relevancy ≥0.80). Kết quả thực tế đạt 2/4 targets (Context Precision 0.82, Context Recall 0.75).
  - **Slide 14 (Enrichment Techniques — 4 kỹ thuật chính):** Đây là slide trực tiếp hướng dẫn implement M5. Em đã implement đúng cả 4 kỹ thuật:
    1. **Summary** (slide: "LLM tạo summary ngắn, giảm noise, focus key info") → `summarize_chunk()` dùng gpt-4o-mini tóm tắt 2-3 câu, fallback extractive lấy 2 câu đầu.
    2. **Contextual Prepend** (slide: "Prepend context giải thích chunk nằm ở đâu, giảm 49% retrieval failure") → `contextual_prepend()` thêm mô tả vị trí/chủ đề của chunk theo phong cách Anthropic.
    3. **HyQA** (slide: "LLM generate câu hỏi mà chunk có thể trả lời, bridge vocabulary gap giống HyDE nhưng offline") → `generate_hypothesis_questions()` tạo 3 câu hỏi giả định, index cùng chunk để user query match tốt hơn.
    4. **Auto Metadata** (slide: "LLM extract topic, entities, date_range → enable rich filtering") → `extract_metadata()` trích xuất topic, entities, category, language cho mỗi chunk.
  - Slide nhấn mạnh "Enrichment = **one-time cost** khi indexing, dùng model rẻ (GPT-4o-mini), ROI cao vì cải thiện **mọi** query sau đó" — em đã áp dụng đúng: dùng gpt-4o-mini cho tất cả enrichment, và chạy 1 lần offline trước khi index.
  - **Slide Error Tree (Fix G / Fix R-A / Fix PreRAG):** Rất hữu ích khi implement `failure_analysis()` — map worst metric → diagnosis → suggested fix theo đúng logic Error Tree trong bài giảng.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Tích hợp RAGAS v0.4.3 — API thay đổi nhiều so với version cũ. Phải dùng `SingleTurnSample` thay vì dict, phải wrap LLM và Embeddings qua `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`, và tên cột output khác nhau giữa các metrics (`llm_context_precision_with_reference` thay vì `context_precision`).

- **Cách giải quyết:** Viết mapping linh hoạt `col_map` để detect tên cột thực tế trong DataFrame, không hardcode. Test nhiều lần với dữ liệu thực để verify kết quả.

- **Thời gian debug:** ~45 phút cho RAGAS API integration, ~30 phút cho enrichment fallback logic.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Bắt đầu với LLM-based enrichment thay vì chỉ rule-based fallback trong pipeline. Rule-based enrichment nhanh (0.01s) nhưng chất lượng thấp hơn LLM. Nên chạy LLM enrichment offline 1 lần rồi cache kết quả.

- **Module nào muốn thử tiếp:** M2 (Hybrid Search) — muốn thử thêm sparse embeddings (SPLADE) và so sánh với BM25, và thử Qdrant with server mode thay vì in-memory để persist index.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
