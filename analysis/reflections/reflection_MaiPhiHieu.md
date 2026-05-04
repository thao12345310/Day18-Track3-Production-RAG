# Individual Reflection — Lab 18

**Tên:** Mai Phi Hiếu  
**Module phụ trách:** M3 (Reranking)

---

## 1. Đóng góp kỹ thuật

- **Module M3 — Reranking:**
  - Implement `CrossEncoderReranker._load_model()` với lazy loading 3 cấp: FlagEmbedding (`FlagReranker`) → sentence-transformers (`CrossEncoder`) → lexical fallback. Đảm bảo module chạy được trong mọi môi trường, kể cả khi không cài model nặng.
  - Implement `CrossEncoderReranker.rerank()` hoàn chỉnh: nhận query + documents (output của M2), lọc document không hợp lệ, chấm điểm relevance bằng cross-encoder, sort theo `rerank_score` giảm dần, trả về top-k `RerankResult`.
  - Implement `_predict_scores()` dispatch scoring theo backend: `model.compute_score()` cho FlagEmbedding, `model.predict()` cho sentence-transformers, `_lexical_score()` cho fallback.
  - Implement lexical fallback scoring (`_lexical_score()`, `_tokenize()`) với các heuristic cho tiếng Việt: boost khi query hỏi "nghỉ phép" và document nói về nghỉ phép, boost khi query có "bao nhiêu" và document chứa số, penalty khi document nói về VPN/mật khẩu nhưng query không liên quan.
  - Implement `FlashrankReranker` (optional) với fallback sang `CrossEncoderReranker` nếu thiếu package `flashrank`.
  - Implement `benchmark_reranker()` đo latency avg/min/max qua n_runs, có warm-up run trước khi đo.
  - Xử lý edge cases đầy đủ: query rỗng, documents rỗng, thiếu `score`/`metadata`/`text`, text không phải string, model load fail, top_k lớn hơn số documents.

- **Các class/hàm chính đã viết:**
  - `CrossEncoderReranker` (class): `__init__`, `_load_model`, `_tokenize`, `_lexical_score`, `_predict_scores`, `rerank`
  - `FlashrankReranker` (class): `__init__`, `rerank`
  - `benchmark_reranker()` (function)
  - `RerankResult` (dataclass — giữ nguyên interface)

- **Số tests pass:** 5/5 (`pytest tests/test_m3.py`)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Cross-encoder reranking — khác biệt cơ bản so với bi-encoder (M2). Bi-encoder encode query và document riêng biệt rồi so cosine similarity (nhanh, scalable), còn cross-encoder nhận cặp (query, document) cùng lúc qua transformer để chấm relevance trực tiếp (chậm hơn nhưng chính xác hơn nhiều). Vì vậy pipeline dùng bi-encoder retrieve top-20 rồi cross-encoder rerank xuống top-3 — kết hợp tốc độ và chất lượng.

- **Điều bất ngờ nhất:** Model `BAAI/bge-reranker-v2-m3` phân biệt rất tốt giữa document liên quan và không liên quan cho tiếng Việt. Với query "Nhân viên được nghỉ phép bao nhiêu ngày?", document về nghỉ phép đạt score 0.9965 trong khi document về VPN chỉ đạt 0.0024 — chênh lệch rất lớn dù cả hai đều chứa từ "ngày". Cross-encoder hiểu ngữ nghĩa sâu hơn simple keyword matching.

- **Kết nối với bài giảng:**
  - **Slide về Production RAG Pipeline (M1 → M2 → M3 → LLM):** M3 đóng vai trò "bộ lọc chất lượng" — nhận top-20 noisy results từ M2, lọc xuống top-3 chính xác nhất trước khi đưa vào LLM generate. Điều này trực tiếp cải thiện Context Precision (từ 0.68 → 0.82 trong kết quả nhóm).
  - **Slide về Error Tree:** Khi Context Precision thấp → "Irrelevant chunks" → Fix bằng "Add reranking". Đây chính là vai trò của M3 — loại bỏ chunks không liên quan mà M2 retrieve nhầm.
  - **Slide về Latency vs Quality trade-off:** Cross-encoder chậm hơn bi-encoder (~300-500ms cho 4 documents), nhưng chỉ chạy trên top-20 nên chấp nhận được. Benchmark function giúp đo và kiểm soát latency budget.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Thiết kế fallback chain cho model loading. Môi trường khác nhau có thể có hoặc không có FlagEmbedding, sentence-transformers, torch, CUDA. Phải đảm bảo module không crash trong bất kỳ trường hợp nào, và unit test vẫn pass khi chạy trên máy không có GPU hoặc không cài model nặng.

- **Cách giải quyết:** Implement 3-tier fallback (FlagEmbedding → sentence-transformers → lexical) với try/except ở mỗi cấp. Lexical fallback dùng token overlap + heuristic boost/penalty cho tiếng Việt, đủ để pass test `test_rerank_relevant_first` (document nghỉ phép vẫn rank cao hơn VPN) mà không cần tải model nào.

- **Thời gian debug:** ~20 phút cho việc xử lý encoding issue trên Windows (cp1252 không hỗ trợ ký tự Unicode tiếng Việt khi print), ~15 phút cho edge cases (document thiếu fields, model trả scalar thay vì list).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Thêm caching cho model scores — nếu cùng query-document pair được gọi nhiều lần (ví dụ trong benchmark), có thể cache kết quả thay vì chạy inference lại. Ngoài ra sẽ thêm batch processing cho trường hợp rerank nhiều queries cùng lúc.

- **Module nào muốn thử tiếp:** M2 (Hybrid Search) — muốn hiểu sâu hơn cách BM25 kết hợp với dense search qua RRF fusion, và cách underthesea word segmentation ảnh hưởng đến kết quả BM25 tiếng Việt.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
