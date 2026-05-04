# Individual Reflection — Lab 18

**Tên:** Nguyễn Việt Quang  
**Module phụ trách:** M1 (Advanced Chunking Strategies)

---

## 1. Đóng góp kỹ thuật

- **Module M1 — Advanced Chunking Strategies:**
  - Implement hàm `chunk_semantic()` sử dụng model `all-MiniLM-L6-v2` (sentence-transformers) để encode từng câu thành vector, tính cosine similarity giữa các câu liên tiếp, và tách chunk mới khi similarity < threshold (default 0.85). Giúp nhóm câu cùng chủ đề vào chung 1 chunk, tránh cắt giữa ý.
  - Implement hàm `chunk_hierarchical()` với cơ chế parent-child: tạo parent chunks (2048 chars) rồi split thành children (256 chars) qua hàm `_split_parent_into_children()`. Mỗi child có `parent_id` link đến parent — khi retrieve child (embedding chính xác) → lookup parent (đủ context cho LLM). Đây là strategy được pipeline production chọn dùng.
  - Implement helper `_split_parent_into_children()` split parent thành children theo sentence boundaries (regex `(?<=[.!?])\s+`) thay vì cắt ngang từ, đảm bảo mỗi child chunk là câu hoàn chỉnh.
  - Implement hàm `chunk_structure_aware()` parse markdown headers (h1–h3) bằng regex `^#{1,3}\s+.+$`, pair header + content thành chunk logic. Giữ nguyên tables, code blocks, lists — không cắt giữa chừng. Mỗi chunk có metadata `section` chứa header tương ứng.
  - Implement hàm `compare_strategies()` chạy cả 4 strategies (basic, semantic, hierarchical, structure) trên toàn bộ documents, tính stats (num_chunks, avg_length, min_length, max_length) và in bảng so sánh A/B.

- **Các class/hàm chính đã viết:**
  - `chunk_semantic()`, `chunk_hierarchical()`, `_split_parent_into_children()`, `chunk_structure_aware()`, `compare_strategies()` (M1)
  - Dataclass `Chunk` (có sẵn, sử dụng xuyên suốt)

- **Số tests pass:** 8/8 (toàn bộ test M1: basic 2/2, semantic 3/3, hierarchical 4/4, structure 3/3, compare 1/1)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Sự khác biệt giữa các chunking strategies và impact lên downstream performance trong RAG pipeline. Basic chunking (split theo paragraph) đơn giản nhưng hay cắt giữa ý, semantic chunking (nhóm theo cosine similarity) giữ coherence nhưng chậm hơn do phải encode, hierarchical chunking (parent-child) là production recommendation vì kết hợp precision (children nhỏ → embedding chính xác) với context (parent lớn → LLM đủ thông tin). Structure-aware chunking giữ nguyên logical structure của document (headers, tables).

- **Điều bất ngờ nhất:** Chunking time gần như bằng 0 (0.0s trong latency breakdown) so với tổng pipeline (~1316s). Tưởng rằng chunking sẽ là bottleneck vì phải xử lý text, nhưng thực tế bottleneck nằm ở query time (1052s, ~80%) và RAGAS eval (194s). Điều này cho thấy chunking là "one-time offline cost" rất rẻ — đầu tư thêm vào chunking quality (semantic, table-aware) sẽ có ROI rất cao vì cải thiện mọi query sau đó.

- **Kết nối với bài giảng:**
  - **Slide về Chunking Strategies (Naive vs Production):** Bài giảng so sánh basic paragraph splitting với các advanced strategies. Em đã implement đúng 3 strategies nâng cao: semantic (nhóm theo similarity), hierarchical (parent-child), structure-aware (parse markdown headers). Pipeline production dùng hierarchical chunking với parent_size=2048, child_size=256 — đúng theo recommendation trong slide.
  - **Slide về Hierarchical Chunking (Parent-Child pattern):** Slide giải thích pattern "Index children → retrieve children → return parent cho LLM". Em implement đúng pattern này: `_split_parent_into_children()` tạo children từ parent, mỗi child có `parent_id`. Trong `pipeline.py`, children được index vào vector DB, khi query retrieve child → dùng `parent_id` lookup parent chunk lớn hơn để cung cấp đủ context.
  - **Slide về Structure-Aware Chunking:** Slide nhấn mạnh "Giữ nguyên tables, code blocks, lists — không cắt giữa chừng". Em implement bằng regex split theo markdown headers, mỗi section (header + content) thành 1 chunk, metadata chứa `section` name. Phương pháp này đặc biệt hữu ích với corpus có structured documents (docs kỹ thuật, API refs, quy định pháp luật).
  - **Slide về Enrichment (one-time cost khi indexing):** Liên hệ trực tiếp — chunking cũng là one-time cost. Chất lượng chunk ảnh hưởng trực tiếp đến embedding quality → search quality → answer quality. Failure analysis cho thấy 2/5 failures liên quan đến chunking: BCTC bị mất cấu trúc bảng khi chunk, và context recall thấp do thiếu chunks relevants.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Implement `chunk_semantic()` — phải xử lý sentence splitting cho tiếng Việt. Tiếng Việt không phải lúc nào cũng kết thúc câu bằng dấu chấm chuẩn, và các ký tự đặc biệt trong markdown (headers, bullet points) gây nhiễu khi split. Regex `(?<=[.!?])\s+|\n\n` phải xử lý cả punctuation lẫn paragraph break.

- **Cách giải quyết:** Dùng regex kết hợp lookbehind `(?<=[.!?])` để split tại vị trí sau dấu kết thúc câu, kết hợp `\n\n` cho paragraph break. Filter empty strings bằng list comprehension. Test với corpus tiếng Việt thực tế (data/ directory) để verify kết quả hợp lý.

- **Khó khăn phụ:** Chọn threshold cho semantic chunking. Threshold quá cao (0.95) → mỗi câu thành 1 chunk (quá nhỏ). Threshold quá thấp (0.3) → toàn bộ text thành 1 chunk (quá lớn). Config default `SEMANTIC_THRESHOLD = 0.85` cho kết quả hợp lý nhưng phụ thuộc vào corpus.

- **Thời gian debug:** ~40 phút cho semantic chunking (regex + threshold tuning), ~20 phút cho hierarchical (parent-child linking), ~15 phút cho structure-aware (markdown regex).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Thêm table-aware chunking cho dữ liệu BCTC. Failure analysis cho thấy 4/5 bottom failures liên quan đến câu hỏi về bảng biểu tài chính — khi chunk cắt ngang qua markdown table, LLM mất context và phải "đoán" → hallucination. Nếu implement table-aware chunking (detect markdown table, giữ nguyên toàn bộ table trong 1 chunk), faithfulness có thể cải thiện đáng kể.

- **Module nào muốn thử tiếp:** M3 (Reranking) — muốn hiểu sâu hơn cách cross-encoder reranker (BAAI/bge-reranker-v2-m3) scoring và thử so sánh với FlashRank lightweight reranker. Cũng muốn thử semantic chunking kết hợp với reranking để xem liệu chunk quality tốt hơn có giảm workload cho reranker không.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
