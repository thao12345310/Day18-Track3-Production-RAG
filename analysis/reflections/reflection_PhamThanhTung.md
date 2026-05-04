# Individual Reflection — Lab 18

**Tên:** Phạm Thanh Tùng
**Module phụ trách:** M2 (Hybrid Search - BM25 + Dense + RRF)

---

## 1. Đóng góp kỹ thuật

- **Module M2 — Hybrid Search:**
  - Triển khai thuật toán `BM25Search` tối ưu riêng cho tiếng Việt. Tích hợp thư viện `underthesea` để segment text (cắt từ ghép) trước khi tokenize, giúp hệ thống không bị mất ngữ nghĩa của các từ ghép tiếng Việt (ví dụ: "nghỉ_phép").
  - Triển khai thuật toán `DenseSearch` sử dụng cơ sở dữ liệu vector `Qdrant` và mô hình embedding đa ngôn ngữ `bge-m3` để tìm kiếm dựa trên ngữ nghĩa (semantic similarity). Đã code thêm logic fallback sang chế độ in-memory nếu Qdrant server không khả dụng.
  - Xây dựng thuật toán `reciprocal_rank_fusion` (RRF) để kết hợp (merge) linh hoạt kết quả từ BM25 (chuyên keyword) và Dense (chuyên ngữ nghĩa) mà không cần phải chuẩn hóa điểm số phức tạp.
  - Gộp chung toàn bộ quy trình vào class `HybridSearch`, tạo ra một pipeline search hoàn chỉnh.

- **Các class/hàm chính đã viết:**
  - Hàm `segment_vietnamese()` xử lý riêng cho tiếng Việt.
  - Các Class: `BM25Search`, `DenseSearch`, `HybridSearch`.
  - Hàm `reciprocal_rank_fusion()` xử lý thuật toán RRF.

- **Số tests pass:** 5/5 (vượt qua toàn bộ 100% test case của `pytest tests/test_m2.py`).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Sự khác biệt giữa Sparse Retrieval (BM25) và Dense Retrieval. Dense hiểu được ngữ nghĩa và các từ đồng nghĩa, nhưng lại thường gặp khó khăn với các từ khóa quá cụ thể (tên riêng, mã số). Ngược lại, BM25 cực kỳ giỏi bắt keyword chính xác nhưng mù ngữ nghĩa. Hybrid Search bù trừ hoàn hảo cho nhau.
- **Điều bất ngờ nhất:** Tầm quan trọng của Word Segmentation trong tiếng Việt. Tiếng Việt là ngôn ngữ âm tiết rời, nếu không dùng `underthesea` ghép từ ("nhân", "viên" thành "nhân_viên") thì thuật toán BM25 sẽ đánh index hoàn toàn sai, dẫn đến kết quả tìm kiếm rất kém.
- **Kết nối với bài giảng:** 
  - Áp dụng đúng công thức `Reciprocal Rank Fusion` theo lý thuyết bài giảng `score = 1 / (k + rank)` để gộp hai bảng xếp hạng độc lập mà không bị dính thiên lệch điểm (scale bias).
  - Áp dụng được mô hình `bge-m3` theo đúng lời khuyên của bài giảng về việc chọn model hỗ trợ đa ngôn ngữ để xử lý tốt corpus tiếng Việt.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Điểm số (score) trả về từ BM25 có range không giới hạn (có thể lên tới hàng chục, hàng trăm) trong khi điểm của Dense Search (Cosine Similarity) luôn bị giới hạn từ -1 đến 1. Lúc đầu rất khó để cộng gộp điểm hai bên lại với nhau để ra được Hybrid score.
- **Cách giải quyết:** Áp dụng phương pháp xếp hạng chéo Reciprocal Rank Fusion (RRF). Thay vì dùng điểm tuyệt đối, RRF chỉ sử dụng thứ hạng (rank) của document trong từng list để cộng dồn, qua đó giải quyết hoàn toàn vấn đề khác biệt thang đo điểm số.
- **Thời gian debug:** Mất nhiều thời gian để setup và hiểu luồng dữ liệu của `Qdrant` (cách đóng gói text và metadata vào Payload / PointStruct). 

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ thử nghiên cứu và thay thế BM25 truyền thống bằng Sparse Embedding models hiện đại hơn như SPLADE để xem có cải thiện được độ chính xác (recall) ở bước này hay không. 
- **Module nào muốn thử tiếp:** M3 (Reranking). Tôi muốn thử đưa output 20 kết quả từ `HybridSearch` qua một cross-encoder mạnh hơn xem kết quả sẽ được chắt lọc ra sao trước khi gửi cho LLM.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |
