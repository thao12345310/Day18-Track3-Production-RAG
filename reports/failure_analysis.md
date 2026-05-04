# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Lab 18 - Track 3  
**Thành viên:** Quang → M1 · Tùng → M2 · Hiếu → M3 · Thảo → M4

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8270 | 0.6567 | -0.1703 |
| Answer Relevancy | 0.3250 | 0.3718 | +0.0468 |
| Context Precision | 0.6792 | 0.8167 | +0.1375 |
| Context Recall | 0.5333 | 0.7500 | +0.2167 |

## Bottom-5 Failures

### #1
- **Question:** Báo cáo tài chính gồm những thành phần nào?
- **Expected:** Liệt kê các thành phần BCTC (Bảng CĐKT, KQKD, LCTT, Thuyết minh)
- **Got:** LLM tạo câu trả lời nhưng thêm thông tin không có trong context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (context chứa thông tin BCTC) → LLM hallucinated thêm nội dung
- **Root cause:** LLM generation thêm kiến thức bên ngoài context. Prompt chưa đủ chặt để ép model chỉ dùng context.
- **Suggested fix:** Thêm instruction "CHỈ sử dụng thông tin trong context, KHÔNG thêm bất kỳ thông tin nào khác" vào system prompt. Giảm temperature xuống 0.

### #2
- **Question:** Tổng tài sản của công ty bao gồm những khoản mục nào?
- **Expected:** Liệt kê khoản mục tài sản từ BCTC
- **Got:** LLM liệt kê đúng nhưng bổ sung thêm giải thích ngoài context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (chunks từ BCTC) → LLM thêm thông tin
- **Root cause:** Dữ liệu BCTC dạng bảng/số liệu, khi chunk bị mất cấu trúc bảng → LLM phải "đoán" format, dẫn đến hallucination.
- **Suggested fix:** Cải thiện chunking cho bảng biểu: giữ nguyên cấu trúc markdown table thay vì split ngang qua bảng. Dùng table-aware chunking.

### #3
- **Question:** Quy trình xử lý dữ liệu cá nhân theo Nghị định 13 gồm những bước nào?
- **Expected:** Các bước theo quy định Nghị định 13/2023
- **Got:** LLM tổng hợp từ nhiều context nhưng thêm bước không có trong context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Một phần (context chứa thông tin liên quan nhưng không đầy đủ) → Query OK? → Có → Fix Generation
- **Root cause:** Context recall chưa cao đủ (thiếu chunks chứa thông tin đầy đủ), kết hợp LLM "lấp đầy" lỗ hổng bằng kiến thức riêng → faithfulness = 0.
- **Suggested fix:** Tăng RERANK_TOP_K từ 3 lên 5 để đưa nhiều context hơn. Cải thiện prompt yêu cầu LLM nói "Không đủ thông tin" thay vì đoán.

### #4
- **Question:** Doanh thu thuần của công ty trong kỳ báo cáo là bao nhiêu?
- **Expected:** Số liệu doanh thu thuần cụ thể từ BCTC
- **Got:** LLM trả lời với số liệu nhưng thêm phân tích không có trong context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (chunk BCTC có số liệu) → LLM thêm nhận xét/phân tích
- **Root cause:** Câu hỏi về số liệu cụ thể nhưng LLM có xu hướng thêm giải thích. Prompt không ép đủ chặt để trả lời ngắn gọn.
- **Suggested fix:** Thêm instruction cho câu hỏi số liệu: "Trả lời ngắn gọn, chỉ đưa số liệu từ context." Hoặc dùng few-shot examples trong prompt.

### #5
- **Question:** Lợi nhuận gộp trong báo cáo tài chính được tính như thế nào?
- **Expected:** Công thức tính lợi nhuận gộp từ BCTC
- **Got:** LLM đưa ra công thức đúng nhưng bổ sung kiến thức kế toán ngoài context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (BCTC có dữ liệu) → LLM thêm kiến thức domain
- **Root cause:** Tương tự #1-#4: LLM hallucination. Model gpt-4o-mini có strong prior về kiến thức kế toán nên tự bổ sung thêm, dù prompt yêu cầu chỉ dùng context.
- **Suggested fix:** Dùng extractive approach thay vì generative: trả lời bằng cách trích dẫn trực tiếp từ context thay vì để LLM tự diễn đạt. Hoặc thêm post-processing kiểm tra faithfulness trước khi trả lời.

## Case Study (cho presentation)

**Question chọn phân tích:** "Báo cáo tài chính gồm những thành phần nào?"

**Error Tree walkthrough:**
1. Output đúng? → Không. Faithfulness = 0.0 — LLM thêm nội dung ngoài context.
2. Context đúng? → Có. Context Precision = cao, chunks từ BCTC.md được retrieve chính xác.
3. Query rewrite OK? → Có. Query đơn giản, BM25 + Dense đều match tốt.
4. Fix ở bước: **Generation (G)**. Vấn đề nằm ở LLM prompt quá loose, cho phép model thêm kiến thức bên ngoài. Cần tighten prompt + giảm temperature + có thể dùng extractive QA thay vì generative.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Cải thiện prompt template cho LLM: thêm few-shot examples, instruction cứng hơn, format câu trả lời
- Tăng RERANK_TOP_K để cung cấp nhiều context hơn cho LLM
- Thêm table-aware chunking để bảo tồn cấu trúc bảng biểu BCTC
- Thử Chain-of-Verification: sau khi LLM trả lời, verify từng claim against context
