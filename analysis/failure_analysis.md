# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Lab 18 - Track 3  
**Thành viên:** Quang → M1 · Tùng → M2 · Hiếu → M3 · Thảo → M4+M5

---

## RAGAS Scores (3 versions)

| Metric | Naive Baseline | Production V1 | Production V2 | Δ (Naive→V2) |
|--------|---------------|--------------|--------------|-------------|
| Faithfulness | 0.8270 | 0.6567 | **0.7867** | -0.0403 |
| Answer Relevancy | 0.3250 | 0.3718 | **0.4066** | +0.0816 |
| Context Precision | 0.6792 | 0.8167 | **0.8179** | +0.1387 |
| Context Recall | 0.5333 | 0.7500 | **0.6833** | +0.1500 |

### Cải tiến V1 → V2

| Thay đổi | Mục đích | Ảnh hưởng |
|----------|----------|-----------|
| Extractive prompt (temperature=0, strict rules, trích dẫn [Nguồn X]) | Giảm hallucination → ↑ Faithfulness | Faithfulness **+0.13** (0.66→0.79) |
| Query rewrite + augmentation (multi-query search) | Mở rộng tìm kiếm → ↑ Answer Relevancy | Answer Relevancy **+0.03** (0.37→0.41) |
| RERANK_TOP_K 3→5 | Nhiều context hơn cho LLM | Hỗ trợ faithfulness, nhưng context recall giảm nhẹ |

## Bottom-5 Failures (V2)

### #1
- **Question:** Lợi nhuận gộp trong báo cáo tài chính được tính như thế nào?
- **Expected:** Công thức tính lợi nhuận gộp từ BCTC
- **Got:** LLM đưa ra công thức đúng nhưng bổ sung kiến thức kế toán ngoài context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (BCTC có dữ liệu) → LLM thêm kiến thức domain
- **Root cause:** Dù đã dùng extractive prompt, gpt-4o-mini vẫn có strong prior về kế toán. Khi chunk chứa số liệu bảng biểu, model tự bổ sung công thức/giải thích.
- **Suggested fix:** Dùng extractive-only mode: chỉ copy-paste từ context, không cho LLM tự diễn đạt. Hoặc thêm post-processing verify claims.

### #2
- **Question:** Báo cáo tài chính gồm những thành phần nào?
- **Expected:** Liệt kê các thành phần BCTC (Bảng CĐKT, KQKD, LCTT, Thuyết minh)
- **Got:** LLM liệt kê thành phần nhưng thêm giải thích ngoài context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (context chứa thông tin BCTC) → LLM hallucinated
- **Root cause:** Câu hỏi liệt kê → LLM có tendency giải thích thêm từng mục dù chỉ cần list. Prompt chưa enforce "chỉ liệt kê, không giải thích".
- **Suggested fix:** Thêm few-shot examples cho câu hỏi dạng liệt kê, chỉ trả list items từ context.

### #3
- **Question:** Tổng tài sản của công ty bao gồm những khoản mục nào?
- **Expected:** Liệt kê khoản mục tài sản từ BCTC
- **Got:** LLM liệt kê đúng nhưng bổ sung thêm giải thích ngoài context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Có (chunks từ BCTC) → LLM thêm thông tin
- **Root cause:** Dữ liệu BCTC dạng bảng/số liệu, khi chunk bị mất cấu trúc bảng → LLM phải "đoán" format → hallucination.
- **Suggested fix:** Table-aware chunking: giữ nguyên cấu trúc markdown table thay vì split ngang qua bảng.

### #4
- **Question:** Quy trình xử lý dữ liệu cá nhân theo Nghị định 13 gồm những bước nào?
- **Expected:** Các bước theo quy định Nghị định 13/2023
- **Got:** LLM tổng hợp từ nhiều context nhưng thêm bước không có trong context
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? → Một phần → Query OK? → Có → Fix Generation
- **Root cause:** Context chứa thông tin liên quan nhưng không đầy đủ → LLM "lấp đầy" lỗ hổng bằng kiến thức riêng.
- **Suggested fix:** Cải thiện prompt: yêu cầu LLM nói "Thông tin trong tài liệu không đề cập đầy đủ" thay vì tự bổ sung.

### #5
- **Question:** Doanh thu thuần của công ty trong kỳ báo cáo là bao nhiêu?
- **Expected:** Số liệu doanh thu thuần cụ thể từ BCTC
- **Got:** Context retrieve được nhưng chứa quá nhiều chunks không liên quan
- **Worst metric:** context_precision = 0.0 (thay đổi so với V1!)
- **Error Tree:** Output sai → Context đúng? → Không chính xác (nhiều noise) → Fix Retrieval
- **Root cause:** Multi-query search kéo về nhiều chunks không liên quan cho câu hỏi số liệu cụ thể. Query rewrite tạo queries quá rộng.
- **Suggested fix:** Thêm metadata filter cho câu hỏi số liệu (category="finance"), hoặc tăng reranker strictness.

## Case Study (cho presentation)

**Question chọn phân tích:** "Lợi nhuận gộp trong báo cáo tài chính được tính như thế nào?"

**Error Tree walkthrough:**
1. Output đúng? → Không. Faithfulness = 0.0 — LLM thêm nội dung ngoài context, dù đã dùng extractive prompt với temperature=0.
2. Context đúng? → Có. Context Precision = cao, chunks từ BCTC.md được retrieve chính xác.
3. Query rewrite OK? → Có. Query đơn giản, cả original và rewritten queries đều match BCTC.
4. Fix ở bước: **Generation (G)**. Model gpt-4o-mini có strong domain prior về kế toán → tự thêm kiến thức dù prompt cấm. Cần dùng extractive-only (copy from context) hoặc Chain-of-Verification.

**Progression V1 → V2:**
- Extractive prompt đã cải thiện Faithfulness tổng thể (+0.13), nhưng 4 câu BCTC vẫn là hard cases.
- Kết luận: prompt engineering có giới hạn — cần thêm post-processing verification cho domain-specific content.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Chain-of-Verification: sau khi LLM trả lời, verify từng claim against context
- Table-aware chunking để bảo tồn cấu trúc bảng biểu BCTC
- Metadata filter cho câu hỏi số liệu tài chính
- Thử gpt-4o thay vì gpt-4o-mini để test nếu model lớn hơn tuân thủ prompt tốt hơn
