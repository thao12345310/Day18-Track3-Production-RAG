# Group Report — Lab 18: Production RAG

**Nhóm:** Lab 18 - Track 3  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyen Viet Quang | M1: Chunking | ☑ | 8/8 |
| Pham Thanh Tung | M2: Hybrid Search | ☑ | 5/5 |
| Mai Phi Hieu | M3: Reranking | ☑ | 5/5 |
| Duong Phuong Thao | M4: Evaluation | ☑ | 4/4 |

**Tổng tests:** 37/37 passed ✓

## Kết quả RAGAS (3 versions)

| Metric | Naive | Production V1 | Production V2 | Δ (Naive→V2) |
|--------|-------|--------------|--------------|-------------|
| Faithfulness | 0.8270 | 0.6567 | **0.7867** | -0.0403 |
| Answer Relevancy | 0.3250 | 0.3718 | **0.4066** | +0.0816 ↑ |
| Context Precision | 0.6792 | 0.8167 | **0.8179** | +0.1387 ↑ |
| Context Recall | 0.5333 | 0.7500 | **0.6833** | +0.1500 ↑ |

### Cải tiến V1 → V2

| Optimization | Technique | Impact |
|-------------|-----------|--------|
| Better prompt | Extractive prompt, temperature=0, trích dẫn [Nguồn X], strict rules | Faithfulness **+0.13** ↑ |
| Query rewrite | LLM rewrite query → multi-query search → merge results | Answer Relevancy **+0.03** ↑ |
| Reranking tuning | RERANK_TOP_K 3→5, weighted score boosting | Context Precision giữ ổn |

**Đạt yêu cầu:** Context Precision (0.8179 ≥ 0.75) ✓

## Key Findings

1. **Biggest improvement:** Context Precision tăng +0.1387 (từ 0.68 → 0.82) xuyên suốt cả V1 và V2. Hybrid Search (BM25 + Dense + RRF) kết hợp Cross-Encoder Reranking giúp retrieve chính xác hơn. BM25 tiếng Việt bắt keyword matches, Dense (bge-m3) bắt semantic similarity, reranker lọc noise.

2. **Biggest challenge:** Faithfulness giảm khi thêm LLM generation. V1 giảm -0.17 do prompt quá loose. V2 khắc phục +0.13 nhờ extractive prompt (temperature=0, strict rules), nhưng vẫn chưa đạt naive baseline. Đặc biệt 4 câu hỏi BCTC (bảng biểu/số liệu) vẫn faithfulness=0 dù prompt đã rất chặt — gpt-4o-mini có strong domain prior về kế toán.

3. **Surprise finding:** Query rewrite cải thiện answer relevancy nhưng giảm context recall (-0.07). Rewritten queries đôi khi lệch hướng → kéo về noise. Giải pháp: dùng original query làm primary, rewritten queries chỉ boost score (weight 0.3) thay vì ngang hàng.

## Latency Breakdown

### V1 (no query rewrite)
| Step | Time | % Total |
|------|------|---------|
| M1: Chunking | 0.0s | 0% |
| M5: Enrichment | 0.01s | 0% |
| M2: Indexing | 49.8s | 9.4% |
| M3: Reranker Load | 16.6s | 3.1% |
| Query (20 queries) | 289.1s | 54.6% |
| M4: RAGAS Eval | 174.4s | 32.9% |
| **Total** | **529.9s** | **100%** |

### V2 (with query rewrite)
| Step | Time | % Total |
|------|------|---------|
| M1: Chunking | 0.0s | 0% |
| M5: Enrichment | 0.01s | 0% |
| M2: Indexing | 53.5s | 4.1% |
| M3: Reranker Load | 16.5s | 1.3% |
| Query (20 queries) | 1052.4s | **79.9%** |
| M4: RAGAS Eval | 193.9s | 14.7% |
| **Total** | **1316.4s** | **100%** |

> **Nhận xét:** V2 chậm hơn ~2.5x do mỗi query có thêm bước LLM rewrite + 3x search calls. Trade-off: +0.13 faithfulness nhưng +786s latency.

## Architecture

```
Documents → M1 (Hierarchical Chunking)
         → M5 (Rule-based Enrichment: contextual + HyQA + metadata)
         → M2 (BM25 Vietnamese + Dense bge-m3 + Qdrant → RRF Fusion)
         ↓
Query → LLM Query Rewrite (V2) → Multi-query Search
     → M3 (Cross-Encoder Reranking: BAAI/bge-reranker-v2-m3, top-5)
     → LLM Generation (gpt-4o-mini, extractive prompt, temp=0)
     → M4 (RAGAS Evaluation)
```

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** 3 versions: Naive → V1 → V2. V2 cải thiện Context Precision +13.9%, Context Recall +15.0%, Answer Relevancy +8.2% so với Naive. Faithfulness giảm nhẹ -4% (trade-off của LLM generation). Biggest win: extractive prompt recovery +13% faithfulness từ V1→V2.

2. **Biggest win — module nào, tại sao:** M2 (Hybrid Search) + M3 (Reranking) cho retrieval quality. Prompt engineering (V2) cho generation quality. Kết hợp cả hai tạo pipeline cân bằng.

3. **Case study — 1 failure, Error Tree walkthrough:** "Lợi nhuận gộp" — Faithfulness = 0.0 ở cả V1 và V2. Context đúng, Query OK → Root cause: LLM domain prior quá mạnh. Fix: cần Chain-of-Verification hoặc extractive-only mode.

4. **Next optimization nếu có thêm 1 giờ:**
   - Chain-of-Verification: verify claims against context trước khi trả lời
   - Table-aware chunking cho BCTC (giữ cấu trúc bảng)
   - Metadata filter cho câu hỏi tài chính
   - Thử gpt-4o thay gpt-4o-mini cho generation
