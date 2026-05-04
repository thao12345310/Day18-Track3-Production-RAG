# Group Report — Lab 18: Production RAG

**Nhóm:** Lab 18 - Track 3  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Thảo | M1: Chunking | ☑ | 8/8 |
| Thảo | M2: Hybrid Search | ☑ | 5/5 |
| Thảo | M3: Reranking | ☑ | 5/5 |
| Thảo | M4: Evaluation | ☑ | 4/4 |

**Tổng tests:** 37/37 passed ✓

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8270 | 0.6567 | -0.1703 ↓ |
| Answer Relevancy | 0.3250 | 0.3718 | +0.0468 ↑ |
| Context Precision | 0.6792 | 0.8167 | +0.1375 ↑ |
| Context Recall | 0.5333 | 0.7500 | +0.2167 ↑ |

**Đạt yêu cầu:** Context Precision (0.8167 ≥ 0.75) ✓ và Context Recall (0.7500 ≥ 0.75) ✓

## Key Findings

1. **Biggest improvement:** Context Recall tăng +0.2167 (từ 0.53 → 0.75). Hybrid Search (BM25 + Dense + RRF) kết hợp với Cross-Encoder Reranking giúp retrieve được nhiều chunks liên quan hơn đáng kể so với dense-only search. BM25 bắt được keyword matches mà dense model bỏ sót, và reranker lọc bỏ noise.

2. **Biggest challenge:** Faithfulness giảm -0.17 khi thêm LLM generation. Naive baseline trả context trực tiếp (faithfulness cao vì "answer" = context), còn production pipeline dùng gpt-4o-mini để generate → model thêm kiến thức ngoài context, gây hallucination. Đặc biệt với câu hỏi về BCTC (bảng biểu, số liệu), LLM có strong domain prior về kế toán.

3. **Surprise finding:** Enrichment (M5) rule-based chạy gần như tức thì (0.01s) nhưng vẫn cải thiện context quality nhờ contextual prepending và HyQA questions. Tuy nhiên, bottleneck lớn nhất là query time (289s cho 20 queries) — mỗi query cần encode + search + rerank + LLM generate, trung bình ~14.5s/query.

## Latency Breakdown

| Step | Time | % Total |
|------|------|---------|
| M1: Chunking | 0.0s | 0% |
| M5: Enrichment | 0.01s | 0% |
| M2: Indexing | 49.8s | 9.4% |
| M3: Reranker Load | 16.6s | 3.1% |
| Query (20 queries) | 289.1s | 54.6% |
| M4: RAGAS Eval | 174.4s | 32.9% |
| **Total** | **529.9s** | **100%** |

## Architecture

```
Documents → M1 (Hierarchical Chunking)
         → M5 (Rule-based Enrichment: contextual + HyQA + metadata)
         → M2 (BM25 Vietnamese + Dense bge-m3 + Qdrant → RRF Fusion)
         → M3 (Cross-Encoder Reranking: BAAI/bge-reranker-v2-m3)
         → LLM Generation (gpt-4o-mini)
         → M4 (RAGAS Evaluation)
```

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** Production pipeline cải thiện Context Precision +13.8% và Context Recall +21.7%. Tuy nhiên Faithfulness giảm 17% do LLM hallucination. Tổng thể: retrieval tốt hơn nhiều, generation cần cải thiện.

2. **Biggest win — module nào, tại sao:** M2 (Hybrid Search) + M3 (Reranking) là biggest wins. BM25 tiếng Việt (underthesea segmentation) bắt keyword matches, Dense (bge-m3) bắt semantic similarity, RRF fusion kết hợp cả hai. Cross-encoder reranker lọc bỏ irrelevant chunks, đưa context precision lên 0.82.

3. **Case study — 1 failure, Error Tree walkthrough:** "Báo cáo tài chính gồm những thành phần nào?" — Faithfulness = 0.0. Context đúng (BCTC chunks retrieved), Query OK → Root cause: LLM thêm kiến thức kế toán ngoài context. Fix: tighten prompt, giảm temperature, dùng extractive QA.

4. **Next optimization nếu có thêm 1 giờ:**
   - Cải thiện prompt template: few-shot examples + strict instruction
   - Table-aware chunking cho BCTC (giữ nguyên cấu trúc bảng)
   - Chain-of-Verification: verify claims against context
   - Semantic chunking thay vì fixed-size cho hierarchical children
