"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4+M5.

Flow:  M1 (Chunking) → M5 (Enrichment) → M2 (Hybrid Search) → M3 (Reranking)
       → LLM Generate → M4 (Evaluation)

Bonus:
  - Enrichment pipeline integrated (+3)
  - Latency breakdown per step   (+2)
  - LLM generation for higher faithfulness (+5)
"""

import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


# ─── Latency tracker ────────────────────────────────────────
_latency: dict[str, float] = {}


def _timed(step_name: str):
    """Return a simple start/stop helper to measure elapsed time."""
    class _Timer:
        def start(self):
            self.t0 = time.perf_counter()
            return self
        def stop(self):
            self.elapsed = time.perf_counter() - self.t0
            _latency[step_name] = self.elapsed
            return self.elapsed
    return _Timer()


# ─── LLM answer generation ──────────────────────────────────

def _generate_answer(query: str, contexts: list[str]) -> str:
    """Generate answer using OpenAI gpt-4o-mini with strict extractive prompt."""
    if not contexts:
        return "Không tìm thấy thông tin."

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Number contexts for traceability
            numbered_contexts = []
            for i, ctx in enumerate(contexts, 1):
                numbered_contexts.append(f"[Nguồn {i}]: {ctx}")
            context_str = "\n\n---\n\n".join(numbered_contexts)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý trích xuất thông tin. Nhiệm vụ: trả lời câu hỏi CHỈ DỰA TRÊN context được cung cấp.\n\n"
                            "QUY TẮC BẮT BUỘC:\n"
                            "1. CHỈ sử dụng thông tin CÓ TRONG context. TUYỆT ĐỐI KHÔNG thêm kiến thức bên ngoài.\n"
                            "2. Trích dẫn hoặc diễn giải sát nghĩa từ context. Ghi rõ [Nguồn X] khi trích dẫn.\n"
                            "3. Nếu context KHÔNG chứa đủ thông tin → trả lời: 'Không tìm thấy thông tin trong tài liệu.'\n"
                            "4. KHÔNG suy luận, KHÔNG giải thích thêm, KHÔNG đưa ví dụ ngoài context.\n"
                            "5. Trả lời bằng tiếng Việt, ngắn gọn, chính xác, đúng trọng tâm câu hỏi."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}\n\nHãy trả lời CHỈ dựa trên context trên:",
                    },
                ],
                max_tokens=500,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠️  LLM generation failed ({e}), using context fallback")

    # Fallback: return top context directly
    return contexts[0]


# ─── Build pipeline ──────────────────────────────────────────

def build_pipeline(use_enrichment: bool = True):
    """Build production RAG pipeline.

    Args:
        use_enrichment: Whether to run M5 enrichment. Set False for speed.

    Returns:
        (search, reranker) tuple ready for querying.
    """
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # ── Step 1: Load & Chunk (M1) ──
    print("\n[1/4] Chunking documents (M1)...")
    t = _timed("m1_chunking").start()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    elapsed = t.stop()
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents  ({elapsed:.1f}s)")

    # ── Step 2: Enrichment (M5) — rule-based only for speed ──
    if use_enrichment:
        print("\n[2/4] Enriching chunks (M5) — rule-based...")
        t = _timed("m5_enrichment").start()
        try:
            # Force rule-based fallback by temporarily clearing the API key in the module
            # This avoids 279× LLM calls that would take forever
            import src.m5_enrichment as m5_mod
            orig_key = m5_mod.OPENAI_API_KEY
            m5_mod.OPENAI_API_KEY = ""  # force rule-based fallback (no LLM calls)
            try:
                enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
            finally:
                m5_mod.OPENAI_API_KEY = orig_key  # restore for later use

            if enriched:
                # Merge enriched text + HyQA questions into indexable text
                for i, e in enumerate(enriched):
                    hyqa_text = " ".join(e.hypothesis_questions) if e.hypothesis_questions else ""
                    combined = e.enriched_text
                    if hyqa_text:
                        combined = f"{combined}\n\n[HyQA] {hyqa_text}"
                    all_chunks[i] = {
                        "text": combined,
                        "metadata": e.auto_metadata,
                    }
                elapsed = t.stop()
                print(f"  ✓ Enriched {len(enriched)} chunks  ({elapsed:.1f}s)")
            else:
                elapsed = t.stop()
                print(f"  ⚠️  Enrichment returned empty — using raw chunks  ({elapsed:.1f}s)")
        except Exception as ex:
            elapsed = t.stop()
            print(f"  ⚠️  Enrichment failed ({ex}) — using raw chunks  ({elapsed:.1f}s)")
    else:
        print("\n[2/4] Skipping enrichment (M5)")

    # ── Step 3: Index (M2) — BM25 + Dense ──
    print("\n[3/4] Indexing — BM25 + Dense (M2)...")
    t = _timed("m2_indexing").start()
    search = HybridSearch()
    search.index(all_chunks)
    elapsed = t.stop()
    print(f"  ✓ Indexed {len(all_chunks)} chunks  ({elapsed:.1f}s)")

    # ── Step 4: Load reranker (M3) ──
    print("\n[4/4] Loading reranker (M3)...")
    t = _timed("m3_reranker_load").start()
    reranker = CrossEncoderReranker()
    # Warm up: trigger lazy model load now
    reranker.rerank("test", [{"text": "test", "score": 0.0, "metadata": {}}])
    elapsed = t.stop()
    print(f"  ✓ Reranker ready (backend={reranker._backend})  ({elapsed:.1f}s)")

    return search, reranker


# ─── Query rewrite + augmentation ────────────────────────────

def _rewrite_query(query: str) -> list[str]:
    """Rewrite query into multiple search queries for better recall.

    Uses LLM to generate augmented queries. Falls back to original query only.
    """
    if not OPENAI_API_KEY:
        return [query]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia viết lại câu hỏi tìm kiếm. "
                        "Cho câu hỏi gốc, hãy viết lại thành 2 phiên bản khác nhau để tìm kiếm hiệu quả hơn.\n"
                        "Quy tắc:\n"
                        "- Mỗi phiên bản trên 1 dòng\n"
                        "- Giữ nguyên ý nghĩa nhưng dùng từ khóa khác\n"
                        "- CHỈ trả về 2 dòng, không giải thích"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        rewrites = [
            line.strip().lstrip("0123456789.-) ")
            for line in resp.choices[0].message.content.strip().split("\n")
            if line.strip()
        ]
        # Return original + rewrites (max 3 total)
        return [query] + rewrites[:2]
    except Exception:
        return [query]


# ─── Query pipeline ─────────────────────────────────────────

def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through the full pipeline.

    Flow: query rewrite → multi-query hybrid search → rerank → LLM generate answer.

    Returns:
        (answer, contexts) tuple.
    """
    # Step 1: Query rewrite + augmentation
    queries = _rewrite_query(query)

    # Step 2: Primary retrieval with original query
    results = search.search(query)
    all_docs = {}
    for r in results:
        all_docs[r.text] = {"text": r.text, "score": r.score, "metadata": r.metadata}

    # Augmented retrieval: boost with rewritten queries (only add new docs, don't lose originals)
    for q in queries[1:]:  # skip original query
        aug_results = search.search(q)
        for r in aug_results:
            if r.text in all_docs:
                # Boost existing doc's score
                all_docs[r.text]["score"] += r.score * 0.3
            else:
                # Add new doc with lower weight
                all_docs[r.text] = {"text": r.text, "score": r.score * 0.5, "metadata": r.metadata}

    docs = list(all_docs.values())

    # Step 3: Rerank via cross-encoder (using original query for relevance)
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # Step 4: Generate answer via LLM
    answer = _generate_answer(query, contexts)
    return answer, contexts


# ─── Evaluation ──────────────────────────────────────────────

def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set with RAGAS metrics."""
    print("\n" + "-" * 60)
    print("[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t = _timed("query_all").start()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:60]}...")
    elapsed = t.stop()
    print(f"  ✓ {len(test_set)} queries answered  ({elapsed:.1f}s)")

    print("\n[Eval] Running RAGAS evaluation (M4)...")
    t = _timed("m4_ragas").start()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    elapsed = t.stop()
    print(f"  ✓ RAGAS evaluation done  ({elapsed:.1f}s)")

    # ── Print scores ──
    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        icon = "✓" if s >= 0.75 else "✗"
        print(f"  {icon} {m}: {s:.4f}")

    # ── Failure analysis ──
    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    # ── Latency breakdown (Bonus +2) ──
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for step, elapsed in _latency.items():
        print(f"  {step:<25} {elapsed:>8.1f}s")
    total = sum(_latency.values())
    print(f"  {'─' * 25} {'─' * 8}")
    print(f"  {'TOTAL':<25} {total:>8.1f}s")

    # Save latency to report
    report_path = "ragas_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        report["latency_breakdown"] = {k: round(v, 2) for k, v in _latency.items()}
        report["total_latency_s"] = round(total, 2)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return results


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline(use_enrichment=True)
    evaluate_pipeline(search, reranker)
    print(f"\n⏱️  Total wall time: {time.time() - start:.1f}s")
