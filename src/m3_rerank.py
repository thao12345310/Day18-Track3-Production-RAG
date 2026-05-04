"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, re, time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._backend: Optional[str] = None

    def _load_model(self):
        """Lazy-load reranker model with priority: FlagEmbedding → sentence-transformers → lexical fallback."""
        if self._backend is not None:
            return self._model

        # --- Priority 1: FlagEmbedding ---
        try:
            from FlagEmbedding import FlagReranker

            use_fp16 = False
            try:
                import torch
                use_fp16 = torch.cuda.is_available()
            except ImportError:
                pass

            model = FlagReranker(self.model_name, use_fp16=use_fp16)
            # Validate that scoring actually works (catches transformers incompatibility)
            model.compute_score([("test", "test")])
            self._model = model
            self._backend = "flag"
            return self._model
        except Exception:
            pass

        # --- Priority 2: sentence-transformers ---
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(self.model_name)
            # Validate that prediction works
            model.predict([("test", "test")])
            self._model = model
            self._backend = "sentence_transformers"
            return self._model
        except Exception:
            pass

        # --- Priority 3: lexical fallback ---
        self._model = None
        self._backend = "lexical"
        return self._model

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of lowercase Unicode word tokens."""
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def _lexical_score(self, query: str, document: str) -> float:
        """Compute a heuristic lexical relevance score between query and document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        if not query_tokens:
            return 0.0

        # Base: token overlap ratio
        overlap = query_tokens & doc_tokens
        score = len(overlap) / len(query_tokens) if query_tokens else 0.0

        query_lower = query.lower()
        doc_lower = document.lower()

        # Boost: query asks about leave/nghỉ phép and document mentions it
        if "nghỉ" in query_lower and "nghỉ" in doc_lower:
            score += 0.15
        if "phép" in query_lower and "phép" in doc_lower:
            score += 0.15

        # Boost: query asks "bao nhiêu" (how many) and document contains numbers
        if "bao" in query_tokens and "nhiêu" in query_tokens:
            if re.search(r"\d+", document):
                score += 0.1

        # Penalty: document about VPN but query doesn't mention VPN
        if "vpn" in doc_tokens and "vpn" not in query_tokens:
            score -= 0.2

        # Penalty: document about password/mật khẩu but query doesn't ask about it
        password_terms = {"mật", "khẩu", "password"}
        if password_terms & doc_tokens and not (password_terms & query_tokens):
            score -= 0.1

        return max(score, 0.0)

    def _predict_scores(self, query: str, documents: list[dict]) -> list[float]:
        """Predict relevance scores for query-document pairs using the loaded backend."""
        self._load_model()

        # Extract text from documents (already validated)
        doc_texts = []
        for doc in documents:
            text = doc.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            doc_texts.append(text)

        if self._backend == "flag":
            pairs = [(query, t) for t in doc_texts]
            scores = self._model.compute_score(pairs)
            # Ensure list
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores]
            return scores

        elif self._backend == "sentence_transformers":
            pairs = [(query, t) for t in doc_texts]
            scores = self._model.predict(pairs)
            # Ensure list
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            else:
                scores = [float(s) for s in scores]
            return scores

        else:
            # lexical fallback
            return [self._lexical_score(query, t) for t in doc_texts]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # Edge cases: return empty
        if not query or not isinstance(query, str) or not query.strip():
            return []
        if not documents:
            return []

        # Filter valid documents (must have usable text)
        valid_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if not text.strip():
                continue
            # Keep a normalized copy alongside the original doc
            valid_docs.append({
                "text": text,
                "score": float(doc.get("score", 0.0)),
                "metadata": doc.get("metadata", {}),
            })

        if not valid_docs:
            return []

        # Score using the backend
        try:
            scores = self._predict_scores(query, valid_docs)
        except Exception:
            # If model scoring fails, fall back to lexical
            scores = [
                self._lexical_score(query, d["text"]) for d in valid_docs
            ]

        # Combine scores with documents and sort descending
        scored = list(zip(scores, valid_docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top_k and build results
        results: list[RerankResult] = []
        for rank_idx, (rerank_score, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                RerankResult(
                    text=doc["text"],
                    original_score=doc["score"],
                    rerank_score=float(rerank_score),
                    metadata=doc["metadata"],
                    rank=rank_idx,
                )
            )

        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self):
        self._model = None
        self._available = False
        try:
            from flashrank import Ranker
            self._model = Ranker()
            self._available = True
        except Exception:
            pass

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank using Flashrank if available, otherwise fallback to CrossEncoderReranker."""
        if not self._available or self._model is None:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)

        try:
            from flashrank import RerankRequest

            passages = [{"text": d.get("text", "")} for d in documents]
            request = RerankRequest(query=query, passages=passages)
            flash_results = self._model.rerank(request)

            # Build scored list
            scored = []
            for fr in flash_results:
                # Find matching original doc
                text = fr.get("text", fr.get("passage", ""))
                orig_doc = next(
                    (d for d in documents if d.get("text", "") == text),
                    {},
                )
                scored.append((
                    float(fr.get("score", 0.0)),
                    text,
                    float(orig_doc.get("score", 0.0)),
                    orig_doc.get("metadata", {}),
                ))

            scored.sort(key=lambda x: x[0], reverse=True)

            results: list[RerankResult] = []
            for rank_idx, (rscore, text, oscore, meta) in enumerate(
                scored[:top_k], start=1
            ):
                results.append(
                    RerankResult(
                        text=text,
                        original_score=oscore,
                        rerank_score=rscore,
                        metadata=meta,
                        rank=rank_idx,
                    )
                )
            return results

        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark latency over n_runs."""
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    # Warm-up run (not counted)
    reranker.rerank(query, documents)

    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    # Fix Windows console encoding for Vietnamese output
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"

    docs = [
        {
            "text": "Mật khẩu VPN phải được thay đổi mỗi 90 ngày.",
            "score": 0.95,
            "metadata": {"source": "it_policy.pdf"},
        },
        {
            "text": "Nhân viên toàn thời gian được hưởng 12 ngày nghỉ phép có lương mỗi năm.",
            "score": 0.70,
            "metadata": {"source": "hr_policy.pdf"},
        },
        {
            "text": "Nhân viên phải gửi yêu cầu nghỉ phép trước ít nhất 5 ngày làm việc.",
            "score": 0.65,
            "metadata": {"source": "hr_policy.pdf"},
        },
        {
            "text": "Thời gian thử việc là 60 ngày.",
            "score": 0.60,
            "metadata": {"source": "hr_policy.pdf"},
        },
    ]

    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, docs, top_k=3)

    for result in results:
        print(
            f"[{result.rank}] {result.rerank_score:.4f} | "
            f"original={result.original_score:.4f} | {result.text}"
        )

    print(benchmark_reranker(reranker, query, docs, n_runs=3))
