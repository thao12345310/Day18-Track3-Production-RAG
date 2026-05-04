"""
Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF.
===========================================================
Kết hợp BM25 keyword search (tối ưu tiếng Việt qua underthesea)
với Dense vector search (bge-m3 + Qdrant) và Reciprocal Rank Fusion
để đạt recall + precision cao hơn single-method search.

Test: pytest tests/test_m2.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    """Kết quả tìm kiếm từ bất kỳ method nào (bm25/dense/hybrid)."""
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Tách từ tiếng Việt bằng underthesea.

    BM25 cần word boundaries chính xác. Với tiếng Việt, "nghỉ phép" là 1 từ
    ghép — nếu không segment, BM25 sẽ tách thành 2 token riêng biệt và
    mất semantic meaning.

    Args:
        text: Văn bản tiếng Việt cần tách từ.

    Returns:
        Chuỗi đã tách từ, các từ phân cách bằng khoảng trắng.
        VD: "Nhân viên được nghỉ phép" → "Nhân_viên được nghỉ_phép"
    """
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


class BM25Search:
    """BM25 keyword search tối ưu cho tiếng Việt.

    Sử dụng underthesea để segment trước khi tokenize,
    đảm bảo từ ghép tiếng Việt được giữ nguyên.
    """

    def __init__(self) -> None:
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index từ danh sách chunks.

        Mỗi chunk được segment tiếng Việt trước khi tokenize
        để BM25 xử lý đúng các từ ghép.

        Args:
            chunks: List of {"text": str, "metadata": dict}.
        """
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        # Segment tiếng Việt → split thành token list cho BM25
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Tìm kiếm BM25 với query đã segment tiếng Việt.

        Args:
            query: Câu hỏi tìm kiếm.
            top_k: Số kết quả trả về tối đa.

        Returns:
            List SearchResult sorted by BM25 score descending.
        """
        if self.bm25 is None or not self.documents:
            return []

        # Segment query tiếng Việt → tokenize
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)

        # Lấy top-k indices theo score giảm dần
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
        ]


class DenseSearch:
    """Dense vector search sử dụng bge-m3 + Qdrant.

    bge-m3 là multilingual embedding model hỗ trợ tốt tiếng Việt,
    output 1024-dim vectors, dùng Qdrant làm vector database.
    """

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        # Try connecting to Qdrant server; fall back to in-memory if unavailable
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
            client.get_collections()  # test connection
            self.client = client
        except Exception:
            print("  ⚠️  Qdrant server not reachable, using in-memory mode")
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        """Lazy-load sentence-transformer encoder (bge-m3)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Encode và index chunks vào Qdrant.

        Quy trình:
        1. Tạo/reset collection với cosine distance
        2. Encode tất cả texts bằng bge-m3
        3. Upload PointStruct (id, vector, payload) vào Qdrant

        Args:
            chunks: List of {"text": str, "metadata": dict}.
            collection: Tên Qdrant collection.
        """
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Tạo collection mới (xóa cũ nếu tồn tại)
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        # Encode toàn bộ texts bằng bge-m3
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        # Tạo PointStruct và upsert vào Qdrant
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Tìm kiếm bằng dense vector similarity.

        Args:
            query: Câu hỏi tìm kiếm.
            top_k: Số kết quả trả về tối đa.
            collection: Tên Qdrant collection.

        Returns:
            List SearchResult sorted by cosine similarity descending.
        """
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                method="dense",
            )
            for hit in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge nhiều ranked lists bằng Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    Trong đó rank bắt đầu từ 1 (top result = rank 1).

    Ưu điểm: không cần normalize scores giữa các methods,
    hoạt động tốt khi kết hợp BM25 (sparse) + Dense (semantic).

    Args:
        results_list: List các ranked lists (VD: [bm25_results, dense_results]).
        k: Constant để smooth ranking (default 60, theo paper gốc).
        top_k: Số kết quả trả về sau merge.

    Returns:
        List SearchResult merged, sorted by RRF score descending, method="hybrid".
    """
    # Tích lũy RRF score cho mỗi document (keyed by text)
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "metadata": result.metadata,
                }
            # rank bắt đầu từ 0, nên +1 để rank 1-based
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending → lấy top_k
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    return [
        SearchResult(
            text=text,
            score=data["score"],
            metadata=data["metadata"],
            method="hybrid",
        )
        for text, data in sorted_docs[:top_k]
    ]


class HybridSearch:
    """Kết hợp BM25 + Dense + RRF thành hybrid search pipeline.

    Flow: Query → [BM25 top-k, Dense top-k] → RRF merge → Hybrid top-k.
    """

    def __init__(self) -> None:
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        """Index chunks vào cả BM25 và Qdrant."""
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        """Hybrid search: BM25 + Dense → RRF fusion."""
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
