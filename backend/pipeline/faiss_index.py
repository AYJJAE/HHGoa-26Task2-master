import time
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class InMemoryBM25:
    """Fast, resident in-memory BM25 index with precomputed IDF and term frequencies."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.inverted_index: Dict[int, List[Tuple[int, float]]] = defaultdict(list) # term_id -> [(doc_idx, tf)]
        self.idf: Dict[int, float] = {}
        self.doc_ids: List[str] = []

    def build_index(self, doc_ids: List[str], sparse_vectors: List[Dict[int, float]]):
        self.doc_ids = doc_ids
        self.corpus_size = len(sparse_vectors)
        if self.corpus_size == 0:
            return

        total_len = 0
        doc_frequencies: Dict[int, int] = defaultdict(int)

        self.doc_lens = []
        for doc_idx, svec in enumerate(sparse_vectors):
            doc_len = sum(svec.values())
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for term_idx, weight in svec.items():
                self.inverted_index[term_idx].append((doc_idx, weight))
                doc_frequencies[term_idx] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 1.0

        # Calculate standard Lucene/BM25 IDF
        for term_idx, df in doc_frequencies.items():
            self.idf[term_idx] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query_sparse: Dict[int, float], top_k: int = 10, filter_fn=None) -> List[Tuple[int, float]]:
        if not query_sparse or self.corpus_size == 0:
            return []

        scores = defaultdict(float)
        for term_idx, q_weight in query_sparse.items():
            if term_idx not in self.inverted_index:
                continue
            idf_val = self.idf.get(term_idx, 0.0)
            for doc_idx, tf in self.inverted_index[term_idx]:
                if filter_fn and not filter_fn(doc_idx):
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf_val * (numerator / denominator) * q_weight

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class FAISSVectorStore:
    """Resident In-Memory FAISS Vector Store + Lexical BM25 Search Engine."""
    def __init__(self, dense_dim: int = 384):
        self.dense_dim = dense_dim
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        # FAISS Index with Inner Product (Cosine similarity on normalized vectors)
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dense_dim)
        else:
            self.index = None
            self.dense_matrix: Optional[np.ndarray] = None
            
        self.bm25 = InMemoryBM25()
        self.sparse_vectors: List[Dict[int, float]] = []

    def count(self) -> int:
        return len(self.chunks)

    def insert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[int, float]]):
        if not chunks:
            return

        start_idx = len(self.chunks)
        dense_arr = np.array(dense_vectors, dtype=np.float32)
        # Normalize vectors for exact cosine similarity with Inner Product
        faiss.normalize_L2(dense_arr) if FAISS_AVAILABLE else None

        if FAISS_AVAILABLE:
            self.index.add(dense_arr)
        else:
            if self.dense_matrix is None:
                self.dense_matrix = dense_arr
            else:
                self.dense_matrix = np.vstack([self.dense_matrix, dense_arr])

        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk["chunk_id"]] = idx

        self.sparse_vectors.extend(sparse_vectors)
        doc_ids = [c["chunk_id"] for c in self.chunks]
        self.bm25.build_index(doc_ids, self.sparse_vectors)

    def search_dense(self, dense_vector: List[float], top_k: int = 10, chunk_strategy: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        if len(self.chunks) == 0:
            return [], (time.perf_counter() - t0) * 1000.0

        query_arr = np.array([dense_vector], dtype=np.float32)
        fetch_k = min(len(self.chunks), max(top_k * 10, 100))
        if FAISS_AVAILABLE:
            faiss.normalize_L2(query_arr)
            scores, indices = self.index.search(query_arr, fetch_k)
            scores = scores[0]
            indices = indices[0]
        else:
            norm_q = query_arr / (np.linalg.norm(query_arr) + 1e-9)
            norm_matrix = self.dense_matrix / (np.linalg.norm(self.dense_matrix, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(norm_matrix, norm_q.T).flatten()
            indices = np.argsort(sims)[::-1][:fetch_k]
            scores = sims[indices]

        results = []
        fallback_results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            item = {
                "id": chunk["chunk_id"],
                "score": float(score),
                "payload": {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "chunk_strategy": chunk.get("chunk_strategy", "passage"),
                    **chunk.get("metadata", {})
                }
            }
            if chunk_strategy and chunk.get("chunk_strategy") != chunk_strategy:
                fallback_results.append(item)
                continue
            results.append(item)
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for item in fallback_results:
                results.append(item)
                if len(results) >= top_k:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, elapsed_ms

    def search_sparse(self, sparse_vector: Dict[int, float], top_k: int = 10, chunk_strategy: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        if len(self.chunks) == 0 or not sparse_vector:
            return [], (time.perf_counter() - t0) * 1000.0

        fetch_k = min(len(self.chunks), max(top_k * 10, 100))
        ranked = self.bm25.search(sparse_vector, top_k=fetch_k)

        results = []
        fallback_results = []
        for doc_idx, score in ranked:
            chunk = self.chunks[doc_idx]
            item = {
                "id": chunk["chunk_id"],
                "score": float(score),
                "payload": {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "chunk_strategy": chunk.get("chunk_strategy", "passage"),
                    **chunk.get("metadata", {})
                }
            }
            if chunk_strategy and chunk.get("chunk_strategy") != chunk_strategy:
                fallback_results.append(item)
                continue
            results.append(item)
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for item in fallback_results:
                results.append(item)
                if len(results) >= top_k:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, elapsed_ms
