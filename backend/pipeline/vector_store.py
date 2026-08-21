import os
import uuid
from typing import List, Dict, Any, Optional
from pipeline.faiss_index import FAISSVectorStore

class VectorStoreCountResult:
    def __init__(self, count: int):
        self.count = count

class VectorStoreClientWrapper:
    """Compatibility client wrapper exposing .count() for FastAPI health/status endpoints."""
    def __init__(self, store: 'VectorStore'):
        self._store = store

    def count(self, collection_name: Optional[str] = None) -> VectorStoreCountResult:
        return VectorStoreCountResult(self._store.faiss_store.count())


class VectorStore:
    """Engineered resident In-Memory Vector Store backed by FAISS and BM25."""
    def __init__(self, collection_name: str = "msmarco_xi", dense_dim: int = 384, in_memory: bool = True):
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.faiss_store = FAISSVectorStore(dense_dim=dense_dim)
        self.client = VectorStoreClientWrapper(self)

    def insert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[int, float]]):
        """Insert a batch of chunks with their dense and sparse vectors into the resident store."""
        self.faiss_store.insert_chunks(chunks, dense_vectors, sparse_vectors)
        print(f"Inserted {len(chunks)} points into resident FAISS store (Total: {self.faiss_store.count()}).")

    def search_dense(self, dense_vector: List[float], top_k: int = 10, chunk_strategy: Optional[str] = None):
        return self.faiss_store.search_dense(dense_vector, top_k=top_k, chunk_strategy=chunk_strategy)

    def search_sparse(self, sparse_vector: Dict[int, float], top_k: int = 10, chunk_strategy: Optional[str] = None):
        return self.faiss_store.search_sparse(sparse_vector, top_k=top_k, chunk_strategy=chunk_strategy)
