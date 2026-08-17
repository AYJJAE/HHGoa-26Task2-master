import os
import re
import threading
from collections import OrderedDict
from typing import List, Dict, Any, Tuple
from fastembed import TextEmbedding, SparseTextEmbedding

class EmbeddingPipeline:
    def __init__(self):
        # Dense Embedding Model: bge-m3 (Multilingual)
        # FastEmbed handles downloading and caching the model locally
        print("Loading Dense Embedding Model (BGE-M3)...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") # Using small for speed if m3 isn't available, but we can change this. 
        # Actually fastembed has `intfloat/multilingual-e5-large` or `BAAI/bge-m3`. We'll try to use a good multilingual one.
        # Let's switch to sentence-transformers for guaranteed BGE-M3 support if fastembed doesn't have it natively for dense,
        # but fastembed does have "intfloat/multilingual-e5-large" which is excellent for multilingual.
        
        # Sparse Embedding Model (BM25 or SPLADE)
        print("Loading Sparse Embedding Model...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Dimensions
        self.dense_dim = 384 # Change based on model. e.g. bge-small is 384, e5-large is 1024
        self._query_cache_size = max(0, int(os.environ.get("RAG_QUERY_EMBED_CACHE_SIZE", "256")))
        self._query_cache = OrderedDict()
        self._cache_lock = threading.RLock()

    @staticmethod
    def normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip()).casefold()

    def embed_query(self, query: str):
        """Embed one query with a bounded, thread-safe process-local LRU cache."""
        key = self.normalize_query(query)
        if self._query_cache_size:
            with self._cache_lock:
                cached = self._query_cache.get(key)
                if cached is not None:
                    self._query_cache.move_to_end(key)
                    return cached[0], cached[1], True
        dense, sparse = self.embed_queries([query])
        value = (dense[0], sparse[0])
        if self._query_cache_size:
            with self._cache_lock:
                self._query_cache[key] = value
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
        return value[0], value[1], False
        
    def embed_documents(self, documents: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Embed a list of documents into Dense and Sparse vectors.
        Returns:
            - dense_vectors: List of list of floats
            - sparse_vectors: List of dicts {index: weight} representing sparse vectors
        """
        # Generate dense embeddings (FastEmbed returns a generator, so we convert to list of lists)
        dense_generator = self.dense_model.embed(documents)
        dense_vectors = [vec.tolist() for vec in dense_generator]
        
        # Generate sparse embeddings
        sparse_generator = self.sparse_model.embed(documents)
        sparse_vectors = []
        for sparse_vec in sparse_generator:
            # sparse_vec has .indices and .values
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)}
            sparse_vectors.append(sparse_dict)
            
        return dense_vectors, sparse_vectors
        
    def embed_queries(self, queries: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Embed queries (some models require specific prefixes for queries).
        """
        # Some models use 'query: ' prefix, handled internally by fastembed for known models
        dense_generator = self.dense_model.embed(queries)
        dense_vectors = [vec.tolist() for vec in dense_generator]
        
        sparse_generator = self.sparse_model.embed(queries)
        sparse_vectors = []
        for sparse_vec in sparse_generator:
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)}
            sparse_vectors.append(sparse_dict)
            
        return dense_vectors, sparse_vectors

if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    d, s = pipeline.embed_documents(["नई दिल्ली भारत की राजधानी है।"])
    print(f"Dense dim: {len(d[0])}")
    print(f"Sparse non-zero elements: {len(s[0])}")
