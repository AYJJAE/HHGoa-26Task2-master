import os
import re
import time
import threading
from collections import OrderedDict
from typing import List, Dict, Any, Tuple
from fastembed import TextEmbedding, SparseTextEmbedding

class EmbeddingPipeline:
    def __init__(self):
        print("Loading Dense Embedding Model (BAAI/bge-small-en-v1.5)...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.model_name = "BAAI/bge-small-en-v1.5"
        
        print("Loading Sparse Embedding Model (Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        self.dense_dim = 384
        self._query_cache_size = max(0, int(os.environ.get("RAG_QUERY_EMBED_CACHE_SIZE", "256")))
        self._query_cache_ttl = float(os.environ.get("RAG_QUERY_EMBED_CACHE_TTL_SEC", "3600"))
        self._query_cache: OrderedDict[str, Tuple[Tuple[List[float], Dict[int, float]], float]] = OrderedDict()
        self._cache_lock = threading.RLock()

    @staticmethod
    def normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip()).casefold()

    def embed_query(self, query: str, use_cache: bool = True):
        """Embed one query with bounded, thread-safe LRU cache with TTL."""
        key = self.normalize_query(query)
        now = time.time()
        
        if use_cache and self._query_cache_size > 0:
            with self._cache_lock:
                if key in self._query_cache:
                    cached_val, timestamp = self._query_cache[key]
                    if now - timestamp < self._query_cache_ttl:
                        self._query_cache.move_to_end(key)
                        return cached_val[0], cached_val[1], True
                    else:
                        del self._query_cache[key]
                        
        dense, sparse = self.embed_queries([query])
        value = (dense[0], sparse[0])
        
        if use_cache and self._query_cache_size > 0:
            with self._cache_lock:
                self._query_cache[key] = (value, now)
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
                    
        return value[0], value[1], False

    def embed_documents(self, documents: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Embed documents into Dense and Sparse vectors."""
        dense_generator = self.dense_model.embed(documents)
        dense_vectors = [vec.tolist() for vec in dense_generator]
        
        sparse_generator = self.sparse_model.embed(documents)
        sparse_vectors = []
        for sparse_vec in sparse_generator:
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)}
            sparse_vectors.append(sparse_dict)
            
        return dense_vectors, sparse_vectors

    def embed_queries(self, queries: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Embed query batch."""
        dense_generator = self.dense_model.embed(queries)
        dense_vectors = [vec.tolist() for vec in dense_generator]
        
        sparse_generator = self.sparse_model.embed(queries)
        sparse_vectors = []
        for sparse_vec in sparse_generator:
            sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_vec.indices, sparse_vec.values)}
            sparse_vectors.append(sparse_dict)
            
        return dense_vectors, sparse_vectors
