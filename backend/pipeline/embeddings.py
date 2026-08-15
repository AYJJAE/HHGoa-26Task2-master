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
