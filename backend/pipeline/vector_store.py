from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any, Optional
import uuid
import os

class VectorStore:
    def __init__(self, collection_name: str = "msmarco_xi", dense_dim: int = 384, in_memory: bool = False):
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_db"))
            local_path = os.environ.get("QDRANT_LOCAL_PATH", default_path)
            try:
                self.client = QdrantClient(path=local_path)
            except Exception as e:
                print(f"Notice: Qdrant local path locked or unavailable ({e}). Falling back to in-memory store.")
                self.client = QdrantClient(":memory:")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            vectors_config = {
                "dense": rest.VectorParams(
                    size=self.dense_dim,
                    distance=rest.Distance.COSINE
                )
            }
            sparse_vectors_config = {
                "sparse": rest.SparseVectorParams(
                    index=rest.SparseIndexParams(on_disk=False)
                )
            }
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="language",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_strategy",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

    def insert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[int, float]]):
        """Insert a batch of chunks with their vectors into Qdrant."""
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
            sparse_indices = list(sparse.keys())
            sparse_values = list(sparse.values())
            
            point = rest.PointStruct(
                id=point_id,
                vector={
                    "dense": dense,
                    "sparse": rest.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "chunk_strategy": chunk["chunk_strategy"],
                    **chunk["metadata"]
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Inserted {len(points)} points into Qdrant.")
