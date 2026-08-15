from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any
import uuid

class VectorStore:
    def __init__(self, collection_name: str = "msmarco_xi", dense_dim: int = 384):
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        # Using local persistent storage instead of requiring Docker to ensure it runs out of the box
        # Production deployment would use a remote URL
        self.client = QdrantClient(path="./qdrant_db")
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection '{self.collection_name}'...")
            
            # We configure hybrid search: dense vector + sparse vector
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
            
            # Create payload indices for fast filtering
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
            print("Collection created successfully.")

    def insert_chunks(self, chunks: List[Dict[str, Any]], dense_vectors: List[List[float]], sparse_vectors: List[Dict[int, float]]):
        """
        Insert a batch of chunks with their vectors into Qdrant.
        """
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            
            # Generate a UUID based on the chunk_id to avoid duplicates if re-ingesting
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
                    **chunk["metadata"] # includes language, source, etc.
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Inserted {len(points)} points into Qdrant.")
