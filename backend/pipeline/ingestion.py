import os
import sys
import argparse
from pipeline.dataset_client import fetch_msmarco_xi_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.chunking import ChunkingPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore

def ingest_dataset(mode: str = "dev", max_records: int = 100):
    print(f"Starting ingestion in '{mode}' mode.")
    dataset = fetch_msmarco_xi_dataset(max_records)
    
    chunker = ChunkingPipeline(token_chunk_size=256, token_overlap=50)
    embedder = EmbeddingPipeline()
    store = VectorStore(collection_name="msmarco_xi", dense_dim=embedder.dense_dim)
    
    processed = 0
    batch_size = 10 # small batch size for embedding to avoid memory spikes
    
    batch_chunks = []
    
    for record in dataset:
        if processed >= max_records:
            break
            
        chunks = chunker.process_record(record)
        batch_chunks.extend(chunks)
        
        if len(batch_chunks) >= batch_size:
            texts = [c["text"] for c in batch_chunks]
            print(f"Embedding batch of {len(texts)} chunks...")
            
            dense_vectors, sparse_vectors = embedder.embed_documents(texts)
            store.insert_chunks(batch_chunks, dense_vectors, sparse_vectors)
            
            batch_chunks = []
            
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} records...")

    # Flush remaining
    if batch_chunks:
        texts = [c["text"] for c in batch_chunks]
        print(f"Embedding final batch of {len(texts)} chunks...")
        dense_vectors, sparse_vectors = embedder.embed_documents(texts)
        store.insert_chunks(batch_chunks, dense_vectors, sparse_vectors)

    print(f"Ingestion complete. Processed {processed} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI dataset into Qdrant")
    parser.add_argument("--mode", type=str, choices=["dev", "eval", "mock"], default="mock",
                        help="Ingestion mode. 'mock' runs local json.")
    parser.add_argument("--records", type=int, default=10, 
                        help="Number of records to ingest (default 10 for dev).")
    args = parser.parse_args()
    
    # Evaluation mode allows overriding the limit if requested, but defaults to a much larger number
    # Since MSMARCO-XI is huge, even eval might be capped at say 10k or whatever is feasible on the hardware.
    limit = args.records if args.mode == "dev" else 10000 
    
    ingest_dataset(mode=args.mode, max_records=limit)
