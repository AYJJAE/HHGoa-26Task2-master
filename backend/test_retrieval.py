import sys, os
from pipeline.dataset_client import fetch_msmarco_xi_dataset
sys.path.append(os.path.dirname(os.path.abspath('benchmark/run_benchmark.py')))
from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore

embedder = EmbeddingPipeline()
store = VectorStore(collection_name='msmarco_xi', dense_dim=embedder.dense_dim)
retriever = RetrievalPipeline(embedder, store, reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2')

dataset = fetch_msmarco_xi_dataset(100)
strategy = {'dense_weight': 0.5, 'sparse_weight': 0.5, 'preferred_chunk_strategy': 'passage', 'top_k_retrieve': 60, 'top_k_rerank': 5, 'rerank_candidates': 50}

for idx, record in enumerate(dataset):
    query = record['query']
    try:
        res = retriever.retrieve(query, strategy)
    except Exception as e:
        print(f'Error at {idx}: {type(e).__name__} - {str(e).encode("ascii", "replace").decode("ascii")}')
        sys.exit(1)
print('Success!')
