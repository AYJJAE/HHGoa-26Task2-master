import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from datasets import load_dataset
from qdrant_client.http import models as rest

def generate_diagnostic_report():
    print("Initializing Diagnostic Harness...")
    embedder = EmbeddingPipeline()
    store = VectorStore(collection_name="msmarco_xi", dense_dim=embedder.dense_dim)
    retriever = RetrievalPipeline(embedder, store, reranker_model="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    
    print("Loading local mock dataset...")
    dataset = load_dataset("json", data_files="data/mock_dataset_100.json", split="train")
    
    report = "# Retrieval Diagnostic Report\n\n"
    
    for i, record in enumerate(dataset):
        if i >= 10:
            break
            
        query = record.get('query', record.get('Eng_Query', ''))
        query_id = str(record.get('query_id', ''))
        is_selected = record.get('passages', {}).get('is_selected', [])
        
        gold_index = is_selected.index(1) if 1 in is_selected else 0
        gold_id = f"{query_id}_p{gold_index}"
        
        # Get the actual gold passage text from the dataset for reference
        passages_info = record.get('passages', {})
        translated_passages = passages_info.get('Translated_passages', [])
        if not translated_passages:
            translated_passages = passages_info.get('English_passages', [])
        gold_text = translated_passages[gold_index] if gold_index < len(translated_passages) else ""
        
        report += f"## Query {i+1}: {query}\n"
        report += f"**Gold ID**: {gold_id}\n\n"
        report += f"**Gold Passage**: {gold_text}\n\n"
        
        # 1. Embed Query
        dense_q, sparse_q = retriever.embedder.embed_queries([query])
        dense_vec = dense_q[0]
        sparse_vec = sparse_q[0]
        
        # 2. Dense Retrieval
        dense_results = retriever.store.client.query_points(
            collection_name=retriever.store.collection_name,
            query=dense_vec,
            using="dense",
            limit=10
        ).points
        
        # 3. Sparse Retrieval
        sparse_indices = list(sparse_vec.keys())
        sparse_values = list(sparse_vec.values())
        sparse_results = retriever.store.client.query_points(
            collection_name=retriever.store.collection_name,
            query=rest.SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=10
        ).points
        
        # 4. Hybrid (RRF)
        fused_results = retriever._reciprocal_rank_fusion(dense_results, sparse_results)[:10]
        
        # 5. Reranked
        # Rerank the top 10 fused results
        candidates = fused_results[:]
        cross_inp = [[query, c['payload']['text']] for c in candidates]
        scores = retriever.reranker.predict(cross_inp)
        for j in range(len(scores)):
            candidates[j]['rerank_score'] = float(scores[j])
        reranked_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:10]
        
        # Helper to format list
        def format_candidates(title, candidates_list):
            out = f"### {title}\n"
            for rank, c in enumerate(candidates_list):
                parent_id = c.payload['parent_id'] if hasattr(c, 'payload') else c['payload']['parent_id']
                score = c.score if hasattr(c, 'score') else c.get('score', c.get('rerank_score', 0.0))
                text = c.payload['text'] if hasattr(c, 'payload') else c['payload']['text']
                match = "✅" if parent_id == gold_id else "❌"
                out += f"{rank+1}. [{match}] **{parent_id}** (Score: {score:.4f}) - {text[:100]}...\n"
            return out + "\n"

        report += format_candidates("Dense Candidates (Top 10)", dense_results)
        report += format_candidates("Sparse Candidates (Top 10)", sparse_results)
        report += format_candidates("Hybrid RRF Candidates (Top 10)", fused_results)
        report += format_candidates("Reranked Candidates (Top 10)", reranked_results)
        
        # Checks
        dense_ids = [c.payload['parent_id'] for c in dense_results]
        sparse_ids = [c.payload['parent_id'] for c in sparse_results]
        hybrid_ids = [c['payload']['parent_id'] for c in fused_results]
        reranked_ids = [c['payload']['parent_id'] for c in reranked_results]
        
        retrieved_before = gold_id in hybrid_ids
        retrieved_after = gold_id in reranked_ids
        
        report += f"**Gold Retrieved Before Reranking?**: {retrieved_before}\n"
        report += f"**Gold Retrieved After Reranking?**: {retrieved_after}\n\n"
        report += "---\n\n"
        
    with open("results/diagnostic_report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Diagnostic complete. Report written to results/diagnostic_report.md")

if __name__ == "__main__":
    generate_diagnostic_report()
