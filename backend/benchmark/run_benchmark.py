import time
import json
import numpy as np
import sys
import os

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from pipeline.generation import GenerationPipeline
from pipeline.grounding import GroundingValidator
from pipeline.dataset_client import fetch_msmarco_xi_dataset

def calculate_percentiles(latencies):
    if not latencies:
        return {"P50": 0.0, "P70": 0.0, "P100": 0.0}
    return {
        "P50": round(np.percentile(latencies, 50), 2),
        "P70": round(np.percentile(latencies, 70), 2),
        "P100": round(np.max(latencies), 2)
    }

def run_benchmark(num_test_queries=100, run_generation=True):
    print("Initializing Benchmark Harness...")
    embedder = EmbeddingPipeline()
    store = VectorStore(collection_name="msmarco_xi", dense_dim=embedder.dense_dim)
    retriever = RetrievalPipeline(embedder, store)
    
    num_test_queries = 100
    print(f"Loading IndicMSMARCO dataset for benchmark...")
    dataset = fetch_msmarco_xi_dataset(num_test_queries)
    test_queries = []
    gold_passage_ids = []
    
    for i, record in enumerate(dataset):
        test_queries.append(record.get('query', ''))
        
        # The canonical source_passage_id is passage_{query_id} as defined in chunking
        query_id = str(record.get('query_id', ''))
        gold_passage_ids.append(f"passage_{query_id}")
        
    print(f"Loaded {len(test_queries)} queries.")
    
    # Define Configurations
    configurations = [
        {"name": "Hybrid RRF (Passage)", "dense_weight": 0.5, "sparse_weight": 0.5, "chunk_strategy": "passage", "rerank": False},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nRunning Configuration: {config['name']}")
        recalls_1 = []
        recalls_5 = []
        recalls_10 = []
        mrrs = []
        
        all_latencies = {
            "query_embedding_ms": [],
            "dense_retrieval_ms": [],
            "sparse_retrieval_ms": [],
            "rrf_fusion_ms": [],
            "reranking_ms": [],
            "context_assembly_ms": [],
            "total_retrieval_ms": []
        }
        
        for q_idx, query in enumerate(test_queries):
            gold_id = gold_passage_ids[q_idx]
            
            strategy = {
                "dense_weight": config["dense_weight"],
                "sparse_weight": config["sparse_weight"],
                "preferred_chunk_strategy": config["chunk_strategy"],
                "top_k_retrieve": 60, # Retrieve more so we can filter up to 50
                "top_k_rerank": 5 if config["rerank"] else 0,
                "rerank_candidates": config.get("rerank_candidates", 20)
            }
            
            # If the configuration requires a specific model, we might need to recreate Retriever
            # But creating retriever in the inner loop is extremely slow!
            # Let's recreate it only if the model changed.
            if config["rerank"]:
                current_model = getattr(retriever.reranker, "model_name", None) # this is generic huggingface cross encoder 
                # we'll just check if we have the right model name
                # CrossEncoder doesn't easily expose model_name. We'll store it on the object when creating it.
                if not hasattr(retriever, "current_model_name") or retriever.current_model_name != config["reranker_model"]:
                    retriever = RetrievalPipeline(embedder, store, reranker_model=config["reranker_model"])
                    retriever.current_model_name = config["reranker_model"]
            
            try:
                # Execute Retrieval
                res = retriever.retrieve(query, strategy)
                final_res = res["results"]
                    
                # Collect metrics
                for k, v in res["latency_ms"].items():
                    if config["rerank"] or k != "reranking_ms": # Don't log rerank time if off
                        all_latencies[k].append(v)
                        
                # Calculate Quality
                retrieved_ids = []
                for r in final_res:
                    pid = r['payload'].get('source_passage_id', r['payload'].get('parent_id'))
                    if pid not in retrieved_ids:
                        retrieved_ids.append(pid)
                
                hit_at_1 = 1 if len(retrieved_ids) > 0 and retrieved_ids[0] == gold_id else 0
                hit_at_5 = 1 if gold_id in retrieved_ids[:5] else 0
                hit_at_10 = 1 if gold_id in retrieved_ids[:10] else 0
                
                try:
                    rank = retrieved_ids.index(gold_id) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0
                    
            except Exception as e:
                # Safely print without UnicodeEncodeError on windows console
                print(f"Error evaluating query (index {q_idx}): {type(e).__name__} - {str(e).encode('ascii', 'replace').decode('ascii')}")
                hit_at_1 = 0
                hit_at_5 = 0
                hit_at_10 = 0
                mrr = 0.0
                
            recalls_1.append(hit_at_1)
            recalls_5.append(hit_at_5)
            recalls_10.append(hit_at_10)
            mrrs.append(mrr)
            
        # Aggregate stats
        config_results = {
            "quality": {
                "Recall@1": round(np.mean(recalls_1) * 100, 2),
                "Recall@5": round(np.mean(recalls_5) * 100, 2),
                "Recall@10": round(np.mean(recalls_10) * 100, 2),
                "MRR": round(np.mean(mrrs), 4)
            },
            "latency": {k: calculate_percentiles(v) for k, v in all_latencies.items() if v}
        }
        
        results[config["name"]] = config_results
        
    # Write JSON
    os.makedirs("results", exist_ok=True)
    with open("results/latest.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Write MD Report
    with open("results/latest.md", "w") as f:
        f.write("# RAG Benchmark Results\n\n")
        f.write("## Retrieval Quality\n\n")
        f.write("| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for name, data in results.items():
            q = data["quality"]
            f.write(f"| {name} | {q['Recall@1']}% | {q['Recall@5']}% | {q['Recall@10']}% | {q['MRR']} |\n")
            
        f.write("\n## Latency (P50 / P70 / P100 in ms)\n\n")
        f.write("| Configuration | Total RAG | Embedding | Dense | Sparse | RRF | Reranking | Context |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for name, data in results.items():
            l = data["latency"]
            total = l.get("total_retrieval_ms", {})
            emb = l.get("query_embedding_ms", {})
            dense = l.get("dense_retrieval_ms", {})
            sparse = l.get("sparse_retrieval_ms", {})
            rrf = l.get("rrf_fusion_ms", {})
            rerank = l.get("reranking_ms", {"P50": "-", "P70": "-", "P100": "-"})
            ctx = l.get("context_assembly_ms", {})
            
            f.write(f"| {name} | {total.get('P50', '-')} / {total.get('P70', '-')} / {total.get('P100', '-')} | ")
            f.write(f"{emb.get('P50', '-')} / {emb.get('P70', '-')} / {emb.get('P100', '-')} | ")
            f.write(f"{dense.get('P50', '-')} / {dense.get('P70', '-')} / {dense.get('P100', '-')} | ")
            f.write(f"{sparse.get('P50', '-')} / {sparse.get('P70', '-')} / {sparse.get('P100', '-')} | ")
            f.write(f"{rrf.get('P50', '-')} / {rrf.get('P70', '-')} / {rrf.get('P100', '-')} | ")
            if isinstance(rerank, dict):
                f.write(f"{rerank.get('P50', '-')} / {rerank.get('P70', '-')} / {rerank.get('P100', '-')} | ")
            else:
                f.write(f"{rerank} | ")
            f.write(f"{ctx.get('P50', '-')} / {ctx.get('P70', '-')} / {ctx.get('P100', '-')} |\n")

    print("Benchmark complete. Results saved to benchmark/results/latest.json and latest.md")

if __name__ == "__main__":
    run_benchmark()
