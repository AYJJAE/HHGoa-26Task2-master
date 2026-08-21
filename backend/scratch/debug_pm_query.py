import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources, process_rag_pipeline

resources.initialize()

q = "who is the prime minister of india"
print(f"\n==========================================")
print(f"QUERY: '{q}'")

route = resources.router.route_query(q)
strategy = route.get("strategy", {})
print(f"ROUTER: {route}")

# Check chunking strategy in strategy
print(f"STRATEGY: {strategy}")

# Retrieve
retrieval_res = resources.retriever.retrieve(q, strategy)
print(f"\nRETRIEVAL RESULTS ({len(retrieval_res['results'])} chunks):")
for idx, r in enumerate(retrieval_res["results"]):
    print(f"--- Chunk {idx+1} [ID: {r.get('id')}] (dense: {r.get('dense_score')}, sparse: {r.get('sparse_score')}, rrf: {r.get('score'):.4f}) ---")
    print(f"Text: {r.get('payload', {}).get('text')}")

# Context gate
from pipeline.context_gate import is_context_sufficient, _check_semantic_conflict, calculate_relevance_score
conflict, reason = _check_semantic_conflict(q, retrieval_res["results"])
print(f"\nCONFLICT CHECK: has_conflict={conflict}, reason='{reason}'")

rel_score, max_sim, overlap, supporting_chunks = calculate_relevance_score(q, retrieval_res["results"])
print(f"RELEVANCE: rel_score={rel_score}, max_sim={max_sim}, overlap={overlap}, supporting_chunks={supporting_chunks}")

c_val = is_context_sufficient(q, retrieval_res["results"], resources.embedder)
print(f"CONTEXT SUFFICIENT: {c_val.sufficient}, reason='{c_val.reason}'")

# Pipeline
res = process_rag_pipeline(q, debug=True)
print(f"\nPIPELINE RESULT:")
print(json.dumps(res, indent=2, ensure_ascii=False))
