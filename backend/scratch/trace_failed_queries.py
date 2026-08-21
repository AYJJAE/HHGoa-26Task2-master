import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources, process_rag_pipeline

resources.initialize()

test_queries = [
    ("Who is the Prime Minister of India?", True),
    ("What is the capital of India?", True),
    ("What is the national flower of the United States?", True),
    ("What is the capital of Rajasthan?", True),
    ("What is the height of Mount Everest?", True),
    ("What is the population of the United States?", True),
    ("What is FAISS used for?", True),
    ("What are the main beaches in North Goa?", True),
    ("What is the capital of Goa?", True),
    ("Tell me a 2 day itinerary for Goa", True),
    ("Who is the President of the United States?", False),
    ("Who is the President of France?", False),
    ("What is the recipe for chocolate lava cake?", False),
    ("Aliens on Mars", False),
]

out_lines = []

for q, expected_answerable in test_queries:
    out_lines.append(f"\n=======================================================")
    out_lines.append(f"QUERY: {q} (Expected Answerable: {expected_answerable})")
    
    # 1. Routing
    route = resources.router.route_query(q)
    strategy = route.get("strategy", {})
    out_lines.append(f"ROUTING: {route.get('intent')} | {route.get('language')} | {route.get('complexity')}")
    
    # 2. Retrieval
    retrieval_res = resources.retriever.retrieve(q, strategy)
    chunks = retrieval_res["results"]
    dense_scores = [round(float(c.get("dense_score", 0)), 3) for c in chunks]
    rrf_scores = [round(float(c.get("score", 0)), 4) for c in chunks]
    out_lines.append(f"RETRIEVAL: {len(chunks)} chunks | Dense: {dense_scores} | RRF: {rrf_scores}")
    if chunks:
        out_lines.append(f"TOP CHUNK ID: {chunks[0].get('id')}")
        out_lines.append(f"TOP CHUNK TEXT: {chunks[0].get('payload', {}).get('text', '')[:100]}...")
    
    # 3. Context gate
    from pipeline.context_gate import is_context_sufficient
    c_val = is_context_sufficient(q, chunks, resources.embedder)
    out_lines.append(f"CONTEXT SUFFICIENT: {c_val.sufficient} (Rel: {c_val.relevance_score:.2f}, Overlap: {c_val.keyword_overlap:.2f}, MaxSim: {c_val.max_similarity})")
    out_lines.append(f"CONTEXT REASON: {c_val.reason}")
    
    # 4. Full pipeline
    res = process_rag_pipeline(q, debug=True)
    status = res.get("status")
    ans = res.get("answer", "")
    conf = res.get("confidence")
    grounded = res.get("grounded")
    refusal_reason = res.get("refusal_reason")
    out_lines.append(f"PIPELINE RESULT: status={status}, confidence={conf}, grounded={grounded}, refusal_reason={refusal_reason}")
    out_lines.append(f"ANSWER: {ans[:120]}...")
    
    # Check correctness
    if expected_answerable and status == "refused":
        out_lines.append(f">>> [REGRESSION BUG] Expected answerable query was REFUSED!")
    elif not expected_answerable and status != "refused":
        out_lines.append(f">>> [GUARDRAIL BUG] Expected unanswerable query was ANSWERED!")
    else:
        out_lines.append(f"CORRECT OUTCOME: {'Answered' if expected_answerable else 'Refused'}")

with open("scratch/trace_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Trace complete. Results written to scratch/trace_results.txt")
