import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources, process_rag_pipeline

resources.initialize()

with open("data/mock_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} records from mock_dataset.json")

refused_queries = []
answered_queries = []

for idx, record in enumerate(dataset):
    q = record.get("query", "")
    qid = record.get("query_id", f"fact_{idx}")
    expected_ans = record.get("Answer", record.get("Eng_Answer", ""))
    
    res = process_rag_pipeline(q, debug=True)
    status = res.get("status")
    ans = res.get("answer", "")
    conf = res.get("confidence")
    grounded = res.get("grounded")
    refusal_reason = res.get("refusal_reason")
    c_suff = res.get("context_sufficient")
    
    report = {
        "id": qid,
        "query": q,
        "status": status,
        "confidence": conf,
        "grounded": grounded,
        "refusal_reason": refusal_reason,
        "context_sufficient": c_suff,
        "answer_snippet": ans[:100]
    }
    
    if status == "refused":
        refused_queries.append(report)
    else:
        answered_queries.append(report)

print(f"\nRESULTS: {len(answered_queries)} ANSWERED, {len(refused_queries)} REFUSED out of {len(dataset)}")
if refused_queries:
    print("\nREFUSED QUERIES FROM KNOWLEDGE BASE:")
    for r in refused_queries:
        print(f"- [{r['id']}] '{r['query']}' -> reason: {r['refusal_reason']} (sufficient={r['context_sufficient']})")

with open("scratch/all_mock_queries_results.json", "w", encoding="utf-8") as f:
    json.dump({"answered": answered_queries, "refused": refused_queries}, f, indent=2, ensure_ascii=False)
