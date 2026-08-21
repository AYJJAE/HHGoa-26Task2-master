import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources

resources.initialize()

q = "गोवा राज्याची राजधानी कोणती?"
res = resources.retriever.retrieve(q, {"dense_weight": 0.5, "sparse_weight": 0.5, "top_k_retrieve": 30, "final_top_k": 5})

with open("scratch/retrieval_out.txt", "w", encoding="utf-8") as f:
    f.write(f"QUERY: {q}\n")
    for idx, r in enumerate(res["results"]):
        f.write(f"\n--- RANK {idx+1} (RRF: {r.get('score', 0):.4f}, Dense: {r.get('dense_score', 0):.4f}) ---\n")
        f.write(f"ID: {r.get('id')}\n")
        f.write(f"TEXT: {r.get('payload', {}).get('text')}\n")

print("WROTE OUTPUT TO scratch/retrieval_out.txt")
