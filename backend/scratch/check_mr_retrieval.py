import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources

resources.initialize()

query = "गोवा राज्याची राजधानी कोणती?"
dense_vec, sparse_vec, _ = resources.embedder.embed_query(query)
print("DENSE VEC len:", len(dense_vec))
print("SPARSE VEC:", sparse_vec)

# Check all chunks containing Panaji or राजधानी
for idx, chunk in enumerate(resources.store.faiss_store.chunks):
    text = chunk["text"]
    if "पणजी" in text or "Panaji" in text or "राजधानी" in text:
        # compute cosine similarity
        import numpy as np
        doc_vec = np.array(resources.store.faiss_store.index.reconstruct(idx))
        q_vec = np.array(dense_vec) / np.linalg.norm(dense_vec)
        cos_sim = float(np.dot(doc_vec, q_vec))
        bm_score = resources.store.faiss_store.bm25.score(sparse_vec, idx)
        print(f"\n[IDX {idx}] strat={chunk.get('chunk_strategy')} score_dense={cos_sim:.3f} bm={bm_score:.3f}")
        print("TEXT:", text[:120])
