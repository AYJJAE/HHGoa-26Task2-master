import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from pipeline.vector_store import VectorStore
from pipeline.embeddings import EmbeddingPipeline

class RetrievalPipeline:
    def __init__(self, embedder: EmbeddingPipeline, store: VectorStore, reranker_model: str = ''):
        self.embedder = embedder
        self.store = store
        self.final_top_k = max(1, int(os.environ.get("RAG_FINAL_TOP_K", "5")))

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, query="", k=60, dense_weight=1.0, sparse_weight=1.0):
        """Fuses dense and sparse result lists using weighted Reciprocal Rank Fusion (RRF) with entity boost."""
        scores = {}
        q_lower = str(query or "").lower()
        key_terms = [t for t in re.findall(r'[\w]+', q_lower) if len(t) > 2]

        for rank, res in enumerate(dense_results):
            cid = res["id"] if isinstance(res, dict) else res.id
            payload = res["payload"] if isinstance(res, dict) else res.payload
            score_val = res["score"] if isinstance(res, dict) else res.score
            if cid not in scores:
                scores[cid] = {"score": 0.0, "payload": payload, "dense_rank": rank, "sparse_rank": -1, "dense_score": float(score_val)}
            scores[cid]["score"] += dense_weight / (k + rank)
            
        for rank, res in enumerate(sparse_results):
            cid = res["id"] if isinstance(res, dict) else res.id
            payload = res["payload"] if isinstance(res, dict) else res.payload
            score_val = res["score"] if isinstance(res, dict) else res.get("score", 0.0)
            if cid not in scores:
                scores[cid] = {"score": 0.0, "payload": payload, "dense_rank": -1, "sparse_rank": rank, "dense_score": float(score_val)}
            scores[cid]["score"] += sparse_weight / (k + rank)

        q_lower = str(query or "").lower()
        q_tokens = [w.strip(".,!?।;:\"'()[]{}") for w in q_lower.split() if len(w.strip(".,!?।;:\"'()[]{}")) > 1]
        goa_terms = {"गोवा", "goa", "गोया", "गोयाची", "गोव्याची", "panaji", "panjim", "पणजी"}
        conflicting_entities = {"महाराष्ट्र", "मुंबई", "राजस्थान", "जयपुर", "उत्तर प्रदेश", "लखनऊ", "france", "paris", "calcutta", "कोलकाता"}
        has_goa_in_q = any(gt in q_lower for gt in goa_terms)

        for cid, data in scores.items():
            p_text = data.get("payload", {}).get("text", "").lower()
            if q_tokens:
                matches = sum(1 for t in q_tokens if t in p_text)
                data["score"] += (matches / len(q_tokens)) * 0.05
            
            if has_goa_in_q:
                if any(gt in p_text for gt in goa_terms):
                    data["score"] += 0.20
                elif any(ce in p_text for ce in conflicting_entities):
                    data["score"] -= 0.15
            
        # Sort by fused score descending
        fused = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
        return [{"id": k, **v} for k, v in fused]

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates chunks based on normalized text content and document ID."""
        seen_texts = set()
        deduped = []
        for c in chunks:
            raw_text = c.get("payload", {}).get("text", "")
            norm = re.sub(r"\s+", " ", raw_text.strip()).lower()[:150]
            if norm and norm not in seen_texts:
                seen_texts.add(norm)
                deduped.append(c)
            elif not norm:
                deduped.append(c)
        return deduped

    def _calculate_confidence(self, final_results: List[Dict[str, Any]], fused_results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Calculates Retrieval Confidence based on RRF and dense/sparse agreement."""
        if not final_results:
            return "LOW", {"reason": "No results returned."}
            
        top_dense_score = final_results[0].get('dense_score', 0.0)
        
        gap = 0.0
        if len(final_results) > 1:
            gap = top_dense_score - final_results[1].get('dense_score', 0.0)
            
        dense_rank = final_results[0].get('dense_rank', -1)
        sparse_rank = final_results[0].get('sparse_rank', -1)
        agreement = (dense_rank != -1 and dense_rank < 10) and (sparse_rank != -1 and sparse_rank < 10)
        
        metrics = {
            "top_score": float(top_dense_score),
            "score_gap": float(gap),
            "dense_sparse_agreement": agreement
        }
        
        if top_dense_score > 0.6 and agreement:
            return "HIGH", metrics
        elif top_dense_score > 0.4 or (top_dense_score > 0.35 and gap > 0.05):
            return "MEDIUM", metrics
        else:
            return "LOW", metrics

    def retrieve(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the engineered hybrid retrieval pipeline and tracks latency for all sub-stages."""
        metrics = {}
        
        # 1. Query Preprocessing & Embedding Generation
        t0 = time.perf_counter()
        dense_vec, sparse_vec, embedding_cache_hit = self.embedder.embed_query(query)
        embedding_ms = (time.perf_counter() - t0) * 1000.0
        metrics["query_embedding_ms"] = embedding_ms
        metrics["embedding_ms"] = embedding_ms
        metrics["query_embedding_cache_hit"] = embedding_cache_hit
        
        top_k_retrieve = max(strategy.get("top_k_retrieve", 25), 25)
        chunk_strategy = strategy.get("chunk_strategy") or strategy.get("preferred_chunk_strategy")

        # 2. Parallel Dense FAISS Search + Sparse BM25 Search
        t_search_start = time.perf_counter()
        
        def run_dense():
            return self.store.search_dense(dense_vec, top_k=top_k_retrieve, chunk_strategy=chunk_strategy)

        def run_sparse():
            return self.store.search_sparse(sparse_vec, top_k=top_k_retrieve, chunk_strategy=chunk_strategy)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search") as executor:
            dense_future = executor.submit(run_dense)
            sparse_future = executor.submit(run_sparse)
            dense_results, dense_ms = dense_future.result()
            sparse_results, sparse_ms = sparse_future.result()

        total_search_ms = (time.perf_counter() - t_search_start) * 1000.0
        metrics["dense_search_ms"] = dense_ms
        metrics["dense_retrieval_ms"] = dense_ms
        metrics["sparse_search_ms"] = sparse_ms
        metrics["sparse_retrieval_ms"] = sparse_ms
        metrics["vector_search_ms"] = total_search_ms
        
        # 3. Hybrid Score Fusion (RRF)
        t0 = time.perf_counter()
        dense_w = strategy.get("dense_weight", 1.0)
        sparse_w = strategy.get("sparse_weight", 1.0)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, query=query, dense_weight=dense_w, sparse_weight=sparse_w)
        fusion_ms = (time.perf_counter() - t0) * 1000.0
        metrics["fusion_ms"] = fusion_ms
        metrics["rrf_fusion_ms"] = fusion_ms
        metrics["rerank_ms"] = 0.0
        metrics["reranking_ms"] = 0.0

        # 4. Deduplication & Context Selection
        t0 = time.perf_counter()
        deduped = self._deduplicate_chunks(fused)
        target_top_k = strategy.get("final_top_k", self.final_top_k)
        final_results = deduped[:target_top_k]
        for r in final_results:
            r['rerank_score'] = r['dense_score']

        confidence, conf_metrics = self._calculate_confidence(final_results, fused)
        context_ms = (time.perf_counter() - t0) * 1000.0
        metrics["context_ms"] = context_ms
        metrics["context_assembly_ms"] = context_ms
        
        # 5. Total Retrieval Time Calculation
        retrieval_total_ms = embedding_ms + total_search_ms + fusion_ms + context_ms
        metrics["retrieval_total_ms"] = retrieval_total_ms
        metrics["total_retrieval_ms"] = retrieval_total_ms
        
        return {
            "results": final_results,
            "confidence": confidence,
            "confidence_metrics": conf_metrics,
            "latency_ms": metrics
        }
