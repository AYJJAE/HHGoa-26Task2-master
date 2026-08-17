import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from pipeline.vector_store import VectorStore
from pipeline.embeddings import EmbeddingPipeline
from qdrant_client.http import models as rest

class RetrievalPipeline:
    def __init__(self, embedder: EmbeddingPipeline, store: VectorStore, reranker_model: str = ''):
        self.embedder = embedder
        self.store = store
        self.final_top_k = max(1, int(os.environ.get("RAG_FINAL_TOP_K", "5")))

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60, dense_weight=1.0, sparse_weight=1.0):
        """
        Fuses lists of results using Reciprocal Rank Fusion with optional weights.
        """
        scores = {}
        for rank, res in enumerate(dense_results):
            if res.id not in scores:
                scores[res.id] = {"score": 0.0, "payload": res.payload, "dense_rank": rank, "sparse_rank": -1, "dense_score": float(res.score)}
            scores[res.id]["score"] += dense_weight / (k + rank)
            
        for rank, res in enumerate(sparse_results):
            if res.id not in scores:
                scores[res.id] = {"score": 0.0, "payload": res.payload, "dense_rank": -1, "sparse_rank": rank, "dense_score": 0.0}
            scores[res.id]["score"] += sparse_weight / (k + rank)
            
        # Sort by fused score descending
        fused = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
        return [{"id": k, **v} for k, v in fused]
        
    def _calculate_confidence(self, final_results: List[Dict[str, Any]], fused_results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Calculates Retrieval Confidence based on RRF and dense/sparse agreement.
        Returns label (HIGH, MEDIUM, LOW) and the reasoning.
        """
        if not final_results:
            return "LOW", {"reason": "No results returned."}
            
        top_dense_score = final_results[0].get('dense_score', 0.0)
        
        # Calculate gap between top 1 and top 2 if available (dense score)
        gap = 0.0
        if len(final_results) > 1:
            gap = top_dense_score - final_results[1].get('dense_score', 0.0)
            
        # Agreement check: Did the top result appear highly in both dense and sparse?
        dense_rank = final_results[0]['dense_rank']
        sparse_rank = final_results[0]['sparse_rank']
        
        agreement = (dense_rank != -1 and dense_rank < 10) and (sparse_rank != -1 and sparse_rank < 10)
        
        metrics = {
            "top_score": float(top_dense_score),
            "score_gap": float(gap),
            "dense_sparse_agreement": agreement
        }
        
        # Cosine similarity thresholds for BGE-M3
        if top_dense_score > 0.6 and agreement:
            return "HIGH", metrics
        elif top_dense_score > 0.4 or (top_dense_score > 0.35 and gap > 0.05):
            return "MEDIUM", metrics
        else:
            return "LOW", metrics

    def retrieve(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full retrieval pipeline and tracks latency for each stage.
        """
        metrics = {}
        
        # 1. Query Embedding
        t0 = time.perf_counter()
        dense_vec, sparse_vec, embedding_cache_hit = self.embedder.embed_query(query)
        metrics["query_embedding_ms"] = (time.perf_counter() - t0) * 1000
        metrics["query_embedding_cache_hit"] = embedding_cache_hit
        
        top_k_retrieve = strategy.get("top_k_retrieve", 10)
        chunk_strategy = strategy.get("chunk_strategy") or strategy.get("preferred_chunk_strategy")
        query_filter = None
        if chunk_strategy:
            query_filter = rest.Filter(must=[rest.FieldCondition(
                key="chunk_strategy", match=rest.MatchValue(value=chunk_strategy)
            )])
        
        sparse_indices = list(sparse_vec.keys())
        sparse_values = list(sparse_vec.values())
        # These searches are independent. This preserves RRF results while reducing
        # wall time when Qdrant is remote or the local index releases the GIL.
        def dense_search():
            started = time.perf_counter()
            points = self.store.client.query_points(self.store.collection_name, dense_vec, using="dense", limit=top_k_retrieve, query_filter=query_filter).points
            return points, (time.perf_counter() - started) * 1000

        def sparse_search():
            started = time.perf_counter()
            points = self.store.client.query_points(self.store.collection_name, rest.SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=top_k_retrieve, query_filter=query_filter).points
            return points, (time.perf_counter() - started) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search") as executor:
            dense_future = executor.submit(dense_search)
            sparse_future = executor.submit(sparse_search)
            dense_results, metrics["dense_retrieval_ms"] = dense_future.result()
            sparse_results, metrics["sparse_retrieval_ms"] = sparse_future.result()
        metrics["vector_search_ms"] = (time.perf_counter() - t0) * 1000
        
        # 4. Score Fusion (RRF)
        t0 = time.perf_counter()
        dense_w = strategy.get("dense_weight", 1.0)
        sparse_w = strategy.get("sparse_weight", 1.0)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, dense_weight=dense_w, sparse_weight=sparse_w)
        final_results = fused[:strategy.get("final_top_k", self.final_top_k)]
        for r in final_results:
            r['rerank_score'] = r['dense_score'] # for compatibility with frontend/downstream
        metrics["rrf_fusion_ms"] = (time.perf_counter() - t0) * 1000
        metrics["reranking_ms"] = 0.0  # no cross-encoder is configured in the serving path
        
        # 6. Context Assembly & Confidence
        t0 = time.perf_counter()
        confidence, conf_metrics = self._calculate_confidence(final_results, fused)
        metrics["context_assembly_ms"] = (time.perf_counter() - t0) * 1000
        
        # Calculate total retrieval pipeline latency
        metrics["total_retrieval_ms"] = (
            metrics["query_embedding_ms"] + metrics["vector_search_ms"] +
            metrics["rrf_fusion_ms"] + metrics["context_assembly_ms"]
        )
        
        return {
            "results": final_results,
            "confidence": confidence,
            "confidence_metrics": conf_metrics,
            "latency_ms": metrics
        }
