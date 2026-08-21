import os
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SearchResponse:
    total_ms: float
    embed_ms: float
    search_ms: float
    results: List[Any] = field(default_factory=list)

_in_process_resources = None

def _get_in_process_retriever():
    global _in_process_resources
    if _in_process_resources is None:
        from api.main import resources
        if not resources.ready:
            resources.initialize()
        _in_process_resources = resources
    return _in_process_resources

def search(query: str, top_k: int = 5) -> SearchResponse:
    """Execute search against local server retrieval endpoint or in-process resident retriever."""
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/api/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=1
        )
        if resp.status_code == 200:
            data = resp.json()
            metrics = data.get("latency_metrics", {})
            embed_ms = float(metrics.get("query_embedding_ms", 0.0))
            search_ms = float(
                metrics.get("dense_retrieval_ms", 0.0) +
                metrics.get("sparse_retrieval_ms", 0.0) +
                metrics.get("rrf_fusion_ms", 0.0)
            )
            total_ms = embed_ms + search_ms
            if total_ms == 0.0:
                total_ms = (time.perf_counter() - t0) * 1000.0
                search_ms = total_ms - embed_ms
            return SearchResponse(
                total_ms=total_ms,
                embed_ms=embed_ms,
                search_ms=search_ms,
                results=data.get("results", [])
            )
    except Exception:
        pass
        
    # In-Process Fallback for Standalone Execution
    try:
        res_obj = _get_in_process_retriever()
        strategy = {"final_top_k": top_k, "top_k_retrieve": 25, "dense_weight": 0.5, "sparse_weight": 0.5}
        res = res_obj.retriever.retrieve(query, strategy)
        m = res.get("latency_ms", {})
        embed_ms = float(m.get("query_embedding_ms", 0.0))
        search_ms = float(m.get("dense_retrieval_ms", 0.0) + m.get("sparse_retrieval_ms", 0.0) + m.get("rrf_fusion_ms", 0.0))
        total_ms = float(m.get("retrieval_total_ms", embed_ms + search_ms))
        return SearchResponse(
            total_ms=total_ms,
            embed_ms=embed_ms,
            search_ms=search_ms,
            results=res.get("results", [])
        )
    except Exception as e:
        raise RuntimeError(f"Failed to execute search query '{query}': {e}")

def warmup():
    """Warmup the model and retriever with a test query."""
    try:
        search("Warmup test query", top_k=5)
    except Exception as e:
        print(f"Warmup warning: {e}")
