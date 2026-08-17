import os
import sys
import time
import requests
from dataclasses import dataclass

@dataclass
class SearchResponse:
    total_ms: float
    embed_ms: float
    search_ms: float
    results: list = None

def search(query: str, top_k: int = 5) -> SearchResponse:
    """Execute search against local server retrieval endpoint."""
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/api/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=10
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
        
    try:
        resp = requests.post(
            f"{api_url}/api/ask",
            json={"query": query},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        metrics = data.get("latency_metrics", {})
        embed_ms = float(metrics.get("query_embedding_ms", 0.0))
        search_ms = float(
            metrics.get("dense_retrieval_ms", 0.0) +
            metrics.get("sparse_retrieval_ms", 0.0) +
            metrics.get("rrf_fusion_ms", 0.0)
        )
        total_ms = embed_ms + search_ms
        return SearchResponse(
            total_ms=total_ms,
            embed_ms=embed_ms,
            search_ms=search_ms,
            results=data.get("sources", [])
        )
    except Exception as e:
        raise RuntimeError(f"Failed to execute search query '{query}': {e}")

def warmup():
    """Warmup the model and retriever with a test query."""
    try:
        search("Warmup test query", top_k=5)
    except Exception as e:
        print(f"Warmup warning: {e}")
