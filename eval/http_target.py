"""Official HTTP Target Adapter for HH Goa Task 2 Evaluation Loop.

Connects the evaluator to the live FastAPI backend over HTTP:
- POST /api/ask: Full RAG pipeline (Retrieval -> Context Gating -> Generation -> Grounding)
- POST /api/retrieve: Direct hybrid retrieval (Dense + Sparse + RRF fusion)
"""

import os
import json
import time
import requests
from typing import Dict, Any, List, Optional


class RAGHTTPTarget:
    """Evaluator client adapter for the HH Goa Task 2 Voice RAG HTTP service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("RAG_API_BASE_URL") or "http://127.0.0.1:8080").rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "rag-local-eval-loop/1.0"
        })

    def check_health(self) -> Dict[str, Any]:
        """Verify service is running and ready."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_ready(self) -> Dict[str, Any]:
        """Verify vector index and knowledge base are loaded."""
        try:
            resp = self.session.get(f"{self.base_url}/health/ready", timeout=self.timeout)
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Call POST /api/retrieve for isolated retrieval evaluation.
        
        Returns:
            {
                "query": query,
                "results": [{"id": ..., "text": ..., "score": ..., "metadata": ...}, ...],
                "confidence": float,
                "latency_metrics": {...}
            }
        """
        payload = {"query": query, "top_k": top_k}
        t0 = time.perf_counter()
        try:
            resp = self.session.post(f"{self.base_url}/api/retrieve", json=payload, timeout=self.timeout)
            client_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            data = resp.json()
            if "latency_metrics" not in data:
                data["latency_metrics"] = {}
            data["latency_metrics"]["http_client_rtt_ms"] = round(client_ms, 2)
            return data
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "confidence": 0.0,
                "error": str(e),
                "latency_metrics": {"http_client_rtt_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
            }

    def ask(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Call POST /api/ask for full end-to-end RAG answer generation & grounding.
        
        Returns:
            {
                "status": "answered" | "refused",
                "grounded": bool,
                "refused": bool,
                "confidence": "HIGH" | "MEDIUM" | "LOW",
                "answer": str,
                "sources": [{"text": ..., "document_id": ..., "relevance": ...}, ...],
                "latency_metrics": {
                    "embedding_ms": ...,
                    "dense_search_ms": ...,
                    "sparse_search_ms": ...,
                    "fusion_ms": ...,
                    "generation_ms": ...,
                    "total_e2e_ms": ...
                }
            }
        """
        payload = {"query": query}
        if history:
            payload["history"] = history
            
        t0 = time.perf_counter()
        try:
            resp = self.session.post(f"{self.base_url}/api/ask", json=payload, timeout=self.timeout)
            client_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            data = resp.json()
            if "latency_metrics" not in data:
                data["latency_metrics"] = {}
            data["latency_metrics"]["http_client_rtt_ms"] = round(client_ms, 2)
            return data
        except Exception as e:
            return {
                "status": "error",
                "grounded": False,
                "refused": True,
                "confidence": "LOW",
                "answer": f"Service error: {e}",
                "sources": [],
                "error": str(e),
                "latency_metrics": {"http_client_rtt_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
            }


# Standalone runner for testing connectivity
if __name__ == "__main__":
    target = RAGHTTPTarget()
    print("Testing /health...")
    print(json.dumps(target.check_health(), indent=2))
    print("\nTesting /health/ready...")
    print(json.dumps(target.check_ready(), indent=2))
    print("\nTesting /api/retrieve...")
    print(json.dumps(target.retrieve("What is the capital of India?", top_k=3), indent=2, ensure_ascii=False))
    print("\nTesting /api/ask...")
    print(json.dumps(target.ask("What is the capital of India?"), indent=2, ensure_ascii=False))
