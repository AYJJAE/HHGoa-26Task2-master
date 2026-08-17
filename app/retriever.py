import os
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if os.path.exists(backend_path) and backend_path not in sys.path:
    sys.path.insert(0, backend_path)

_session = requests.Session()

@dataclass
class RetrievalResponse:
    query: str
    results: list = field(default_factory=list)
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0

# Backward compatibility alias
SearchResponse = RetrievalResponse

_in_process_embedder = None
_in_process_store = None
_in_process_retriever = None
_in_process_router = None

def _get_in_process_retriever():
    global _in_process_embedder, _in_process_store, _in_process_retriever, _in_process_router
    if _in_process_retriever is None:
        try:
            import importlib
            retrieval_mod = importlib.import_module("pipeline.retrieval")
            embeddings_mod = importlib.import_module("pipeline.embeddings")
            vector_store_mod = importlib.import_module("pipeline.vector_store")
            query_router_mod = importlib.import_module("pipeline.query_router")

            RetrievalPipeline = retrieval_mod.RetrievalPipeline
            EmbeddingPipeline = embeddings_mod.EmbeddingPipeline
            VectorStore = vector_store_mod.VectorStore
            QueryRouter = query_router_mod.QueryRouter
            
            _in_process_embedder = EmbeddingPipeline()
            try:
                _in_process_store = VectorStore(collection_name="msmarco_xi", dense_dim=_in_process_embedder.dense_dim)
            except Exception:
                _in_process_store = VectorStore(collection_name="msmarco_xi", dense_dim=384, in_memory=True)
            _in_process_retriever = RetrievalPipeline(_in_process_embedder, _in_process_store)
            _in_process_router = QueryRouter()
        except Exception as e:
            print(f"Direct retriever init note: {e}")
    return _in_process_retriever, _in_process_router

_server_checked = False
_server_available = False

def search(query: str, top_k: int = 5) -> RetrievalResponse:
    """Execute search against local server retrieval endpoint or in-process pipeline."""
    global _server_checked, _server_available
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    t0 = time.perf_counter()

    if not _server_checked:
        try:
            check_resp = _session.get(f"{api_url}/health", timeout=0.3)
            _server_available = (check_resp.status_code == 200)
        except Exception:
            _server_available = False
        _server_checked = True

    if _server_available:
        try:
            resp = _session.post(
                f"{api_url}/api/retrieve",
                json={"query": query, "top_k": top_k},
                timeout=2
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
                    search_ms = max(0.01, total_ms - embed_ms)
                return RetrievalResponse(
                    query=query,
                    results=data.get("results", []),
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    total_ms=total_ms
                )
        except Exception:
            _server_available = False

    # In-process execution fallback
    retriever_inst, router_inst = _get_in_process_retriever()
    if retriever_inst is not None and router_inst is not None:
        routing_info = router_inst.route_query(query)
        strategy = {**routing_info.get("strategy", {}), "final_top_k": top_k}
        retrieval_res = retriever_inst.retrieve(query, strategy)
        metrics = retrieval_res.get("latency_ms", {})
        embed_ms = float(metrics.get("query_embedding_ms", 0.0))
        search_ms = float(
            metrics.get("dense_retrieval_ms", 0.0) +
            metrics.get("sparse_retrieval_ms", 0.0) +
            metrics.get("rrf_fusion_ms", 0.0)
        )
        total_ms = embed_ms + search_ms
        if total_ms == 0.0:
            total_ms = (time.perf_counter() - t0) * 1000.0
            search_ms = max(0.01, total_ms - embed_ms)
        return RetrievalResponse(
            query=query,
            results=retrieval_res.get("results", []),
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms
        )

    raise RuntimeError(f"Failed to execute search query '{query}' via both API and in-process pipeline.")

def warmup():
    """Warmup the model and retriever with a test query."""
    try:
        search("What is retrieval augmented generation?", top_k=5)
    except Exception as e:
        print(f"Warmup warning: {e}")
