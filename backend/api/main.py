from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import time
import asyncio

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# --- Rate Limiting Setup ---
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    print("WARNING: slowapi not installed. Rate limiting disabled.")
    print("         Install with: pip install slowapi")
    RATE_LIMITING_AVAILABLE = False
    limiter = None

from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from pipeline.query_router import QueryRouter
from pipeline.generation import GenerationPipeline
from pipeline.grounding import GroundingValidator
from .sarvam_client import SarvamClient

app = FastAPI(title="HH Goa 2026 RAG API")

# --- Rate Limiting Registration ---
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Configuration (production-safe) ---
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed_origins_str == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Pipelines (This may take a moment to load models)...")
# Initialize singletons to keep models in memory
try:
    embedder = EmbeddingPipeline()
    store = VectorStore(collection_name="msmarco_xi", dense_dim=embedder.dense_dim)
    retriever = RetrievalPipeline(embedder, store)
    router = QueryRouter()
    generator = GenerationPipeline(provider_name="gemini") # Or dynamic based on env
    grounder = GroundingValidator()
    stt_client = SarvamClient()
    print("Pipelines initialized successfully.")
    
    # --- Gemini API Key Health Check ---
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key and generator.provider.model:
        print("[OK] GEMINI_API_KEY is set and Gemini model is loaded. Real generative answers enabled.")
    elif gemini_key and not generator.provider.model:
        print("[WARNING] GEMINI_API_KEY is set but Gemini model failed to initialize. Using extractive fallback.")
    else:
        print("[WARNING] GEMINI_API_KEY not set. Using extractive answer generation (returns top passage).")
        
except Exception as e:
    print(f"Error initializing pipelines: {e}")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    """Health check endpoint — also used by Docker healthcheck."""
    gemini_status = "active" if generator.provider.model else "extractive_fallback"
    sarvam_status = "active" if stt_client.api_key else "mock"
    return {
        "status": "ok",
        "message": "HH Goa 2026 API is running",
        "services": {
            "gemini": gemini_status,
            "sarvam_stt": sarvam_status,
        }
    }

@app.post("/api/ask")
@limiter.limit("30/minute") if RATE_LIMITING_AVAILABLE else lambda f: f
async def ask_question(payload: QueryRequest, request: Request):
    return process_rag_pipeline(payload.query)

@app.post("/api/voice_ask")
@limiter.limit("20/minute") if RATE_LIMITING_AVAILABLE else lambda f: f
async def ask_voice_question(request: Request, audio: UploadFile = File(...), debug: bool = False):
    # 1. Speech-to-Text
    t0 = time.perf_counter()
    audio_bytes = await audio.read()
    success, transcript = stt_client.transcribe_audio(audio_bytes, audio.filename)
    stt_latency = (time.perf_counter() - t0) * 1000
    
    if not success or not transcript.strip():
        return {
            "status": "error",
            "message": "We couldn't transcribe that audio. Please try again.",
            "latency": {"stt_ms": stt_latency}
        }
        
    # Process through standard RAG pipeline
    response = process_rag_pipeline(transcript)
    response["latency_metrics"]["stt_ms"] = stt_latency
    response["latency_metrics"]["total_e2e_ms"] = response["latency_metrics"]["total_e2e_ms"] + stt_latency
    return response

def process_rag_pipeline(query: str, debug: bool = False):
    t_start = time.perf_counter()
    metrics = {}

    # 1. Query Router
    t0 = time.perf_counter()
    routing_info = router.route_query(query)
    metrics["query_routing_ms"] = (time.perf_counter() - t0) * 1000

    # 1.5 Chitchat guard
    if routing_info["intent"] == "Chitchat":
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        response = {
            "status": "answered",
            "grounded": True,
            "answer": "Hello! I'm your multilingual RAG assistant. Ask me a question about the dataset and I'll find the answer for you! 🙏",
            "sources": [],
            "routing": routing_info,
            "retrieval_confidence": "N/A",
            "context_sufficient": True,
            "grounding": {"status": "SKIP", "reason": "Chitchat query — no retrieval needed."},
            "latency_metrics": metrics
        }
        if debug:
            response["debug"] = {"stage": "chitchat"}
        return response

    # 2. Hybrid Retrieval & Reranking
    retrieval_res = retriever.retrieve(query, routing_info["strategy"])
    retrieval_metrics = retrieval_res["latency_ms"]
    confidence = retrieval_res["confidence"]

    # 3. Context Sufficiency Check (Answerability)
    t0 = time.perf_counter()
    from pipeline.context_gate import is_context_sufficient
    context_val = is_context_sufficient(query, retrieval_res["results"], embedder)
    
    # Check debug flag
    rag_debug = os.environ.get("RAG_DEBUG", "false").lower() == "true"
    debug_payload = None

    if rag_debug:
        debug_payload = {
            "query": query,
            "detected_language": routing_info.get("language"),
            "intent": routing_info.get("intent"),
            "dense_top_score": retrieval_res["results"][0].get('dense_score') if retrieval_res["results"] else None,
            "sparse_top_score": retrieval_res["results"][0].get('sparse_score') if retrieval_res["results"] else None,
            "rrf_score": "high" if confidence == "HIGH" else "low",
            "supporting_chunks": context_val.supporting_chunks,
            "semantic_similarity": context_val.max_similarity,
            "keyword_overlap": context_val.keyword_overlap,
            "context_sufficient": context_val.sufficient,
            "generation_executed": False,
            "grounding_result": None,
            "final_decision": None
        }

    metrics["answerability_ms"] = (time.perf_counter() - t0) * 1000

    if not context_val.sufficient:
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        response = {
            "status": "refused",
            "grounded": False,
            "answer": "I couldn't find enough information in my knowledge base to answer that reliably.",
            "sources": [],
            "routing": routing_info,
            "retrieval_confidence": confidence,
            "context_sufficient": False,
            "refusal_reason": "insufficient_context",
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if rag_debug:
            debug_payload["final_decision"] = "REFUSED"
            response["debug"] = debug_payload
        return response

    # 4. LLM Generation
    t0 = time.perf_counter()
    answer = generator.generate_answer(query, retrieval_res["results"])
    metrics["generation_ms"] = (time.perf_counter() - t0) * 1000
    
    if rag_debug:
        debug_payload["generation_executed"] = True
        
    if "INSUFFICIENT_CONTEXT" in answer:
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        response = {
            "status": "refused",
            "grounded": False,
            "answer": "I found related information, but not enough reliable evidence to answer your question.",
            "sources": [],
            "routing": routing_info,
            "retrieval_confidence": confidence,
            "context_sufficient": False,
            "refusal_reason": "insufficient_context_after_generation",
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if rag_debug:
            debug_payload["final_decision"] = "REFUSED"
            response["debug"] = debug_payload
        return response

    # 5. Lightweight Grounding
    t0 = time.perf_counter()
    grounding_status, grounding_reason = grounder.validate(answer, retrieval_res["results"], llm_client=generator.provider)
    metrics["grounding_ms"] = (time.perf_counter() - t0) * 1000

    if rag_debug:
        debug_payload["grounding_result"] = grounding_status

    if grounding_status == "FAIL":
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        response = {
            "status": "refused",
            "grounded": False,
            "answer": "I found relevant information, but I couldn't verify that the generated answer is fully supported by it.",
            "sources": [],
            "routing": routing_info,
            "retrieval_confidence": confidence,
            "context_sufficient": True,
            "refusal_reason": "grounding_failed",
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if rag_debug:
            debug_payload["final_decision"] = "REFUSED"
            response["debug"] = debug_payload
        return response

    metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000

    # Format sources
    sources = []
    for r in retrieval_res["results"]:
        sources.append({
            "relevance": round(r.get("rerank_score", 0), 2),
            "language": r["payload"].get("language", "en"),
            "strategy": r["payload"].get("chunk_strategy", "unknown"),
            "document_id": r["payload"].get("document_id", ""),
            "text": r["payload"]["text"]
        })

    response = {
        "status": "answered",
        "grounded": True,
        "answer": answer,
        "sources": sources,
        "routing": routing_info,
        "retrieval_confidence": confidence,
        "context_sufficient": True,
        "latency_metrics": {**metrics, **retrieval_metrics}
    }
    if rag_debug:
        debug_payload["final_decision"] = "ANSWERED"
        response["debug"] = debug_payload
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
