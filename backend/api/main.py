from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
import time
import asyncio
from typing import Optional, List, Dict, Any

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
    RATE_LIMITING_AVAILABLE = False
    limiter = None

try:
    from app.config import (
        HIGH_CONFIDENCE_THRESHOLD,
        MEDIUM_CONFIDENCE_THRESHOLD,
        VERIFIER_MIN_CONFIDENCE
    )
except ImportError:
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.50
    VERIFIER_MIN_CONFIDENCE = 0.55

from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from pipeline.query_router import QueryRouter
from pipeline.generation import GenerationPipeline
from pipeline.grounding import GroundingValidator
from pipeline.context_gate import is_context_sufficient
from .sarvam_client import TranscriptionResult
from .stt_service import STTService
from contextlib import asynccontextmanager

from qdrant_client.http import models as rest

import threading
import traceback

class RAGResources:
    def __init__(self):
        self.router = None
        self.generator = None
        self.grounder = None
        self.stt_client = None
        self.embedder = None
        self.store = None
        self.retriever = None
        self.ready = False
        self.error = None

    def initialize(self):
        try:
            print("[RAG] Initializing QueryRouter...")
            self.router = QueryRouter()
            
            print("[RAG] Initializing GenerationPipeline...")
            self.generator = GenerationPipeline(provider_name="gemini")
            
            print("[RAG] Initializing GroundingValidator...")
            self.grounder = GroundingValidator()
            
            print("[RAG] Initializing STTService...")
            self.stt_client = STTService()
            
            print("[RAG] Initializing EmbeddingPipeline...")
            self.embedder = EmbeddingPipeline()
            
            print("[RAG] Initializing VectorStore...")
            self.store = VectorStore(collection_name="msmarco_xi", dense_dim=self.embedder.dense_dim)
            
            print("[RAG] Initializing RetrievalPipeline...")
            self.retriever = RetrievalPipeline(self.embedder, self.store)
            
            print("Pipelines, Verifier, and STT Failover Service initialized successfully.")
            
            count = self.store.client.count(self.store.collection_name).count
            print(f"Collection '{self.store.collection_name}' has {count} vectors.")
            if count == 0:
                print("Index is empty. Ingesting dataset ONCE for production memory...")
                from pipeline.ingestion import ingest_dataset
                ingest_dataset(mode="mock", max_records=100, store=self.store)
                count = self.store.client.count(self.store.collection_name).count
                print(f"Ingestion complete. Collection now has {count} vectors.")
            
            self.ready = True
            print("[RAG] RAG initialization complete")
            
        except Exception as e:
            print(f"[RAG] Initialization failed: {e}")
            traceback.print_exc()
            self.error = str(e)
            
            # Attempt to set up fallback memory retriever if Qdrant disk failed
            if not self.retriever:
                print(f"Notice: Initializing fallback retrieval: {e}")
                try:
                    self.embedder = EmbeddingPipeline()
                    self.store = VectorStore(collection_name="msmarco_xi", dense_dim=384, in_memory=True)
                    self.retriever = RetrievalPipeline(self.embedder, self.store)
                    self.ready = True
                except Exception as e2:
                    print(f"Error initializing fallback retriever: {e2}")

resources = RAGResources()

REQUIRED_ORIGINS = [
    "https://task2-horizonlabs.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

env_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

ALLOWED_ORIGINS = list(dict.fromkeys(REQUIRED_ORIGINS + env_origins))

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("[STARTUP] HH Goa Voice RAG API starting")
    print(f"[STARTUP] Python version: {sys.version.split()[0]}")
    port = os.getenv("PORT", "8080")
    print(f"[STARTUP] PORT: {port}")
    print("[STARTUP] Binding: 0.0.0.0")

    print("[CORS] Production origins loaded:")
    for o in ALLOWED_ORIGINS:
        print(f"  - {o}")

    print("[ROUTES]")
    for route in app_instance.routes:
        methods = getattr(route, "methods", None)
        route_path = getattr(route, "path", None)
        if route_path:
            methods_str = ",".join(methods) if methods else "GET"
            print(f"{methods_str}  {route_path}")
            
    print("[STARTUP] FastAPI application loaded")
    print("[STARTUP] Initializing RAG resources synchronously before accepting traffic...")

    # Run initialize synchronously so container doesn't accept requests until ready
    await asyncio.to_thread(resources.initialize)
    print("[STARTUP] Initialization complete. Server is now ready!")
    
    yield

app = FastAPI(title="HH Goa 2026 Voice RAG API", lifespan=lifespan)

# --- Rate Limiting Registration ---
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

APP_BUILD_ID = "STT-FIX-2026-08-18-01"

@app.get("/")
@app.get("/health")
def health():
    """Fast, lightweight health check endpoint."""
    return {"status": "ok", "service": "HH Goa 2026 Voice RAG API", "build_id": APP_BUILD_ID}

@app.get("/health/ready")
def health_ready():
    """Detailed readiness check for external dependencies."""
    if not resources.ready:
        if resources.error:
            return JSONResponse(status_code=503, content={"status": "error", "message": resources.error})
        return JSONResponse(status_code=503, content={"status": "initializing"})

    gemini_status = "active" if (resources.generator and resources.generator.provider.model) else "extractive_fallback"
    el_configured = bool(getattr(resources.stt_client.primary, "api_key", False)) if resources.stt_client else False
    sv_configured = bool(getattr(resources.stt_client.fallback, "api_key", False)) if resources.stt_client else False
    
    if el_configured and sv_configured:
        stt_status = "ready"
    elif el_configured or sv_configured:
        stt_status = "degraded"
    else:
        stt_status = "unconfigured"

    return {
        "status": "ready",
        "services": {
            "gemini": gemini_status,
            "stt": {
                "primary": "ready" if el_configured else "not_configured",
                "fallback": "ready" if sv_configured else "not_configured",
                "status": stt_status,
            },
            "vector_store": "ready" if resources.store else "offline"
        }
    }

@app.get("/health/retrieval")
def health_retrieval():
    """Diagnostic endpoint for retrieval index status."""
    if not resources.ready:
        return JSONResponse(status_code=503, content={"status": "initializing"})
        
    try:
        count = resources.store.client.count(resources.store.collection_name).count if resources.store else 0
        embedding_model = resources.embedder.model_name if resources.embedder else "unknown"
        dense_dim = resources.embedder.dense_dim if resources.embedder else 0
        
        # safely access path if not in memory
        index_path = getattr(resources.store.client, "_client", None) if resources.store else None
        path_str = getattr(index_path, "_path", "in_memory") if index_path else "in_memory"
        
        return {
            "index_loaded": count > 0,
            "vector_count": count,
            "embedding_dimension": dense_dim,
            "embedding_model": embedding_model,
            "top_k": resources.retriever.final_top_k if resources.retriever else 0,
            "grounding_threshold": resources.grounder.min_confidence if resources.grounder else 0,
            "dataset_loaded": count > 0,
            "chunk_count": count,
            "index_path": str(path_str)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/retrieve")
async def retrieve_only(payload: RetrieveRequest):
    """Direct retrieval endpoint for low-latency benchmarking and indexing checks."""
    if not resources.ready:
        raise HTTPException(status_code=503, detail="Service is initializing")
        
    t0 = time.perf_counter()
    routing_info = resources.router.route_query(payload.query)
    strategy = {**routing_info.get("strategy", {}), "final_top_k": payload.top_k or 5}
    retrieval_res = await asyncio.to_thread(resources.retriever.retrieve, payload.query, strategy)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "query": payload.query,
        "results": retrieval_res["results"],
        "confidence": retrieval_res["confidence"],
        "latency_metrics": {**retrieval_res["latency_ms"], "total_ms": total_ms}
    }

@app.post("/api/ask")
@limiter.limit("60/minute") if RATE_LIMITING_AVAILABLE else lambda f: f
async def ask_question(payload: QueryRequest, request: Request):
    """Process question through the full RAG and Gemini Verification pipeline."""
    return await asyncio.to_thread(process_rag_pipeline, payload.query, False, None, payload.history)

@app.post("/api/transcribe")
@limiter.limit("40/minute") if RATE_LIMITING_AVAILABLE else lambda f: f
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Voice transcription endpoint with ElevenLabs primary and Sarvam fallback."""
    if not resources.ready:
        return {"success": False, "error": "Service is initializing. Please try again in a few seconds."}
        
    audio_bytes = await audio.read()
    
    transcription = await asyncio.to_thread(
        resources.stt_client.transcribe_audio, audio_bytes, audio.filename, audio.content_type, language
    )
    
    if isinstance(transcription, tuple):
        success, text = transcription
        transcription = TranscriptionResult(success=success, text=text if success else "", error=None if success else text)
    
    if not transcription.success:
        return {
            "success": False,
            "error": transcription.error or "Transcription failed.",
            "provider": getattr(transcription, "provider", None),
            "fallback_used": getattr(transcription, "fallback_used", False),
            "latency_ms": getattr(transcription, "latency_ms", 0.0)
        }
        
    return {
        "success": True,
        "text": transcription.text,
        "language": transcription.language,
        "provider": getattr(transcription, "provider", "elevenlabs"),
        "fallback_used": getattr(transcription, "fallback_used", False),
        "latency_ms": getattr(transcription, "latency_ms", 0.0)
    }

@app.post("/api/voice_ask")
@limiter.limit("30/minute") if RATE_LIMITING_AVAILABLE else lambda f: f
async def voice_ask(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    debug: bool = Form(False),
    history: Optional[str] = Form(None)
):
    """End-to-end voice question answering."""
    if not resources.ready:
        return {"status": "error", "message": "Service is initializing. Please try again in a few seconds."}
        
    audio_bytes = await audio.read()
    filename = audio.filename or "recording.webm"
    content_type = audio.content_type or "audio/webm"
    
    print(f"[STT-PROD] REQUEST_RECEIVED")
    print(f"[STT-PROD] AUDIO_SIZE={len(audio_bytes)}")
    print(f"[STT-PROD] AUDIO_CONTENT_TYPE={content_type}")
    
    el_key_present = bool(resources.stt_client.primary.api_key)
    sv_key_present = bool(resources.stt_client.fallback.api_key)
    print(f"[STT-PROD] ELEVENLABS_KEY_CONFIGURED={el_key_present}")
    print(f"[STT-PROD] SARVAM_KEY_CONFIGURED={sv_key_present}")
    print(f"[STT-PROD] ELEVENLABS_KEY_LENGTH={len(resources.stt_client.primary.api_key) if el_key_present else 0}")
    print(f"[STT-PROD] SARVAM_KEY_LENGTH={len(resources.stt_client.fallback.api_key) if sv_key_present else 0}")
    
    transcription = await asyncio.to_thread(
        resources.stt_client.transcribe_audio, audio_bytes, filename, content_type, language
    )
    
    if isinstance(transcription, tuple):
        success, text = transcription
        transcription = TranscriptionResult(success=success, text=text if success else "", error=None if success else text)
    
    if not transcription.success:
        print(f"[VOICE] STT failed | error='{transcription.error}' | provider={transcription.provider} | latency={transcription.latency_ms:.1f}ms")
        return {
            "status": "error",
            "message": "We couldn't transcribe that audio. Please try again.",
            "transcription": {
                "success": False,
                "error": transcription.error or "Speech transcription unavailable.",
                "provider": transcription.provider,
                "fallback_used": getattr(transcription, 'fallback_used', False)
            },
            "latency_metrics": {
                "stt_ms": transcription.latency_ms,
                "transcription_total_ms": transcription.latency_ms
            },
        }
        
    safe_snippet = (transcription.text or "")[:40].encode("ascii", "backslashreplace").decode("ascii")
    print(f"[VOICE] STT succeeded | provider={transcription.provider} | latency={transcription.latency_ms:.1f}ms | text='{safe_snippet}...'")
    
    parsed_history = None
    if history:
        try:
            parsed_history = json.loads(history)
        except Exception:
            parsed_history = None

    response = await asyncio.to_thread(process_rag_pipeline, transcription.text, debug, transcription.language, parsed_history)
    response["latency_metrics"]["stt_ms"] = transcription.latency_ms
    response["latency_metrics"]["transcription_total_ms"] = transcription.latency_ms
    response["latency_metrics"]["total_e2e_ms"] = response["latency_metrics"].get("total_e2e_ms", 0.0) + transcription.latency_ms
    response["transcription"] = {
        "success": True,
        "text": transcription.text,
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "provider": transcription.provider,
        "fallback_used": getattr(transcription, 'fallback_used', False),
        "latency_ms": transcription.latency_ms
    }
    return response

def calculate_final_confidence(
    question_relevant: bool,
    answers_question: bool,
    supported_by_context: bool,
    verifier_confidence: float,
    grounded: bool,
    relevance_score: float = 0.0,
    retrieval_score: float = 0.0
) -> str:
    """Calculates Final Answer Confidence using a calibrated multi-signal policy:
    - HIGH: Strong retrieval/relevance + fully grounded + high verifier confidence (>= 0.75)
    - MEDIUM: Relevant evidence + supported answer + moderate confidence (>= 0.50)
    - LOW: Supported but weak/partial evidence or ungrounded/refused
    """
    if not grounded or not question_relevant or not answers_question or not supported_by_context:
        return "LOW"
    
    if verifier_confidence >= HIGH_CONFIDENCE_THRESHOLD and (relevance_score >= 0.50 or retrieval_score >= 0.60):
        return "HIGH"
    elif verifier_confidence >= MEDIUM_CONFIDENCE_THRESHOLD or relevance_score >= 0.45:
        return "MEDIUM"
    else:
        return "LOW"

def classify_goa_category(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["itinerary", "plan", "day 1", "day 2", "day 3", "2 day", "3 day", "5 day", "trip", "tour", "योजना", "दौरा", "प्लॅन", "ट्रिप"]):
        return "Itinerary"
    if any(w in q for w in ["family", "families", "children", "kids", "parents", "elderly", "परिवार", "कुटुंब"]):
        return "Family"
    if any(w in q for w in ["beach", "beaches", "sea", "coast", "shore", "sand", "ocean", "बीच", "समुद्र तट", "दर्यावेळ", "किनारा"]):
        return "Beaches"
    if any(w in q for w in ["food", "dish", "eat", "curry", "fish curry", "poi", "bebinca", "vindaloo", "xacuti", "feni", "drink", "cuisine", "taste", "खाना", "व्यंजन", "जेवण", "खाद्य"]):
        return "Food"
    if any(w in q for w in ["church", "basilica", "cathedral", "heritage", "history", "portuguese", "monument", "fort", "aguada", "chapora", "cabo de rama", "old goa", "चर्च", "किला", "इतिहास", "वारसा", "किल्ले"]):
        return "Heritage"
    if any(w in q for w in ["carnival", "shigmo", "sao joao", "culture", "tradition", "festival", "dance", "mando", "folk", "संस्कृती", "त्योहार", "उत्सव", "परंपरा"]):
        return "Culture"
    if any(w in q for w in ["waterfall", "dudhsagar", "bird", "salim ali", "spice", "plantation", "wildlife", "sanctuary", "forest", "nature", "झरना", "पक्षी", "मसाला"]):
        return "Nature"
    if any(w in q for w in ["water sports", "scuba", "diving", "parasailing", "trek", "jeep safari", "adventure", "साहस", "सफारी"]):
        return "Adventure"
    if any(w in q for w in ["club", "party", "nightlife", "pub", "bar", "night market", "नाईट"]):
        return "Nightlife"
    if any(w in q for w in ["budget", "cheap", "affordable", "cost", "free", "बजट", "कमी खर्च"]):
        return "Budget"
    if any(w in q for w in ["transport", "scooter", "bike", "car", "taxi", "cab", "goamiles", "bus", "ferry", "train", "airport", "travel", "reach", "गाड़ी", "किराया", "वाहतूक"]):
        return "Transport"
    if any(w in q for w in ["goa", "panaji", "panjim", "margao", "गोवा", "गोव्यात", "गोय"]):
        return "General Goa"
    return "General"

def expand_query_with_history(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    if not history:
        return query
    q_lower = query.lower().strip()
    follow_up_cues = ["which one", "which of", "what about", "how about", "where is it", "how to reach", "how to go", "tell me more", "how much", "why is it", "is it", "are they", "there", "these", "those", "that", "it", "यापैकी", "त्यातले", "उनमें से", "वहाँ"]
    is_follow_up = any(cue in q_lower for cue in follow_up_cues) or (len(query.split()) <= 4 and not any(w in q_lower for w in ["capital", "who", "when", "what is", "where is", "largest"]))
    
    if is_follow_up:
        last_user_query = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user_query = turn.get("content", "")
                break
        if last_user_query:
            return f"{last_user_query} {query}"
    return query

def process_rag_pipeline(query: str, debug: bool = False, language_hint: str = None, history: Optional[List[Dict[str, str]]] = None):
    """Calibrated Three-Stage RAG Pipeline:
    STAGE 1: FAISS / Qdrant Hybrid Retrieval
    STAGE 2: Evidence Relevance Scoring & Soft Gating
    STAGE 3: Answer Generation & Grounding Verification
    """
    t_start = time.perf_counter()
    metrics = {}
    debug_info = {}

    # 1. Input Validation
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "error",
            "message": "Please provide a non-empty question.",
            "latency_metrics": {"total_e2e_ms": (time.perf_counter() - t_start) * 1000},
        }
    query = query.strip()
    category = classify_goa_category(query)
    retrieval_query = expand_query_with_history(query, history)

    # 2. Query Processing & Routing
    t0 = time.perf_counter()
    if not resources.ready:
        return {"status": "error", "message": "Service is still initializing, please try again."}
        
    routing_info = resources.router.route_query(query, language_hint=language_hint)
    metrics["query_routing_ms"] = (time.perf_counter() - t0) * 1000
    strategy = routing_info.get("strategy", {})

    # Chitchat Guardrail
    if routing_info["intent"] == "Chitchat":
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        res = {
            "status": "answered",
            "grounded": True,
            "refused": False,
            "confidence": "HIGH",
            "category": category,
            "answer": "Hello! I'm your multilingual voice RAG assistant. Ask me anything about Goa or the knowledge base! 🙏",
            "sources": [],
            "routing": routing_info,
            "context_sufficient": True,
            "verification": {
                "question_relevant": True,
                "answers_question": True,
                "supported_by_context": True,
                "supported": True,
                "verifier_confidence": 1.0,
                "confidence": 1.0,
                "reason": "Chitchat query.",
                "unsupported_claims": []
            },
            "retrieval": {"top_score": 1.0},
            "latency_metrics": metrics
        }
        if debug:
            res["debug"] = {"generation_executed": False, "chitchat": True}
        return res

    # 3. STAGE 1: Hybrid Retrieval
    safe_q = retrieval_query.encode("ascii", "backslashreplace").decode("ascii")
    print(f"[RETRIEVAL] query={safe_q}")
    print(f"[RETRIEVAL] embedding_dimension={resources.embedder.dense_dim if resources.embedder else 0}")
    print(f"[RETRIEVAL] top_k={strategy.get('final_top_k', resources.retriever.final_top_k if resources.retriever else 0)}")
    try:
        retrieval_res = resources.retriever.retrieve(retrieval_query, strategy)
        vec_count = resources.store.client.count(resources.store.collection_name).count if resources.store else 0
        print(f"[RETRIEVAL] index_vectors={vec_count}")
        print(f"[RETRIEVAL] scores={[r.get('dense_score', 0) for r in retrieval_res['results']]}")
        print(f"[RETRIEVAL] returned_chunks={len(retrieval_res['results'])}")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        return {
            "status": "error",
            "message": "The knowledge search service is temporarily unavailable. Please try again.",
            "latency_metrics": metrics,
        }
    retrieval_metrics = retrieval_res["latency_ms"]
    top_dense_score = retrieval_res["results"][0].get("dense_score", 0.0) if retrieval_res["results"] else 0.0

    # 4. STAGE 2: Relevance Scoring & Soft Gating
    t0 = time.perf_counter()
    context_val = is_context_sufficient(retrieval_query, retrieval_res["results"], resources.embedder)
    metrics["answerability_ms"] = (time.perf_counter() - t0) * 1000

    print(f"[GROUNDING] threshold={resources.grounder.min_confidence if resources.grounder else 0}")
    print(f"[GROUNDING] context_sufficient={context_val.sufficient}")

    if debug:
        debug_info["retrieved_chunks"] = [
            {
                "id": r.get("id"),
                "dense_score": r.get("dense_score", 0.0),
                "sparse_score": r.get("sparse_score", 0.0),
                "rrf_score": r.get("score", 0.0),
                "text": r.get("payload", {}).get("text", "")[:120]
            }
            for r in retrieval_res.get("results", [])
        ]
        debug_info["scores"] = [r.get("dense_score", 0.0) for r in retrieval_res.get("results", [])]
        debug_info["threshold"] = resources.grounder.min_confidence if resources.grounder else 0.55
        debug_info["context_sufficient"] = context_val.sufficient
        debug_info["context_reason"] = context_val.reason

    if not context_val.sufficient:
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        debug_info["generation_executed"] = False
        res = {
            "status": "refused",
            "grounded": False,
            "refused": True,
            "confidence": "LOW",
            "category": category,
            "answer": "I couldn't find enough relevant information in the retrieved knowledge base to answer that question accurately.",
            "sources": [],
            "routing": routing_info,
            "context_sufficient": False,
            "refusal_reason": "insufficient_context",
            "reason": context_val.reason,
            "verification": {
                "question_relevant": False,
                "answers_question": False,
                "supported_by_context": False,
                "supported": False,
                "verifier_confidence": 0.0,
                "confidence": 0.0,
                "reason": context_val.reason,
                "unsupported_claims": ["Context insufficient for query intent."]
            },
            "retrieval": {
                "top_score": round(float(top_dense_score), 3)
            },
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if debug:
            res["debug"] = debug_info
        return res

    # 5. Context Assembly
    t0 = time.perf_counter()
    context_text = "\n".join(r["payload"].get("text", "") for r in retrieval_res["results"])
    metrics["context_ms"] = (time.perf_counter() - t0) * 1000
    metrics["context_chars"] = len(context_text)
    metrics["context_estimated_tokens"] = max(1, len(context_text) // 4) if context_text else 0
    if debug:
        debug_info["final_context"] = context_text

    # 6. STAGE 3: Answer Generation
    t0 = time.perf_counter()
    candidate_answer = resources.generator.generate_answer(query, retrieval_res["results"], history=history)
    metrics["generation_ms"] = (time.perf_counter() - t0) * 1000
    metrics["output_estimated_tokens"] = max(1, len(candidate_answer) // 4) if candidate_answer else 0
    debug_info["generation_executed"] = True
    
    if "INSUFFICIENT_CONTEXT" in candidate_answer:
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        if debug:
            debug_info["grounding_score"] = 0.0
            debug_info["verification_result"] = {"status": "insufficient_context_from_generator"}
        res = {
            "status": "refused",
            "grounded": False,
            "refused": True,
            "confidence": "LOW",
            "category": category,
            "answer": "I couldn't find enough relevant information in the retrieved knowledge base to answer that question accurately.",
            "sources": [],
            "routing": routing_info,
            "context_sufficient": False,
            "refusal_reason": "insufficient_context_after_generation",
            "reason": "Generator detected insufficient context.",
            "verification": {
                "question_relevant": False,
                "answers_question": False,
                "supported_by_context": False,
                "supported": False,
                "verifier_confidence": 0.0,
                "confidence": 0.0,
                "reason": "Generator explicitly detected insufficient context.",
                "unsupported_claims": ["Missing factual basis in context."]
            },
            "retrieval": {
                "top_score": round(float(top_dense_score), 3)
            },
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if debug:
            res["debug"] = debug_info
        return res

    # 7. Grounding & Semantic Verification
    t0 = time.perf_counter()
    verification_res = resources.grounder.verify_answer(
        question=retrieval_query,
        candidate_answer=candidate_answer,
        retrieved_chunks=retrieval_res["results"],
        gemini_provider=resources.generator.provider
    )
    metrics["verification_ms"] = (time.perf_counter() - t0) * 1000

    if debug:
        debug_info["grounding_score"] = verification_res.confidence
        debug_info["verification_result"] = {
            "question_relevant": verification_res.question_relevant,
            "answers_question": verification_res.answers_question,
            "supported_by_context": verification_res.supported_by_context,
            "verifier_confidence": verification_res.confidence,
            "reason": verification_res.reason,
            "unsupported_claims": verification_res.unsupported_claims
        }

    # 8. Calibrated Acceptance Policy:
    is_valid = (
        verification_res.question_relevant
        and verification_res.answers_question
        and verification_res.supported_by_context
        and (verification_res.confidence >= resources.grounder.min_confidence or context_val.relevance_score >= 0.50)
    )

    final_conf = calculate_final_confidence(
        question_relevant=verification_res.question_relevant,
        answers_question=verification_res.answers_question,
        supported_by_context=verification_res.supported_by_context,
        verifier_confidence=verification_res.confidence,
        grounded=is_valid,
        relevance_score=context_val.relevance_score,
        retrieval_score=top_dense_score
    )

    if not is_valid:
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        res = {
            "status": "refused",
            "grounded": False,
            "refused": True,
            "confidence": "LOW",
            "category": category,
            "answer": "I couldn't find enough relevant information in the retrieved knowledge base to answer that question accurately.",
            "sources": [],
            "routing": routing_info,
            "context_sufficient": True,
            "refusal_reason": "grounding_verification_failed",
            "reason": verification_res.reason,
            "verification": {
                "question_relevant": verification_res.question_relevant,
                "answers_question": verification_res.answers_question,
                "supported_by_context": verification_res.supported_by_context,
                "supported": False,
                "verifier_confidence": verification_res.confidence,
                "confidence": verification_res.confidence,
                "reason": verification_res.reason,
                "unsupported_claims": verification_res.unsupported_claims
            },
            "retrieval": {
                "top_score": round(float(top_dense_score), 3)
            },
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        if debug:
            res["debug"] = debug_info
        return res

    # 9. Format Traceable Sources
    t0 = time.perf_counter()
    sources = []
    for r in retrieval_res["results"]:
        sources.append({
            "relevance": round(r.get("rerank_score", 0), 2),
            "language": r["payload"].get("language", "en"),
            "strategy": r["payload"].get("chunk_strategy", "unknown"),
            "document_id": r["payload"].get("document_id", ""),
            "text": r["payload"].get("text", "")
        })
    metrics["response_processing_ms"] = (time.perf_counter() - t0) * 1000
    metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000

    res = {
        "status": "answered",
        "grounded": True,
        "refused": False,
        "confidence": final_conf,
        "category": category,
        "answer": candidate_answer,
        "sources": sources,
        "routing": routing_info,
        "context_sufficient": True,
        "verification": {
            "question_relevant": verification_res.question_relevant,
            "answers_question": verification_res.answers_question,
            "supported_by_context": verification_res.supported_by_context,
            "supported": True,
            "verifier_confidence": verification_res.confidence,
            "confidence": verification_res.confidence,
            "reason": verification_res.reason,
            "unsupported_claims": []
        },
        "retrieval": {
            "top_score": round(float(top_dense_score), 3)
        },
        "latency_metrics": {**metrics, **retrieval_metrics}
    }
    if debug:
        res["debug"] = debug_info
    return res

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
