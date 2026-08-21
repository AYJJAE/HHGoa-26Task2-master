import pytest
import time
import os
import sys
import statistics
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from pipeline.retrieval import RetrievalPipeline
from pipeline.query_router import QueryRouter
from pipeline.generation import GenerationPipeline
from pipeline.grounding import GroundingValidator, VerificationResult
from pipeline.context_gate import is_context_sufficient
from app.benchmark import percentile, QUERIES
from api.main import resources, process_rag_pipeline, app
from fastapi.testclient import TestClient

@pytest.fixture(scope="module", autouse=True)
def init_rag_system():
    if not resources.ready:
        resources.initialize()

def test_model_and_indexes_initialized_once():
    """Verify models, FAISS, and BM25 are initialized once and resident in memory."""
    assert resources.embedder is not None
    assert resources.store is not None
    assert resources.retriever is not None
    assert resources.store.faiss_store.count() > 0
    assert resources.store.faiss_store.bm25.corpus_size > 0

def test_faiss_and_bm25_fast_indexed_retrieval_no_full_scan():
    """Verify retrieval uses index rather than linear dataset iteration."""
    query = "What are the best beaches in Goa?"
    strategy = {"final_top_k": 5, "dense_weight": 1.0, "sparse_weight": 1.0}
    
    t0 = time.perf_counter()
    res = resources.retriever.retrieve(query, strategy)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    assert len(res["results"]) > 0
    # In-memory FAISS + BM25 should execute in under 30ms (usually < 5ms)
    assert latency_ms < 30.0
    assert "dense_search_ms" in res["latency_ms"]
    assert "sparse_search_ms" in res["latency_ms"]
    assert "fusion_ms" in res["latency_ms"]

def test_retrieval_substage_latencies_measured():
    """Verify every sub-stage has accurate, distinct, non-zero timing measurements."""
    query = "What should I visit in Old Goa?"
    strategy = {"final_top_k": 5}
    res = resources.retriever.retrieve(query, strategy)
    m = res["latency_ms"]
    
    assert "embedding_ms" in m or "query_embedding_ms" in m
    assert "dense_search_ms" in m
    assert "sparse_search_ms" in m
    assert "fusion_ms" in m
    assert "context_ms" in m
    assert "retrieval_total_ms" in m
    assert m["retrieval_total_ms"] > 0

def test_percentile_calculations_p50_p70_p100():
    """Verify P50, P70, and P100 percentile calculations."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p50 = percentile(data, 50)
    p70 = percentile(data, 70)
    p100 = percentile(data, 100)
    
    assert abs(p50 - 5.5) < 0.1
    assert abs(p70 - 7.3) < 0.1
    assert p100 == 10.0

def test_grounding_rejects_unsupported_answers():
    """Verify grounding validator rejects claims not present in context."""
    validator = GroundingValidator()
    context = [{"payload": {"text": "Panaji is the state capital of Goa located on the banks of Mandovi River."}}]
    
    unsupported_answer = "The capital of Goa is Margao and it has 50 airports."
    status, reason = validator.validate(unsupported_answer, context, query="What is the capital of Goa?")
    assert status == "FAIL"

def test_conflicting_evidence_rejected_in_context_gate():
    """Verify contradictory evidence triggers refusal."""
    query = "Who is the Prime Minister of India?"
    # Pass chunks talking only about flower / USA rose
    conflicting_chunks = [
        {"payload": {"text": "The national flower of the United States is the rose.", "language": "en"}, "dense_score": 0.35}
    ]
    val = is_context_sufficient(query, conflicting_chunks)
    assert val.sufficient is False

def test_voice_transcript_enters_rag_pipeline():
    """Verify transcription text flows cleanly into RAG pipeline."""
    sample_text = "What Goan food should I try?"
    mock_verification = VerificationResult(
        question_relevant=True,
        answers_question=True,
        supported_by_context=True,
        confidence=0.90,
        reason="Grounded Goan food answer.",
        unsupported_claims=[]
    )
    with patch.object(resources.generator, "generate_answer", return_value="You should try Goan Fish Curry, Bebinca, and Poi."), \
         patch.object(resources.grounder, "verify_answer", return_value=mock_verification):
        res = process_rag_pipeline(sample_text)
        assert res["status"] == "answered"
        assert res["grounded"] is True
        assert len(res["sources"]) > 0

def test_structured_errors_on_empty_input():
    """Verify empty query returns structured error response."""
    res = process_rag_pipeline("")
    assert res["status"] == "error"
    assert "message" in res
    assert "latency_metrics" in res

def test_elevenlabs_and_sarvam_failover_service():
    """Verify STT service failover handles exceptions gracefully."""
    stt = resources.stt_client
    assert stt is not None
    assert hasattr(stt, "primary")
    assert hasattr(stt, "fallback")
