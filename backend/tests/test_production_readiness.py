import pytest
from fastapi.testclient import TestClient
from api.main import app, resources

@pytest.fixture(scope="module")
def client():
    # Force initialization if not ready (usually handled by lifespan, but we can call it explicitly for tests)
    if not resources.ready:
        resources.initialize()
    with TestClient(app) as c:
        yield c

def test_production_knowledge_base_not_empty(client):
    """Ensure the KB is actively loaded and not an empty shell."""
    assert resources.ready is True, "RAGResources failed to initialize"
    
    # Assert faiss vector and chunk counts
    chunks_loaded = len(resources.store.faiss_store.chunks)
    assert chunks_loaded > 0, "No chunks loaded in vector store!"
    
    if getattr(resources.store.faiss_store, "index", None):
        faiss_vectors = resources.store.faiss_store.index.ntotal
        assert faiss_vectors > 0, "FAISS index ntotal is 0!"
        
    sparse_docs = getattr(resources.store.faiss_store, "bm25", None)
    if sparse_docs:
        assert sparse_docs.corpus_size > 0, "Sparse BM25 document count is 0!"

def test_known_query_returns_chunks(client):
    """Ensure /api/retrieve correctly fetches chunks."""
    response = client.post("/api/retrieve", json={"query": "What is the capital of India?", "top_k": 5})
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0, "Retrieval returned 0 chunks for a basic query."

def test_known_query_can_answer(client):
    """Ensure /api/ask properly responds and does not refuse due to empty KB."""
    response = client.post("/api/ask", json={"query": "What is the capital of India?"})
    assert response.status_code == 200
    
    data = response.json()
    assert data.get("refused") is False, f"Request was refused: {data.get('refusal_reason')}"
    assert "answer" in data
    assert len(data["answer"]) > 10, "Answer was surprisingly short."
