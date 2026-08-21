import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import resources, process_rag_pipeline
from pipeline.context_gate import is_context_sufficient


@pytest.fixture(scope="module", autouse=True)
def setup_pipeline():
    if not resources.ready:
        resources.initialize()


def test_regression_valid_factual_query_pm_india():
    """Valid query with direct knowledge base support must NOT be refused."""
    q = "Who is the Prime Minister of India?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "answered", f"Expected answered, got {res.get('status')}: {res.get('refusal_reason')}"
    assert res["refused"] is False
    assert res["grounded"] is True
    assert "Modi" in res["answer"] or "modi" in res["answer"].lower()
    assert res["confidence"] in ("HIGH", "MEDIUM")


def test_regression_valid_factual_query_us_flower():
    """Valid query with direct knowledge base support must NOT be refused."""
    q = "What is the national flower of the United States?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "answered", f"Expected answered, got {res.get('status')}: {res.get('refusal_reason')}"
    assert res["refused"] is False
    assert res["grounded"] is True
    assert "Rose" in res["answer"] or "rose" in res["answer"].lower()


def test_regression_valid_factual_query_mount_everest():
    """Valid query with Mount Everest elevation must NOT be refused."""
    q = "What is the height of Mount Everest?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "answered", f"Expected answered, got {res.get('status')}: {res.get('refusal_reason')}"
    assert res["refused"] is False
    assert res["grounded"] is True
    assert "8,849" in res["answer"] or "8849" in res["answer"] or "8848" in res["answer"]


def test_regression_valid_goa_beaches_query():
    """Valid Goa travel query must answer with beach recommendations."""
    q = "What are the main beaches in North Goa?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "answered", f"Expected answered, got {res.get('status')}: {res.get('refusal_reason')}"
    assert res["refused"] is False
    assert any(b in res["answer"].lower() for b in ["baga", "calangute", "anjuna", "vagator"])


def test_regression_unsupported_out_of_domain_refused():
    """Query with NO supporting evidence in dataset MUST be cleanly refused."""
    q = "What is the secret recipe for Martian chocolate lava cake?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "refused"
    assert res["refused"] is True
    assert res["confidence"] in ("LOW", "REFUSED")
    assert "couldn't find enough relevant information" in res["answer"]


def test_regression_conflicting_entity_attribute_refused():
    """Query asking for President of France when context discusses India/Goa must be refused."""
    q = "Who is the President of France?"
    res = process_rag_pipeline(q, debug=True)
    assert res["status"] == "refused"
    assert res["refused"] is True
    assert res["grounded"] is False


def test_regression_debug_payload_exposed_when_requested():
    """Grounding debug mode must expose scores, final_context, threshold, and verification."""
    q = "Who is the Prime Minister of India?"
    res = process_rag_pipeline(q, debug=True)
    assert "debug" in res
    dbg = res["debug"]
    assert "retrieved_chunks" in dbg
    assert "scores" in dbg
    assert "threshold" in dbg
    assert "context_sufficient" in dbg
    assert "final_context" in dbg
