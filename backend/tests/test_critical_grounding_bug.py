import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.grounding import GroundingValidator
from pipeline.context_gate import is_context_sufficient
from api.main import calculate_final_confidence

def make_chunk(text: str, score: float = 0.63):
    return {
        "dense_score": score,
        "payload": {
            "text": text,
            "document_id": "test_doc_1",
            "language": "en"
        }
    }


# ==============================================================================
# SECTION 21: MANDATORY FINAL ACCEPTANCE TESTS (1 - 5)
# ==============================================================================

def test_final_acceptance_1_national_flower_of_usa():
    """TEST 1:
    Question: 'What is the national flower of the United States?'
    Context: 'The Rose is the national flower of the United States and England.'
    Expected: ANSWER, Confidence: HIGH or MEDIUM, grounded: true, refused: false
    """
    query = "What is the national flower of the United States?"
    context = "The Rose is the national flower of the United States and England."
    candidate = "The Rose is the national flower of the United States."
    chunks = [make_chunk(context, score=0.88)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.question_relevant is True
    assert result.answers_question is True
    assert result.supported_by_context is True
    assert result.supported is True

    conf = calculate_final_confidence(
        result.question_relevant,
        result.answers_question,
        result.supported_by_context,
        result.confidence,
        result.supported,
        val.relevance_score,
        0.88
    )
    assert conf in ("HIGH", "MEDIUM")


def test_final_acceptance_2_president_of_usa_with_rose_refused():
    """TEST 2:
    Question: 'President Of the USA'
    Context: 'The Rose is the national flower of the United States and England.'
    Expected: NO ANSWER, Confidence: LOW, grounded: false, refused: true
    """
    query = "President Of the USA"
    context = "The Rose is the national flower of the United States and England."
    candidate = "The Rose is the national flower of the United States and England."
    chunks = [make_chunk(context, score=0.63)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False, "Context gate must reject when context discusses national flower instead of President."

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.question_relevant is False or result.answers_question is False
    assert result.supported is False

    conf = calculate_final_confidence(
        result.question_relevant,
        result.answers_question,
        result.supported_by_context,
        result.confidence,
        result.supported
    )
    assert conf == "LOW"


def test_final_acceptance_3_prime_minister_of_india():
    """TEST 3:
    Question: 'Who is the Prime Minister of India?'
    Context: 'Narendra Modi is the Prime Minister of India.'
    Expected: ANSWER, Confidence: HIGH or MEDIUM, grounded: true, refused: false
    """
    query = "Who is the Prime Minister of India?"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Narendra Modi is the Prime Minister of India."
    chunks = [make_chunk(context, score=0.92)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.question_relevant is True
    assert result.answers_question is True
    assert result.supported_by_context is True
    assert result.supported is True

    conf = calculate_final_confidence(
        result.question_relevant,
        result.answers_question,
        result.supported_by_context,
        result.confidence,
        result.supported,
        val.relevance_score,
        0.92
    )
    assert conf in ("HIGH", "MEDIUM")


def test_final_acceptance_4_who_leads_india():
    """TEST 4:
    Question: 'Who leads India?'
    Context: 'Narendra Modi is the Prime Minister of India.'
    Expected: ANSWER, Confidence: HIGH or MEDIUM, grounded: true, refused: false
    """
    query = "Who leads India?"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Narendra Modi is the Prime Minister of India."
    chunks = [make_chunk(context, score=0.82)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True, "Semantic query 'Who leads India?' must be recognized as sufficient with PM context."

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.question_relevant is True
    assert result.answers_question is True
    assert result.supported_by_context is True
    assert result.supported is True

    conf = calculate_final_confidence(
        result.question_relevant,
        result.answers_question,
        result.supported_by_context,
        result.confidence,
        result.supported,
        val.relevance_score,
        0.82
    )
    assert conf in ("HIGH", "MEDIUM")


def test_final_acceptance_5_capital_of_france_with_india_context_refused():
    """TEST 5:
    Question: 'What is the capital of France?'
    Context: 'Narendra Modi is the Prime Minister of India.'
    Expected: REFUSE, Confidence: LOW, grounded: false, refused: true
    """
    query = "What is the capital of France?"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Narendra Modi is the Prime Minister of India."
    chunks = [make_chunk(context, score=0.45)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False, "Context gate must reject entity mismatch (France vs India)."

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is False

    conf = calculate_final_confidence(
        result.question_relevant,
        result.answers_question,
        result.supported_by_context,
        result.confidence,
        result.supported
    )
    assert conf == "LOW"


# ==============================================================================
# SECTION 13: BALANCED TEST SUITE (VALID 1-7 & INVALID 8-12)
# ==============================================================================

# --- VALID QUERIES ---

def test_valid_2_us_national_flower_paraphrase():
    """Valid 2: 'US national flower?' with Rose context -> ACCEPT."""
    query = "US national flower?"
    context = "The Rose is the national flower of the United States and England."
    candidate = "The national flower of the US is the Rose."
    chunks = [make_chunk(context, score=0.86)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is True


def test_valid_3_what_flower_represents_united_states():
    """Valid 3: 'What flower represents the United States?' with Rose context -> ACCEPT."""
    query = "What flower represents the United States?"
    context = "The Rose is the national flower of the United States and England."
    candidate = "The Rose represents the United States as its national flower."
    chunks = [make_chunk(context, score=0.84)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is True


def test_valid_6_what_is_faiss():
    """Valid 6: 'What is FAISS?' with FAISS context -> ACCEPT."""
    query = "What is FAISS?"
    context = "FAISS is a library for efficient similarity search and clustering of dense vectors."
    candidate = "FAISS is a library used for efficient similarity search and clustering of vector embeddings."
    chunks = [make_chunk(context, score=0.89)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is True


def test_valid_7_what_is_rag():
    """Valid 7: 'What is retrieval augmented generation?' with RAG context -> ACCEPT."""
    query = "What is retrieval augmented generation?"
    context = "Retrieval-augmented generation (RAG) enhances LLM outputs by retrieving relevant documents from an external knowledge base."
    candidate = "RAG enhances language models by retrieving relevant information from an external knowledge base."
    chunks = [make_chunk(context, score=0.91)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is True

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is True


# --- INVALID QUERIES ---

def test_invalid_8_who_is_president_of_usa_with_rose_refused():
    """Invalid 8: 'Who is the President of the United States?' with Rose context -> REFUSE."""
    query = "Who is the President of the United States?"
    context = "The Rose is the national flower of the United States and England."
    candidate = "The Rose is the national flower of the United States."
    chunks = [make_chunk(context, score=0.60)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is False


def test_invalid_10_how_tall_is_mount_everest_refused():
    """Invalid 10: 'How tall is Mount Everest?' with unrelated context -> REFUSE."""
    query = "How tall is Mount Everest?"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Mount Everest is located in the Himalayas."
    chunks = [make_chunk(context, score=0.35)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is False


def test_invalid_11_what_is_india_population_refused():
    """Invalid 11: 'What is India's population?' with India PM-only context -> REFUSE."""
    query = "What is India's population?"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Narendra Modi is the Prime Minister of India."
    chunks = [make_chunk(context, score=0.52)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False, "Context gate must reject population query when context lacks demographic data."

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is False


def test_invalid_12_current_american_president_refused():
    """Invalid 12: 'Current American president' with India PM context -> REFUSE."""
    query = "Current American president"
    context = "Narendra Modi is the Prime Minister of India."
    candidate = "Narendra Modi is the Prime Minister of India."
    chunks = [make_chunk(context, score=0.48)]

    val = is_context_sufficient(query, chunks)
    assert val.sufficient is False

    validator = GroundingValidator()
    result = validator.verify_answer(query, candidate, chunks)
    assert result.supported is False
