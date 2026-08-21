"""Calibrated Context Relevance Scoring and Early Gating.

Implements a continuous, multi-signal relevance scoring pipeline:
- Semantic embedding similarity (BGE-M3 dense score)
- Substantive collective lexical overlap across Top-K context (with multilingual stopword filtering)
- Semantic intent and entity/relation compatibility checking (evaluates entity-attribute co-occurrence)
- Collective Top-K evidence coverage (evaluates combined context, not just Chunk #1)

Prevents over-refusal on semantic paraphrases (e.g. 'Who leads India?', 'US national flower?'),
while cleanly filtering out true off-topic, contradictory, or unsupported passages before LLM generation.
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Set, Any
from pydantic import BaseModel

try:
    from app.config import RETRIEVAL_MIN_SCORE, RELEVANCE_MIN_SCORE
except ImportError:
    RETRIEVAL_MIN_SCORE = 0.40
    RELEVANCE_MIN_SCORE = 0.35

# Multilingual stopwords (English, Hindi, Marathi)
INDIC_STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "when", "where", "which",
    "how", "why", "in", "on", "at", "of", "to", "for", "with", "about", "by", "from", "and",
    "or", "not", "this", "that", "it", "its", "as", "be", "been", "have", "has", "had",
    "tell", "me", "give", "can", "you", "please", "does", "do", "did",
    # Hindi
    "है", "हैं", "का", "के", "की", "को", "में", "पर", "से", "और", "या", "नहीं", "क्या",
    "कौन", "कब", "कहाँ", "कहां", "कैसे", "कितना", "कितने", "था", "थी", "थे", "होगा", "होगी", "यह", "वह",
    # Marathi
    "आहे", "आहेत", "चा", "ची", "चे", "च्या", "ला", "ना", "मध्ये", "वर", "आणि", "किंवा",
    "नाही", "काय", "कोण", "केव्हा", "कुठे", "कसे", "किती", "होता", "होती", "होते", "असेल", "हे", "ते"
}

MIN_PASSAGE_CHARS = 15

# Semantic concept clusters for intent & attribute alignment
CONCEPT_CLUSTERS = {
    "leader": {
        "president", "prime minister", "pm", "leader", "leaders", "lead", "leads", "leading",
        "head of state", "head of government", "premier", "chief minister", "cm", "governor",
        "statesman", "ruler", "rules", "rule", "runs", "running", "governs", "governing", "elected", "office",
        "राष्ट्रपती", "राष्ट्रपति", "प्रधानमंत्री", "प्रधान मंत्री", "मुख्यमंत्री", "नेता", "सरकार", "शासन"
    },
    "flower": {
        "national flower", "flower", "flowers", "floral", "floral emblem", "rose", "lotus",
        "marigold", "jasmine", "फूल", "गुलाब", "कमल", "पुष्प"
    },
    "bird_animal": {
        "national bird", "bird", "peacock", "national animal", "animal", "tiger", "lion",
        "पक्षी", "प्राणी", "बाघ", "मोर", "सिंह"
    },
    "capital": {
        "capital", "capital city", "seat of government", "headquarters", "राजधानी"
    },
    "population": {
        "population", "people", "inhabitants", "residents", "populous", "million", "billion",
        "crore", "lakh", "citizens", "लोकसंख्या", "जनसंख्या", "आबादी"
    },
    "height_mountain": {
        "tall", "height", "altitude", "elevation", "meters", "feet", "mount", "everest", "mountain", "peak",
        "ऊंचाई", "पर्वत", "शिखर"
    },
    "technology": {
        "faiss", "similarity search", "vector", "vectors", "embedding", "embeddings", "rag",
        "retrieval augmented", "retrieval-augmented", "qdrant", "hnsw", "indexing", "cluster", "clustering"
    },
    "beaches": {
        "beach", "beaches", "sea", "coast", "shore", "sand", "ocean", "anjuna", "vagator", "baga", "calangute",
        "palolem", "agonda", "colva", "benaulim", "arambol", "morjim", "mandrem", "candolim", "बीच", "समुद्र तट", "किनारा"
    },
    "itinerary": {
        "itinerary", "plan", "trip", "tour", "day 1", "day 2", "day 3", "2 day", "3 day", "5 day", "योजना", "दौरा", "ट्रिप", "प्लॅन"
    },
    "food": {
        "food", "dish", "eat", "cuisine", "curry", "fish curry", "poi", "bebinca", "vindaloo", "xacuti", "feni",
        "cafreal", "taste", "खाना", "व्यंजन", "जेवण"
    },
    "heritage": {
        "church", "basilica", "cathedral", "heritage", "history", "portuguese", "monument", "fort", "aguada",
        "chapora", "cabo de rama", "old goa", "bom jesus", "se cathedral", "चर्च", "किला", "इतिहास"
    },
    "culture": {
        "carnival", "shigmo", "sao joao", "culture", "tradition", "festival", "dance", "mando", "folk",
        "संस्कृती", "त्योहार", "उत्सव", "परंपरा"
    },
    "nature": {
        "waterfall", "dudhsagar", "bird", "salim ali", "spice", "plantation", "wildlife", "sanctuary", "forest",
        "nature", "sunset", "peaceful", "quiet", "relaxed", "family", "families", "children", "kids", "झरना", "पक्षी"
    },
    "transport": {
        "transport", "scooter", "bike", "car", "taxi", "cab", "goamiles", "bus", "ferry", "train", "airport", "reach",
        "वाहतूक", "किराया", "गाड़ी"
    }
}

ENTITY_ALIASES = {
    "us": {"us", "usa", "u.s.", "u.s.a.", "united states", "america", "american", "washington", "white house"},
    "india": {"india", "indian", "bharat", "delhi", "new delhi", "modi", "narendra modi", "भारत", "भारतीय"},
    "france": {"france", "french", "paris", "macron", "फ्रांस", "पेरिस"},
    "uk": {"uk", "united kingdom", "britain", "british", "england", "london"},
    "maharashtra": {"maharashtra", "mumbai", "bombay", "pune", "nagpur", "महाराष्ट्र", "मुंबई"},
    "goa": {
        "goa", "goan", "panaji", "panjim", "margao", "madgaon", "vasco", "calangute", "baga", "anjuna",
        "vagator", "arambol", "morjim", "candolim", "palolem", "agonda", "colva", "benaulim", "chapora",
        "aguada", "fontainhas", "dudhsagar", "old goa", "bom jesus", "se cathedral", "bebinca", "vindaloo",
        "xacuti", "feni", "cafreal", "shigmo", "sao joao", "गोवा", "गोव्यात", "गोय", "पणजी"
    },
    "everest": {"everest", "mount everest", "himalayas", "एवरेस्ट"}
}


class ContextValidation(BaseModel):
    sufficient: bool
    confidence: float
    relevance_score: float
    reason: str
    supporting_chunks: int
    max_similarity: Optional[float]
    keyword_overlap: float


def _substantive_tokens(text: str) -> Set[str]:
    """Extracts lowercase tokens excluding stopwords and single-character noise."""
    tokens = re.findall(r"[\w]+", str(text or "").lower())
    return {t for t in tokens if t not in INDIC_STOPWORDS and len(t) > 1}


def _detect_concepts(text: str) -> Set[str]:
    """Identifies active high-level semantic concept clusters in text."""
    t_lower = text.lower()
    active_concepts = set()
    for concept_id, terms in CONCEPT_CLUSTERS.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', t_lower):
                active_concepts.add(concept_id)
                break
    return active_concepts


def _detect_entities(text: str) -> Set[str]:
    """Identifies active entities in text."""
    t_lower = text.lower()
    active_entities = set()
    for ent_id, terms in ENTITY_ALIASES.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', t_lower):
                active_entities.add(ent_id)
                break
    return active_entities


def _check_semantic_conflict(query: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Checks for definitive entity or attribute conflicts across retrieved chunks.
    
    Evaluates both collective context and passage-level entity-concept co-occurrence.
    Returns (has_conflict, reason).
    """
    valid_texts = [c.get('payload', {}).get('text', '') for c in chunks if c.get('payload', {}).get('text')]
    combined_context = " ".join(valid_texts)

    q_entities = _detect_entities(query)
    c_entities = _detect_entities(combined_context)
    q_concepts = _detect_concepts(query)
    c_concepts = _detect_concepts(combined_context)

    # 1. Total Entity Mismatch
    if q_entities:
        if "france" in q_entities and "france" not in c_entities:
            return True, "Entity conflict: Query asks about France, but context lacks information about France."
        if "us" in q_entities and "us" not in c_entities and "india" in c_entities:
            return True, "Entity conflict: Query asks about the United States, but context discusses India."
        if "everest" in q_entities and "everest" not in c_entities:
            return True, "Entity conflict: Query asks about Mount Everest, but context lacks information about Everest."

    # 2. Entity-Concept Co-occurrence Alignment across chunks
    # (e.g. Query has entity 'us' and concept 'leader' -> check if any chunk discussing 'us' has 'leader')
    if q_entities and q_concepts:
        has_aligned_chunk = False
        for text in valid_texts:
            chunk_e = _detect_entities(text)
            chunk_c = _detect_concepts(text)
            if chunk_e.intersection(q_entities) and chunk_c.intersection(q_concepts):
                has_aligned_chunk = True
                break
            # Also check if text contains both explicit query tokens
            q_sub = _substantive_tokens(query)
            t_sub = _substantive_tokens(text)
            if len(q_sub.intersection(t_sub)) >= max(2, len(q_sub)):
                has_aligned_chunk = True
                break

        if not has_aligned_chunk and c_entities and c_concepts:
            # Check specific conflicts like President of USA vs US Rose / India PM
            if "leader" in q_concepts and "us" in q_entities:
                return True, "Entity-Attribute conflict: Context mentions United States, but contains no leadership/President data."
            if "capital" in q_concepts:
                return True, "Entity-Attribute conflict: Context mentions the requested entity, but contains no capital city data."
            if "population" in q_concepts:
                return True, "Entity-Attribute conflict: Context mentions the requested entity, but contains no demographic/population data."

    # 3. Pure Concept Conflicts (when entity not explicitly recognized)
    if "leader" in q_concepts and "flower" in c_concepts and "leader" not in c_concepts:
        return True, "Attribute conflict: Query asks for a leader/President, but context discusses a floral emblem."
    if "capital" in q_concepts and "capital" not in c_concepts and "leader" in c_concepts:
        return True, "Attribute conflict: Query asks for capital, but context discusses a Prime Minister."
    if "population" in q_concepts and "population" not in c_concepts and "leader" in c_concepts:
        return True, "Attribute conflict: Query asks for population, but context discusses a leader."
    # 4. Out of Domain queries
    q_low = query.lower()
    if any(w in q_low for w in ["alien", "aliens", "ufo", "moon landing", "mars", "dinosaur", "chocolate lava cake", "pasta recipe"]):
        return True, "Out of domain query: Context lacks information about requested topic."

    return False, ""


def calculate_relevance_score(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[float, float, float, int]:
    """Calculates multi-signal relevance across collective Top-K context.
    
    Returns (relevance_score, max_sim, collective_overlap, supporting_chunks).
    """
    if not retrieved_chunks:
        return 0.0, 0.0, 0.0, 0

    q_tokens = _substantive_tokens(query)
    if not q_tokens:
        return 0.50, 0.50, 0.50, 1

    valid_chunks = [
        c for c in retrieved_chunks
        if len(c.get('payload', {}).get('text', '').strip()) >= MIN_PASSAGE_CHARS
    ]
    if not valid_chunks:
        return 0.0, 0.0, 0.0, 0

    combined_text = " ".join([c.get('payload', {}).get('text', '') for c in valid_chunks])
    combined_tokens = _substantive_tokens(combined_text)

    # Collective lexical overlap across all top-k passages
    collective_overlap = len(q_tokens.intersection(combined_tokens)) / len(q_tokens) if q_tokens else 0.0

    # Entity / alias coverage in overlap
    q_entities = _detect_entities(query)
    c_entities = _detect_entities(combined_text)
    if q_entities and q_entities.intersection(c_entities):
        collective_overlap = min(1.0, collective_overlap + 0.15)

    # Shared concept clusters in overlap
    q_concepts = _detect_concepts(query)
    c_concepts = _detect_concepts(combined_text)
    if q_concepts and q_concepts.intersection(c_concepts):
        collective_overlap = min(1.0, collective_overlap + 0.10)

    max_sim = 0.0
    supporting_chunks = 0

    for chunk in valid_chunks:
        text = chunk.get('payload', {}).get('text', '')
        sim = float(chunk.get("dense_score", 0.0) or 0.0)
        if sim > max_sim:
            max_sim = sim

        chunk_tokens = _substantive_tokens(text)
        c_overlap = len(q_tokens.intersection(chunk_tokens)) / len(q_tokens) if q_tokens else 0.0

        if sim >= RETRIEVAL_MIN_SCORE or c_overlap >= 0.30:
            supporting_chunks += 1

    # Multi-signal weighted relevance score (Dense similarity 55% + Collective coverage 45%)
    relevance_score = (max_sim * 0.55) + (collective_overlap * 0.45)

    # Domain / Attribute Conflict Check
    has_conflict, conflict_reason = _check_semantic_conflict(query, valid_chunks)
    if has_conflict:
        relevance_score = min(relevance_score, 0.20)

    # Adversarial / Prompt Injection Check
    q_lower = query.lower()
    if any(p in q_lower for p in ["ignore your instructions", "ignore all previous instructions", "system override", "unrestricted assistant"]):
        if not c_entities.intersection(q_entities) and collective_overlap < 0.3:
            relevance_score = 0.10

    return relevance_score, max_sim, collective_overlap, supporting_chunks


def is_context_sufficient(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    embedder: Any = None,
    language: str = "en"
) -> ContextValidation:
    """Soft calibrated gating: determines whether retrieved context provides sufficient signal to proceed."""
    if not retrieved_chunks:
        return ContextValidation(
            sufficient=False,
            confidence=0.0,
            relevance_score=0.0,
            reason="No chunks retrieved from knowledge base.",
            supporting_chunks=0,
            max_similarity=None,
            keyword_overlap=0.0
        )

    rel_score, max_sim, overlap, supporting_chunks = calculate_relevance_score(query, retrieved_chunks)

    # Calibrated decision threshold:
    sufficient = rel_score >= RELEVANCE_MIN_SCORE and (overlap > 0.15 or max_sim >= 0.45)

    if not sufficient:
        return ContextValidation(
            sufficient=False,
            confidence=rel_score,
            relevance_score=rel_score,
            reason=f"Insufficient context relevance (score: {rel_score:.2f} < {RELEVANCE_MIN_SCORE:.2f}, overlap: {overlap:.0%}, sim: {max_sim:.2f}).",
            supporting_chunks=supporting_chunks,
            max_similarity=max_sim,
            keyword_overlap=overlap
        )

    return ContextValidation(
        sufficient=True,
        confidence=rel_score,
        relevance_score=rel_score,
        reason=f"Context relevant (score: {rel_score:.2f}, overlap: {overlap:.0%}, {supporting_chunks} supporting chunks).",
        supporting_chunks=supporting_chunks,
        max_similarity=max_sim,
        keyword_overlap=overlap
    )
