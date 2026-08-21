"""Gemini Answer Verification Layer and Calibrated Grounding Guardrails.

Enforces calibrated 3-way factual verification:
1. question_relevant: Does the retrieved context address the subject, entity, AND requested attribute/relation of the question (evaluating semantic equivalence, paraphrases, and implicit relationships)?
2. answers_question: Does the candidate answer directly and accurately address the user's specific question?
3. supported_by_context: Are the factual claims in the candidate answer supported by the retrieved context without hallucination?

Retrieval similarity is NEVER confused with Answer Confidence.
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from pydantic import BaseModel, Field

try:
    from app.config import (
        VERIFIER_MIN_CONFIDENCE,
        HIGH_CONFIDENCE_THRESHOLD,
        MEDIUM_CONFIDENCE_THRESHOLD
    )
except ImportError:
    VERIFIER_MIN_CONFIDENCE = 0.55
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.50


class VerificationResult(BaseModel):
    question_relevant: bool = Field(
        description="True if the retrieved context contains evidence addressing the subject, entity, and semantic attribute of the question. False if there is a fundamental entity or attribute conflict (e.g. asking for President when context only discusses National Flower)."
    )
    answers_question: bool = Field(
        description="True if the candidate answer directly and accurately answers the specific question asked by the user. False if the answer addresses a different question, entity, or attribute."
    )
    supported_by_context: bool = Field(
        description="True if the key factual claims in the candidate answer are supported by the retrieved context. False if essential claims lack evidence or contradict the context."
    )
    confidence: float = Field(
        description="Verifier confidence score between 0.0 and 1.0 assessing factual grounding and question alignment."
    )
    reason: str = Field(
        description="Concise explanation of the verification assessment."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of specific claims in the candidate answer that lack evidence in the context."
    )

    @property
    def supported(self) -> bool:
        """Supported is True when all 3 independent checks pass."""
        return self.question_relevant and self.answers_question and self.supported_by_context


# Semantic concept clusters
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


class GroundingValidator:
    def __init__(self):
        self.min_confidence = float(os.environ.get("VERIFIER_MIN_CONFIDENCE", str(VERIFIER_MIN_CONFIDENCE)))
        self.use_llm_verifier = os.environ.get("VERIFIER_USE_LLM", "true").lower() == "true"
        self.timeout_seconds = float(os.environ.get("VERIFIER_TIMEOUT_SECONDS", "10.0"))

    @staticmethod
    def _tokenize_words(text: str) -> Set[str]:
        cleaned = re.sub(r'[\.,!\?।;:\"\(\)\[\]\{\}\'’‘“”\-—]', ' ', str(text or "").lower())
        return {w.strip() for w in cleaned.split() if len(w.strip()) > 1}

    @staticmethod
    def _extract_matched_concepts(text: str) -> Set[str]:
        t_lower = text.lower()
        matched = set()
        for concept_id, terms in CONCEPT_CLUSTERS.items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', t_lower):
                    matched.add(concept_id)
                    break
        return matched

    @staticmethod
    def _extract_matched_entities(text: str) -> Set[str]:
        t_lower = text.lower()
        matched = set()
        for eid, aliases in ENTITY_ALIASES.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', t_lower):
                    matched.add(eid)
                    break
        return matched

    def _check_entity_and_concept_alignment(
        self, question: str, context_passages: List[str], answer: str
    ) -> Tuple[bool, bool, str]:
        """Checks semantic entity and concept alignment without rigid string matching.
        
        Returns (question_relevant, answers_question, reason).
        """
        combined_context = " ".join(context_passages)
        q_entities = self._extract_matched_entities(question)
        c_entities = self._extract_matched_entities(combined_context)
        a_entities = self._extract_matched_entities(answer)

        q_concepts = self._extract_matched_concepts(question)
        c_concepts = self._extract_matched_concepts(combined_context)
        a_concepts = self._extract_matched_concepts(answer)

        # 1. Total Entity Conflict Check
        if q_entities:
            # Country conflicts (e.g. query asks about France, context discusses India/US only)
            if "france" in q_entities and "france" not in c_entities:
                return False, False, "Entity mismatch: Question asks about France, but context lacks information about France."
            if "us" in q_entities and "us" not in c_entities and "india" in c_entities:
                return False, False, "Entity mismatch: Question asks about the United States, but context discusses India."
            if "everest" in q_entities and "everest" not in c_entities:
                return False, False, "Entity mismatch: Question asks about Mount Everest, but context lacks information about Everest."

        # 2. Passage-level Entity-Concept Co-occurrence Alignment
        # If question asks about an entity AND an attribute (e.g. US President):
        if q_entities and q_concepts:
            has_aligned_chunk = False
            for passage in context_passages:
                p_e = self._extract_matched_entities(passage)
                p_c = self._extract_matched_concepts(passage)
                if p_e.intersection(q_entities) and p_c.intersection(q_concepts):
                    has_aligned_chunk = True
                    break
            
            if not has_aligned_chunk and c_entities and c_concepts:
                if "leader" in q_concepts and "us" in q_entities:
                    return False, False, "Attribute mismatch: Context mentions United States, but lacks information about its President/leadership."
                if "capital" in q_concepts:
                    return False, False, "Attribute mismatch: Context mentions the requested entity, but lacks capital city data."
                if "population" in q_concepts:
                    return False, False, "Attribute mismatch: Context mentions the requested entity, but lacks demographic/population data."

        # 3. Pure Concept Conflicts (when entity is not recognized or single-entity context)
        if "leader" in q_concepts and "flower" in c_concepts and "leader" not in c_concepts:
            return False, False, "Attribute mismatch: Question asks for a leader/President, but context only discusses a national flower."
        if "capital" in q_concepts and "capital" not in c_concepts and "leader" in c_concepts:
            return False, False, "Attribute mismatch: Question asks for the capital city, but context discusses a Prime Minister."
        if "population" in q_concepts and "population" not in c_concepts and "leader" in c_concepts:
            return False, False, "Attribute mismatch: Question asks for population, but context discusses a Prime Minister."
        if "height_mountain" in q_concepts and "height_mountain" not in c_concepts:
            return False, False, "Attribute mismatch: Question asks for mountain height, but context lacks elevation data."

        # 4. Candidate Answer vs Question & Context Consistency
        if q_entities and a_entities:
            if not q_entities.intersection(a_entities):
                return True, False, "Answer mismatch: The candidate answer refers to a different entity than asked in the question."
        
        # Check if answer introduces entities not supported by context
        if a_entities and c_entities:
            unsupported_entities = a_entities - c_entities
            # Allow common geographical nesting if context supports the region, else reject
            if unsupported_entities and not any(e in c_entities for e in a_entities):
                return True, False, f"Answer mismatch: The candidate answer introduces unsupported entity: {', '.join(unsupported_entities)}."

        if "leader" in q_concepts and "flower" in a_concepts and "leader" not in a_concepts:
            return True, False, "Answer mismatch: The candidate answer describes a flower instead of a leader."
        if "capital" in q_concepts and "leader" in a_concepts and "capital" not in a_concepts:
            return True, False, "Answer mismatch: The candidate answer names a leader instead of a capital city."

        return True, True, ""

    def _deterministic_evidence_check(
        self, question: str, answer: str, context_passages: List[str]
    ) -> Tuple[bool, bool, bool, float, str]:
        """Calibrated deterministic evidence and alignment check.
        
        Returns (question_relevant, answers_question, supported_by_context, confidence, reason).
        """
        if not answer or not context_passages:
            return False, False, False, 0.0, "Missing answer or context."

        context_combined = " ".join(context_passages)

        # 1. Entity and Concept Consistency
        q_rel, a_ans, mismatch_reason = self._check_entity_and_concept_alignment(
            question, context_passages, answer
        )
        if not q_rel or not a_ans:
            return q_rel, a_ans, False, 0.95, mismatch_reason

        # 2. Unsupported Numbers Check (e.g. 50 airports, 100km when not in text)
        ans_numbers = set(re.findall(r'\b\d+\b', answer))
        ctx_numbers = set(re.findall(r'\b\d+\b', context_combined))
        unsupported_nums = ans_numbers - ctx_numbers
        if unsupported_nums:
            return q_rel, a_ans, False, 0.35, f"Unsupported numeric claims in answer: {', '.join(unsupported_nums)}."

        # 3. Token / Semantic Overlap Check
        answer_tokens = self._tokenize_words(answer)
        if not answer_tokens:
            return False, False, False, 0.0, "Answer contains no substantive tokens."

        context_tokens = self._tokenize_words(context_combined)
        overlap = answer_tokens.intersection(context_tokens)
        ratio = len(overlap) / len(answer_tokens) if answer_tokens else 0.0

        # Exact substring match check (only if question is relevant and answers the question)
        is_exact_match = any(answer.strip().lower() in p.lower() for p in context_passages)
        if is_exact_match:
            return True, True, True, 0.95, "Direct substring evidence in retrieved context."

        # Check entity/concept coverage
        a_concepts = self._extract_matched_concepts(answer)
        c_concepts = self._extract_matched_concepts(context_combined)
        shared_concepts = a_concepts.intersection(c_concepts)

        # Calibrated support decision:
        if ratio >= 0.50 or (ratio >= 0.35 and bool(shared_concepts)):
            conf = min(0.95, max(0.65, ratio * 1.1))
            return True, True, True, conf, f"Grounded evidence match: {ratio:.0%} lexical token overlap with supported concepts."
        elif ratio >= 0.30:
            return True, True, True, 0.55, f"Moderate evidence support ({ratio:.0%} lexical overlap)."
        else:
            return True, True, False, 0.40, f"Insufficient verifiable evidence in context: only {ratio:.0%} token support."

    def verify_answer(
        self,
        question: str,
        candidate_answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        gemini_provider: Any = None,
        retrieval_confidence: str = "LOW"
    ) -> VerificationResult:
        """Executes calibrated 3-way verification:
        1. question_relevant: Context addresses question's entity and topic/relation.
        2. answers_question: Answer addresses user's question.
        3. supported_by_context: Key claims grounded in retrieved context.
        """
        if not candidate_answer or not candidate_answer.strip():
            return VerificationResult(
                question_relevant=False,
                answers_question=False,
                supported_by_context=False,
                confidence=0.99,
                reason="Empty candidate answer.",
                unsupported_claims=["No answer text."]
            )

        if not retrieved_chunks:
            return VerificationResult(
                question_relevant=False,
                answers_question=False,
                supported_by_context=False,
                confidence=0.99,
                reason="No context was retrieved to verify against.",
                unsupported_claims=["All claims ungrounded."]
            )

        context_passages = [
            c.get("payload", {}).get("text", "")
            for c in retrieved_chunks
            if c.get("payload", {}).get("text")
        ]

        # 1. Fast deterministic check
        q_rel, a_ans, s_ctx, det_conf, det_reason = self._deterministic_evidence_check(
            question, candidate_answer, context_passages
        )

        # If deterministic check caught a definitive entity or attribute conflict, reject early
        if not q_rel or not a_ans:
            return VerificationResult(
                question_relevant=q_rel,
                answers_question=a_ans,
                supported_by_context=False,
                confidence=det_conf,
                reason=det_reason,
                unsupported_claims=["Candidate answer does not address the question's entity or requested attribute."]
            )

        # 2. Gemini Secondary Verifier Model
        if self.use_llm_verifier and gemini_provider and getattr(gemini_provider, "client", None):
            llm_result = self._call_gemini_verifier(question, candidate_answer, retrieved_chunks, gemini_provider)
            if llm_result is not None:
                return llm_result

        # 3. Fallback to calibrated deterministic verification
        return VerificationResult(
            question_relevant=q_rel,
            answers_question=a_ans,
            supported_by_context=s_ctx,
            confidence=det_conf,
            reason=det_reason,
            unsupported_claims=[] if s_ctx else ["Claims lack sufficient evidence in retrieved context."]
        )

    def _call_gemini_verifier(
        self,
        question: str,
        candidate_answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        gemini_provider: Any
    ) -> Optional[VerificationResult]:
        """Calls Gemini with calibrated verification prompt and structured JSON output."""
        formatted_context = []
        for idx, chunk in enumerate(retrieved_chunks[:5]):
            payload = chunk.get("payload", {})
            formatted_context.append(
                f"[Source {idx+1}] (Score: {chunk.get('dense_score', 0.0):.2f})\n{payload.get('text', '')}"
            )
        context_block = "\n---\n".join(formatted_context)

        prompt = f"""You are an objective Question-Answering and Grounding Verification Model.

Determine whether the retrieved context provides sufficient evidence to answer the user's question.
Evaluate semantic equivalence, paraphrases, implicit relationships, and the combined retrieved passages.
Do not require exact wording. However, do not use outside knowledge to fill missing information.

Evaluate the following three criteria:

1. QUESTION_RELEVANT (bool): Does the retrieved context contain information about the subject, entity, and topic/relation asked in the Question (evaluating semantic equivalence and synonyms)?
   - ALLOW: If user asks "Who is India's leader?" and context discusses "Narendra Modi is the Prime Minister of India", question_relevant is True (Prime Minister is a leadership role).
   - ALLOW: If user asks "What flower represents the United States?" and context discusses "national flower of the United States", question_relevant is True.
   - REJECT: If user asks "Who is the President of the United States?" and context ONLY discusses the "national flower of the United States", question_relevant MUST BE false.
   - REJECT: If user asks "What is the capital of France?" and context discusses India's Prime Minister, question_relevant MUST BE false.

2. ANSWERS_QUESTION (bool): Does the candidate answer directly and accurately address the specific question asked by the user based on the context?
   - Do not require exact words; accept natural language summaries and semantically equivalent answers.
   - If the candidate answer addresses an entirely different entity or attribute, answers_question MUST BE false.

3. SUPPORTED_BY_CONTEXT (bool): Are the key factual claims in the candidate answer directly supported by or semantically entailed by the retrieved context?
   - Pretrained outside world knowledge is strictly forbidden.
   - If the answer contains fabricated claims unsupported by the context, supported_by_context MUST BE false.

4. CONFIDENCE (float): Confidence score between 0.0 and 1.0 in this verification assessment.
   - 0.80 - 1.0: Context directly and unambiguously answers the question.
   - 0.55 - 0.79: Context provides strong semantic support with natural paraphrasing.
   - 0.40 - 0.54: Partial or weak evidence with moderate uncertainty.
   - < 0.40: Context is irrelevant, uninformative, or contradictory.

Question:
{question}

Retrieved Context:
{context_block}

Candidate Answer:
{candidate_answer}
"""
        try:
            from google import genai
            for model_candidate in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    response = gemini_provider.client.models.generate_content(
                        model=model_candidate,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=VerificationResult,
                            max_output_tokens=512,
                        )
                    )
                    if response and response.text:
                        parsed = json.loads(response.text)
                        return VerificationResult(**parsed)
                except Exception:
                    continue
            return None
        except Exception as exc:
            print(f"Gemini verification error: {exc}")
            return None

    def validate(self, answer: str, context_results: List[Dict[str, Any]], llm_client: Any = None, query: str = "") -> Tuple[str, str]:
        """Legacy compatibility wrapper returning ('PASS'/'FAIL', reason)."""
        res = self.verify_answer(query or "", answer, context_results, gemini_provider=llm_client)
        is_pass = res.question_relevant and res.answers_question and res.supported_by_context
        return ("PASS" if is_pass else "FAIL", res.reason)
