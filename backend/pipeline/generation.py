import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "when", "where", "which",
    "how", "why", "in", "on", "at", "of", "to", "for", "with", "about", "by", "from", "and",
    "or", "not", "this", "that", "it", "its", "as", "be", "been", "have", "has", "had",
    "tell", "me", "give", "can", "you", "please", "does", "do", "did",
    "है", "हैं", "का", "के", "की", "को", "में", "पर", "से", "और", "या", "नहीं", "क्या",
    "कौन", "कब", "कहाँ", "कहां", "था", "थी", "थे", "होगा", "होगी", "यह", "वह",
    "आहे", "आहेत", "चा", "ची", "चे", "च्या", "ला", "ना", "मध्ये", "वर", "आणि", "किंवा",
    "नाही", "काय", "कोण", "केव्हा", "कुठे", "होता", "होती", "होते"
}

class AnswerabilityResult(BaseModel):
    answerable: bool = Field(description="True if the provided context contains sufficient information to fully answer the query.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 of this assessment.")
    reason: str = Field(description="Detailed reason explaining why the context is or is not sufficient.")

class LLMProvider:
    def check_answerability(self, query: str, context: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def generate(self, query: str, context: List[str]) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        self.timeout_seconds = float(os.environ.get("RAG_LLM_TIMEOUT_SECONDS", "20"))
        self.max_context_chars = max(256, int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "12000")))
        self.max_output_tokens = max(32, int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "512")))
        self.answerability_use_llm = os.environ.get("RAG_ANSWERABILITY_USE_LLM", "false").lower() == "true"
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.model = True
                print(f"Initialized Gemini Provider (Live API with {self.model_name})")
            except Exception as e:
                print(f"Gemini init failed: {e}. Falling back to extractive.")
                self.model = None
                self.client = None
        else:
            self.client = None
            print("No GEMINI_API_KEY found. Using extractive answer generation.")
            
    def check_answerability(self, query: str, context: List[str]) -> Dict[str, Any]:
        from pipeline.context_gate import is_context_sufficient
        chunks = [{"payload": {"text": c}} for c in context]
        val = is_context_sufficient(query, chunks)
        if not val.sufficient:
            return {
                "answerable": False,
                "confidence": val.confidence,
                "reason": val.reason,
            }

        if not self.answerability_use_llm:
            return {
                "answerable": True,
                "confidence": val.confidence,
                "reason": val.reason,
            }
        if not self.model or not self.client:
            return {"answerable": False, "confidence": 0.0, "reason": "LLM answerability is unavailable."}
            
        context_block = "\n---\n".join(context[:5])
        prompt = f"""Evaluate if the following context contains enough factual information to answer the question.
Do NOT use pretrained knowledge. The context is the ONLY source of truth.

Context:
{context_block}

Question:
{query}
"""
        try:
            from google import genai
            for model_candidate in [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    response = self.client.models.generate_content(
                        model=model_candidate,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AnswerabilityResult
                        )
                    )
                    return json.loads(response.text)
                except Exception:
                    continue
            return {"answerable": False, "confidence": 0.0, "reason": "All models failed check"}
        except Exception as e:
            return {"answerable": False, "confidence": 0.0, "reason": f"Error: {e}"}
        
    def generate(self, query: str, context: List[str], history: Optional[List[Dict[str, str]]] = None) -> str:
        if self.model and self.client:
            return self._generate_with_gemini(query, context, history=history)
        else:
            return self._extractive_answer(query, context)
    
    def _generate_with_gemini(self, query: str, context: List[str], history: Optional[List[Dict[str, str]]] = None) -> str:
        """Call real Gemini API for grounded answer generation with multi-model fallback."""
        context_block = "\n---\n".join(context)[:self.max_context_chars]
        
        history_block = ""
        if history:
            recent_turns = []
            for h in history[-4:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                recent_turns.append(f"{role}: {h.get('content', '')}")
            if recent_turns:
                history_block = "Previous Conversation History:\n" + "\n".join(recent_turns) + "\n\n"

        prompt = f"""You are a strictly grounded RAG answer generator.

CRITICAL RULES:
1. You MUST answer ONLY using the facts explicitly stated in the supplied Context.
2. The user's question is NOT evidence. Your pretrained knowledge is NOT evidence.
3. If the Context does not contain the factual answer to the Question, or if the Question asks about future/hypothetical/speculative events not explicitly documented in the context (e.g. year 2050, alien visits, Mars colonies), you MUST respond with EXACTLY:
INSUFFICIENT_CONTEXT
4. Do NOT attempt to answer from general knowledge. Do NOT provide speculative explanations if the direct answer is missing.
5. You MUST answer in the EXACT SAME LANGUAGE as the user's Question. (e.g., Hindi for Hindi, Marathi for Marathi, Konkani for Konkani).

Context:
{context_block}

{history_block}Question: {query}

Answer:"""

        try:
            from google import genai
            for model_candidate in [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    response = self.client.models.generate_content(
                        model=model_candidate,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(max_output_tokens=self.max_output_tokens)
                    )
                    if response and response.text:
                        text = response.text.strip()
                        if "INSUFFICIENT_CONTEXT" in text or "उपलब्ध नाही" in text or "माहिती नाही" in text or "not available" in text.lower() or "not mentioned" in text.lower():
                            return "INSUFFICIENT_CONTEXT"
                        return text
                except Exception as model_err:
                    continue
            return self._extractive_answer(query, context)
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return self._extractive_answer(query, context)
    
    def _tokenize_words(self, text: str) -> set:
        cleaned = re.sub(r'[\.,!\?।;:\"\(\)\[\]\{\}\'’‘“”\-—]', ' ', str(text or "").lower())
        return {w.strip() for w in cleaned.split() if w.strip() and w.strip() not in STOPWORDS and len(w.strip()) > 1}

    def _extractive_answer(self, query: str, context: List[str]) -> str:
        """Fallback: return the most relevant passage or complete sentences directly as the answer."""
        if not context:
            return "INSUFFICIENT_CONTEXT"
        
        query_words = self._tokenize_words(query)
        if not query_words:
            return context[0].strip() if context else "INSUFFICIENT_CONTEXT"

        best_score = -1.0
        best_passage = ""
        
        for passage in context:
            p_words = self._tokenize_words(passage)
            if not p_words:
                continue
            overlap = query_words.intersection(p_words)
            score = len(overlap) / len(query_words) if query_words else 0
            if score > best_score:
                best_score = score
                best_passage = passage.strip()
        
        if best_score < 0.20 or not best_passage:
            return "INSUFFICIENT_CONTEXT"
            
        return best_passage

class GenerationPipeline:
    def __init__(self, provider_name: str = "gemini"):
        self.provider_name = provider_name
        self.provider = GeminiProvider()

    def _extract_texts(self, context_results: List[Any]) -> List[str]:
        context = []
        for r in context_results:
            if isinstance(r, dict):
                if 'payload' in r and isinstance(r['payload'], dict) and 'text' in r['payload']:
                    context.append(r['payload']['text'])
                elif 'text' in r:
                    context.append(r['text'])
                else:
                    context.append(str(r))
            else:
                context.append(str(r))
        return context

    def check_answerability(self, query: str, context_results: List[Any]) -> Dict[str, Any]:
        context = self._extract_texts(context_results)
        return self.provider.check_answerability(query, context)

    def generate_answer(self, query: str, context_results: List[Any], history: Optional[List[Dict[str, str]]] = None) -> str:
        context = self._extract_texts(context_results)
        return self.provider.generate(query, context, history=history)

