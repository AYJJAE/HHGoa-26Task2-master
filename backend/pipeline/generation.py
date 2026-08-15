import os
import re
import json
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

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
        
        if self.api_key:
            try:
                global genai
                import google.generativeai as genai
                model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                print(f"Initialized Gemini Provider (Live API with {model_name})")
            except Exception as e:
                print(f"Gemini init failed: {e}. Falling back to extractive.")
                self.model = None
        else:
            print("No GEMINI_API_KEY found. Using extractive answer generation.")
            
    def check_answerability(self, query: str, context: List[str]) -> Dict[str, Any]:
        if not self.model:
            return {"answerable": True, "confidence": 0.5, "reason": "Fallback mode - skipping check."}
            
        context_block = "\n---\n".join(context[:5])
        prompt = f"""Evaluate if the following context contains enough factual information to answer the question.
Do NOT use pretrained knowledge. The context is the ONLY source of truth.

Context:
{context_block}

Question: {query}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=AnswerabilityResult
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Answerability check failed: {e}")
            return {"answerable": False, "confidence": 0.0, "reason": f"Error: {e}"}
        
    def generate(self, query: str, context: List[str]) -> str:
        if self.model:
            return self._generate_with_gemini(query, context)
        else:
            return self._extractive_answer(query, context)
    
    def _generate_with_gemini(self, query: str, context: List[str]) -> str:
        """Call real Gemini API for answer generation."""
        context_block = "\n---\n".join(context[:5])  # Top 5 passages
        
        prompt = f"""You are a grounded RAG answer generator.

You MUST answer ONLY using the supplied CONTEXT.
The user's question is NOT evidence.
Your pretrained knowledge is NOT evidence.
Do not use outside knowledge.
Do not infer facts that are not supported by the context.

If the context does not contain enough information to answer the question,
return exactly:
INSUFFICIENT_CONTEXT

You MUST answer in the EXACT SAME LANGUAGE as the user's Question. (e.g., if the Question is in Hindi, answer in Hindi).

Context:
{context_block}

Question: {query}

Answer:"""

        for attempt in range(2):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                print(f"Gemini generation error (attempt {attempt+1}): {error_str}")
                if attempt == 0 and ("429" in error_str or "quota" in error_str.lower()):
                    print("Rate limited — waiting 12s before retry...")
                    time.sleep(12)
                    continue
                return self._extractive_answer(query, context)
    
    def _tokenize_words(self, text: str) -> set:
        cleaned = re.sub(r'[\.,!\?।;:\"\(\)\[\]\{\}\'’‘“”\-—]', ' ', text.lower())
        return {w.strip() for w in cleaned.split() if w.strip()}

    def _extractive_answer(self, query: str, context: List[str]) -> str:
        """Fallback: return the most relevant passage or sentence directly as the answer."""
        if not context:
            return "I couldn't find enough information in the dataset to answer that."
        
        q_words = self._tokenize_words(query)
        if not q_words:
            return context[0][:500]
            
        all_sentences = []
        for passage in context:
            s_list = [s.strip() for s in re.split(r'[.!?।\n]', passage) if len(s.strip()) > 8]
            all_sentences.extend(s_list)
            
        best_sentence = None
        best_score = -1.0
        
        for s in all_sentences:
            s_words = self._tokenize_words(s)
            if not s_words:
                continue
            intersection = q_words.intersection(s_words)
            union = q_words.union(s_words)
            jaccard = len(intersection) / len(union) if union else 0.0
            # Density weighted score: favors tight matches over verbose distractors
            score = len(intersection) + jaccard * 5.0
            if score > best_score:
                best_score = score
                best_sentence = s
                
        if best_sentence and best_score > 1.0:
            punct = "।" if any('\u0900' <= c <= '\u097F' for c in best_sentence) else "."
            return best_sentence + (punct if not best_sentence.endswith(('.', '!', '?', '।')) else '')

        return context[0][:500]


class GenerationPipeline:
    def __init__(self, provider_name: str = "gemini"):
        self.provider_name = provider_name.lower()
        if self.provider_name == "gemini":
            self.provider = GeminiProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
            
    def check_answerability(self, query: str, context_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        passages = [c['payload']['text'] for c in context_results]
        return self.provider.check_answerability(query, passages)
            
    def generate_answer(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the configured LLM and strict RAG prompt.
        """
        passages = [c['payload']['text'] for c in context_results]
        return self.provider.generate(query, passages)

