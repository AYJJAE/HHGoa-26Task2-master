import json
import os
import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

# Configurable grounding threshold via environment variable
GROUNDING_OVERLAP_THRESHOLD = float(os.environ.get("GROUNDING_THRESHOLD", "0.6"))

class GroundingResult(BaseModel):
    grounded: bool = Field(description="True ONLY if every single factual claim in the answer is explicitly supported by the context. If uncertain, false.")
    reason: str = Field(description="Reasoning for the grounding decision.")

class GroundingValidator:
    def __init__(self):
        self.threshold = GROUNDING_OVERLAP_THRESHOLD

    def _tokenize_words(self, text: str) -> set:
        cleaned = re.sub(r'[\.,!\?।;:\"\(\)\[\]\{\}\'’‘“”\-—]', ' ', text.lower())
        return {w.strip() for w in cleaned.split() if len(w.strip()) > 1}

    def _lexical_grounding_fallback(self, answer: str, passages: List[str]) -> Tuple[str, str]:
        """
        Lightweight lexical grounding: checks if key tokens from the answer appear in the context.
        Used as a fallback when the LLM grounding call fails (rate limits, errors, etc.).
        """
        if not answer or not passages:
            return "FAIL", "Missing answer or context for lexical check."
        
        answer_tokens = self._tokenize_words(answer)
        if not answer_tokens:
            return "FAIL", "Answer has no meaningful tokens."
        
        # Combine all context passages
        context_combined = " ".join(passages)
        context_tokens = self._tokenize_words(context_combined)
        
        overlap = answer_tokens.intersection(context_tokens)
        ratio = len(overlap) / len(answer_tokens) if answer_tokens else 0
        
        if ratio >= self.threshold:
            return "PASS", f"Lexical fallback: {ratio:.0%} of answer tokens found in context (threshold: {self.threshold:.0%})."
        else:
            return "FAIL", f"Lexical fallback: only {ratio:.0%} of answer tokens found in context (threshold: {self.threshold:.0%})."

    def validate(self, answer: str, context_results: List[Dict[str, Any]], llm_client: Any = None) -> Tuple[str, str]:
        """
        Runs the full grounding pipeline with strict claim-level verification using the LLM.
        Falls back to lexical check if LLM is unavailable (rate limits, errors).
        Returns (Verdict, Reason) -> "PASS" or "FAIL"
        """
        if not answer or not context_results:
            return "FAIL", "Missing answer or context."
            
        passages = [c['payload']['text'] for c in context_results]
        
        if not llm_client or not getattr(llm_client, "model", None):
            # No LLM available — use lexical fallback
            return self._lexical_grounding_fallback(answer, passages)
            
        context_block = "\n---\n".join(passages[:5])
        
        prompt = f"""Verify if the provided Answer is strictly grounded in the provided Context.

Instructions:
1. Extract all factual claims from the Answer.
2. For each claim, check if it is explicitly supported by the Context.
3. If ANY claim is NOT supported by the Context, or if you are uncertain, you MUST return grounded=false.
4. Only return grounded=true if ALL claims are directly supported. Do not rely on external knowledge.

Context:
{context_block}

Answer to verify:
{answer}
"""
        
        try:
            import google.generativeai as genai
            response = llm_client.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=GroundingResult
                )
            )
            result = json.loads(response.text)
            if result.get("grounded") is True:
                return "PASS", result.get("reason", "All claims supported.")
            else:
                return "FAIL", result.get("reason", "Not all claims supported.")
        except Exception as e:
            error_str = str(e)
            print(f"Grounding LLM check failed: {error_str}. Falling back to lexical grounding.")
            return self._lexical_grounding_fallback(answer, passages)

