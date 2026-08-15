import os
import re
import nltk
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from pipeline.embeddings import EmbeddingPipeline
import numpy as np

# Ensure NLTK resources are available (stopwords)
try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Configurable Thresholds
RAG_MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.80"))
RAG_MIN_KEYWORD_OVERLAP = float(os.environ.get("RAG_MIN_KEYWORD_OVERLAP", "0.20"))
RAG_MIN_SUPPORTING_CHUNKS = int(os.environ.get("RAG_MIN_SUPPORTING_CHUNKS", "1"))

MIN_PASSAGE_CHARS = 20

class ContextValidation(BaseModel):
    sufficient: bool
    confidence: float
    reason: str
    supporting_chunks: int
    max_similarity: Optional[float]
    keyword_overlap: float

def _keyword_overlap(query: str, text: str) -> float:
    """Returns the ratio of query keywords found in the text (0.0 to 1.0)."""
    q_tokens = {t.lower() for t in re.findall(r"\w+", query) if t.lower() not in STOPWORDS and len(t) > 1}
    if not q_tokens:
        return 0.0
    t_tokens = {t.lower() for t in re.findall(r"\w+", text) if t.lower() not in STOPWORDS}
    overlap = q_tokens.intersection(t_tokens)
    return len(overlap) / len(q_tokens)

def is_context_sufficient(query: str, retrieved_chunks: List[Dict[str, any]], embedder: EmbeddingPipeline, language: str = "en") -> ContextValidation:
    """Determine if the retrieved context likely contains enough evidence to answer the query."""
    if not retrieved_chunks:
        return ContextValidation(
            sufficient=False,
            confidence=0.0,
            reason="No chunks retrieved.",
            supporting_chunks=0,
            max_similarity=None,
            keyword_overlap=0.0
        )

    # 0. Filter out empty / trivially short chunks
    valid_chunks = []
    for chunk in retrieved_chunks:
        text = chunk.get('payload', {}).get('text', '')
        if text and len(text.strip()) >= MIN_PASSAGE_CHARS:
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ContextValidation(
            sufficient=False,
            confidence=0.0,
            reason=f"All {len(retrieved_chunks)} retrieved chunks are empty or shorter than {MIN_PASSAGE_CHARS} chars.",
            supporting_chunks=0,
            max_similarity=None,
            keyword_overlap=0.0
        )

    best_overlap = 0.0
    for chunk in valid_chunks:
        text = chunk.get('payload', {}).get('text', '')
        overlap = _keyword_overlap(query, text)
        if overlap > best_overlap:
            best_overlap = overlap

    # Get query embedding for semantic similarity check
    query_emb = None
    try:
        query_emb = embedder.embed_queries([query])[0]
        if isinstance(query_emb, tuple):
            query_emb = query_emb[0]
    except Exception as e:
        pass # Handle below if needed, but it's optional

    max_sim = None
    supporting_chunks = 0

    for chunk in valid_chunks:
        text = chunk.get('payload', {}).get('text', '')
        
        sim = None
        if query_emb is not None:
            try:
                passage_emb = embedder.embed_documents([text])[0]
                if isinstance(passage_emb, tuple):
                    passage_emb = passage_emb[0]
                sim = np.dot(query_emb, passage_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(passage_emb) + 1e-9)
                if max_sim is None or sim > max_sim:
                    max_sim = sim
            except Exception:
                pass
        
        chunk_overlap = _keyword_overlap(query, text)
        
        # A chunk is considered supporting if it passes either threshold
        if chunk_overlap >= RAG_MIN_KEYWORD_OVERLAP or (sim is not None and sim >= RAG_MIN_SIMILARITY):
            supporting_chunks += 1

    sufficient = supporting_chunks >= RAG_MIN_SUPPORTING_CHUNKS
    
    if sufficient:
        confidence = 0.9 if supporting_chunks >= 2 else 0.7
        reason = f"Context sufficient: found {supporting_chunks} supporting chunks exceeding thresholds."
    else:
        confidence = 0.1
        reason = f"Context insufficient: only {supporting_chunks} supporting chunks found (requires {RAG_MIN_SUPPORTING_CHUNKS})."

    return ContextValidation(
        sufficient=sufficient,
        confidence=confidence,
        reason=reason,
        supporting_chunks=supporting_chunks,
        max_similarity=float(max_sim) if max_sim is not None else None,
        keyword_overlap=float(best_overlap)
    )
