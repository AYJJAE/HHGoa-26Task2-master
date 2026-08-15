import re
from typing import Dict, Any

# Lazy-load lightweight language detector for accurate Hindi vs Marathi detection
_langdetect_fn = None

def _get_langdetect():
    global _langdetect_fn
    if _langdetect_fn is None:
        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0
            def _detect_wrapper(text: str):
                langs = detect_langs(text)
                if langs:
                    return {"lang": langs[0].lang, "score": langs[0].prob}
                return {"lang": "hi", "score": 0.0}
            _langdetect_fn = _detect_wrapper
            print("Loaded langdetect language detector.")
        except ImportError:
            try:
                from ftlangdetect import detect as ft_detect
                _langdetect_fn = lambda text: ft_detect(text, low_memory=True)
                print("Loaded fasttext language detector.")
            except ImportError:
                print("WARNING: Neither langdetect nor ftlangdetect installed. Falling back to regex-based language detection.")
                print("         Install with: pip install langdetect")
                _langdetect_fn = None
    return _langdetect_fn


class QueryRouter:
    """
    Analyzes the query and determines:
    - Language (English, Hindi, Marathi, Code-mixed)
    - Intent (Factual, Entity-heavy, Comparative, Multi-hop, Chitchat)
    - Complexity (Simple, Moderate, Complex)
    - Routing Strategy (Weightings for Dense vs Sparse, Chunking Strategy Preference)
    """
    
    # Compiled patterns for entity recognition
    # Dates: "15 August", "2024", "January 5th", etc.
    DATE_PATTERN = re.compile(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?)\b',
        re.IGNORECASE
    )
    # Numbers with units: "500 km", "1.5 million", "$200", etc.
    NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\s*(?:km|kg|m|cm|million|billion|crore|lakh|%|rupees|dollars)?\b', re.IGNORECASE)
    # Common location indicators
    LOCATION_KEYWORDS = {
        "city", "country", "state", "capital", "district", "province", "region",
        "शहर", "देश", "राज्य", "राजधानी", "जिला", "प्रांत",
    }
    # Question words that suggest factual queries
    FACTUAL_STARTERS = re.compile(
        r'\b(?:what|who|when|where|which|how many|how much|क्या|कौन|कब|कहाँ|कहां|कितने|कितना|काय|कोण|केव्हा|कुठे)\b',
        re.IGNORECASE
    )
    
    def __init__(self):
        # Basic Devanagari range for Hindi/Marathi detection (fallback)
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.latin_pattern = re.compile(r'[a-zA-Z]+')
        
        # Pre-warm the langdetect model
        ft = _get_langdetect()
        if ft:
            try:
                # Warm up with a dummy call
                ft("test")
            except Exception:
                pass
        
    def detect_language(self, query: str) -> str:
        has_devanagari = bool(self.devanagari_pattern.search(query))
        has_latin = bool(self.latin_pattern.search(query))
        
        # Code-mixed check first
        if has_devanagari and has_latin:
            return "Code-mixed"
        
        if has_devanagari:
            # Use model-based detection for accurate Hindi vs Marathi distinction
            ft = _get_langdetect()
            if ft:
                try:
                    result = ft(query)
                    lang_code = result.get("lang", "hi")
                    confidence = result.get("score", 0.0)
                    
                    if lang_code == "mr" and confidence > 0.3:
                        return "Marathi"
                    elif lang_code == "hi" and confidence > 0.3:
                        return "Hindi"
                    elif lang_code in ("sa", "ne", "bh"):
                        # Related Devanagari-script languages — treat as Hindi for retrieval
                        return "Hindi"
                    else:
                        # fasttext returned something unexpected; use heuristic
                        return self._devanagari_heuristic(query)
                except Exception as e:
                    print(f"fasttext langdetect error: {e}")
                    return self._devanagari_heuristic(query)
            else:
                return self._devanagari_heuristic(query)
        
        return "English"
    
    def _devanagari_heuristic(self, query: str) -> str:
        """Fallback regex heuristic for Hindi vs Marathi when fasttext is unavailable."""
        # Marathi-specific morphological markers
        marathi_markers = ["आहे", "काय", "नाही", "होते", "आहेत", "मध्ये", "साठी", "केले", "झाले", "कोण"]
        for marker in marathi_markers:
            if marker in query:
                return "Marathi"
        return "Hindi"
        
    def detect_intent(self, query: str) -> str:
        query_lower = query.lower().strip()
        
        # Chitchat / Greeting detection — match whole words only
        chitchat_starters = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhow are you\b", 
            r"\bhow's it going\b", r"\bgood morning\b", r"\bgood evening\b",
            r"\bgood night\b", r"\bwhat's up\b", r"\bthanks\b", r"\bthank you\b",
            r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b",
            r"\bनमस्ते\b", r"\bनमस्कार\b", r"\bकैसे हो\b", r"\bधन्यवाद\b"
        ]
        # Only classify as chitchat if the query is SHORT and matches a greeting
        if len(query_lower.split()) <= 5:
            for pattern in chitchat_starters:
                if re.search(pattern, query_lower):
                    return "Chitchat"
        
        # Comparative detection
        comparative_keywords = ["vs", "difference", "compare", "better", "worse", "versus",
                                "तुलना", "अंतर", "विरुद्ध", "बेहतर", "फर्क"]
        if any(kw in query_lower for kw in comparative_keywords):
            return "Comparative"
            
        # Multi-hop detection (requires chaining facts)
        multihop_keywords = ["because", "after", "before", "result of", "led to", "caused by",
                             "कारण", "नंतर", "पहले", "बाद में", "के कारण", "फलस्वरूप"]
        if any(kw in query_lower for kw in multihop_keywords):
            return "Multi-hop"
        
        # --- Enhanced Entity Recognition ---
        entity_score = 0
        
        # 1. Capitalized words (proper nouns) in non-Devanagari text
        words = query.split()
        capitalized = [w for w in words if w[0:1].isupper() and len(w) > 1 and w not in ("What", "Who", "When", "Where", "Which", "How", "Is", "Are", "The", "A", "An")]
        entity_score += len(capitalized) * 2
        
        # 2. Date entities
        if self.DATE_PATTERN.search(query):
            entity_score += 3
        
        # 3. Quoted entities (e.g., "Taj Mahal")
        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        entity_score += len(quoted) * 3
        
        # 4. Location keywords
        if any(kw in query_lower for kw in self.LOCATION_KEYWORDS):
            entity_score += 2
        
        # 5. Devanagari proper noun heuristic — tokens that are standalone (likely names)
        devanagari_tokens = re.findall(r'[\u0900-\u097F]+', query)
        if len(devanagari_tokens) <= 3 and len(devanagari_tokens) > 0 and not self.FACTUAL_STARTERS.search(query_lower):
            entity_score += 2
        
        # Threshold-based entity classification
        if entity_score >= 4:
            return "Entity-heavy"
        
        # Default to Factual for question-like queries
        return "Factual"
        
    def detect_complexity(self, query: str) -> str:
        word_count = len(query.split())
        if word_count < 6:
            return "Simple"
        elif word_count < 15:
            return "Moderate"
        else:
            return "Complex"
            
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for routing logic.
        """
        lang = self.detect_language(query)
        intent = self.detect_intent(query)
        complexity = self.detect_complexity(query)
        
        # Determine retrieval strategy based on analysis
        strategy = {
            "dense_weight": 0.5,
            "sparse_weight": 0.5,
            "preferred_chunk_strategy": "sentence", # default
            "top_k_retrieve": 10, # hybrid retrieve candidate pool
            "top_k_rerank": 0     # final context size (0 = disable cross-encoder reranker)
        }
        
        if intent == "Entity-heavy":
            strategy["dense_weight"] = 0.6
            strategy["sparse_weight"] = 0.4 # Favor dense semantic matching for entity distinction
            strategy["preferred_chunk_strategy"] = "passage" # Needs more context for entities
            
        elif intent == "Comparative" or intent == "Multi-hop" or complexity == "Complex":
            strategy["dense_weight"] = 0.7
            strategy["sparse_weight"] = 0.3 # Favor semantic similarity for complex reasoning
            strategy["preferred_chunk_strategy"] = "parent" # Needs full passage or Parent-Child
            strategy["top_k_retrieve"] = 20 # Cast a wider net
            strategy["top_k_rerank"] = 0
            
        elif lang == "Code-mixed":
            # Code-mixed often struggles with dense semantic models unless they are specifically trained on code-mixed
            # So we balance or slightly favor sparse if words are phonetic
            strategy["dense_weight"] = 0.5
            strategy["sparse_weight"] = 0.5
            
        return {
            "language": lang,
            "intent": intent,
            "complexity": complexity,
            "strategy": strategy
        }

if __name__ == "__main__":
    router = QueryRouter()
    print(router.route_query("What is the capital of India?"))
    print(router.route_query("एमएस धोनी vs विराट कोहली in stats?"))
    print(router.route_query("भारत की राजधानी क्या है?"))
    print(router.route_query("महाराष्ट्राची राजधानी काय आहे?"))  # Marathi test
    print(router.route_query("hello"))
    print(router.route_query("Taj Mahal was built in 1632"))
