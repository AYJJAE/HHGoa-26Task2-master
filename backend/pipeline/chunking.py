import re
import tiktoken
from typing import List, Dict, Any, Optional

# Preload tiktoken encoding for token-based chunking
_enc = tiktoken.get_encoding("cl100k_base")

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences, supporting both English and Indic (e.g. Hindi/Marathi) punctuation."""
    # Split on standard punctuation (. ! ?) or the Devanagari danda (।)
    # Keep the delimiter by using capturing group and zipping
    parts = re.split(r'([.!?।])', text)
    sentences = []
    current_sentence = ""
    for i in range(0, len(parts)):
        if parts[i] in ['.', '!', '?', '।']:
            current_sentence += parts[i]
            sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += parts[i]
    if current_sentence.strip():
         sentences.append(current_sentence.strip())
    return [s for s in sentences if s]

class ChunkingPipeline:
    def __init__(self, token_chunk_size: int = 256, token_overlap: int = 50):
        self.token_chunk_size = token_chunk_size
        self.token_overlap = token_overlap

    def process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes a raw MSMARCO-XI / IndicMSMARCO record and returns a list of chunks across multiple strategies.
        Language-aware metadata is injected into every chunk.
        
        Supports both flat format (record['passage']) and nested MSMARCO-XI format
        (record['passages']['English_passages'] / record['passages']['Translated_passages']).
        """
        query_id = str(record.get('query_id', ''))
        target_lang = record.get('target_lang', record.get('language', 'en'))
        
        # ---- Extract passages from the MSMARCO-XI nested schema ----
        passages_to_index: List[Dict[str, Any]] = []  # list of {"text": ..., "lang": ..., "is_gold": ...}
        
        passages_obj = record.get('passages', {})
        if passages_obj:
            eng_passages = passages_obj.get('English_passages', [])
            trans_passages = passages_obj.get('Translated_passages', [])
            is_selected = passages_obj.get('is_selected', [])
            
            for idx, p in enumerate(eng_passages):
                if p and p.strip():
                    passages_to_index.append({
                        "text": p.strip(),
                        "lang": "en",
                        "is_gold": bool(is_selected[idx]) if idx < len(is_selected) else False
                    })
            for idx, p in enumerate(trans_passages):
                if p and p.strip():
                    passages_to_index.append({
                        "text": p.strip(),
                        "lang": target_lang,
                        "is_gold": bool(is_selected[idx]) if idx < len(is_selected) else False
                    })
        
        # Fallback: flat 'passage' key (for API-fetched or simplified records)
        flat_passage = record.get('passage', '')
        if not passages_to_index and flat_passage and flat_passage.strip():
            passages_to_index.append({
                "text": flat_passage.strip(),
                "lang": target_lang,
                "is_gold": record.get('is_selected', False)
            })
        
        if not passages_to_index:
            print(f"WARNING: No passages found for record {query_id}. Skipping.")
            return []
        
        chunks = []
        
        for p_idx, passage_info in enumerate(passages_to_index):
            passage_text = passage_info["text"]
            passage_lang = passage_info["lang"]
            is_gold = passage_info["is_gold"]
            source_passage_id = f"passage_{query_id}_{p_idx}"
            
            # Base Metadata
            base_metadata = {
                "document_id": source_passage_id,
                "source_passage_id": source_passage_id,
                "query_id": query_id,
                "language": passage_lang,
                "is_gold_passage": is_gold,
                "source": "MSMARCO-XI"
            }
            
            # STRATEGY A: Passage-Aware Chunking (The entire passage)
            chunks.append({
                "chunk_id": f"{source_passage_id}_passage",
                "text": passage_text,
                "chunk_strategy": "passage",
                "metadata": {**base_metadata, "chunk_id": f"{source_passage_id}_passage"}
            })
            
            # STRATEGY B: Sentence-Aware Chunking
            sentences = split_into_sentences(passage_text)
            for j, sentence in enumerate(sentences):
                if len(sentence) > 10:
                    chunk_id = f"{source_passage_id}_sent_{j}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": sentence,
                        "chunk_strategy": "sentence",
                        "metadata": {**base_metadata, "chunk_id": chunk_id, "sentence_index": j}
                    })
                    
            # STRATEGY C: Token-Based Chunking (with overlap)
            tokens = _enc.encode(passage_text)
            token_chunks = []
            start = 0
            while start < len(tokens):
                end = start + self.token_chunk_size
                chunk_tokens = tokens[start:end]
                token_chunks.append(_enc.decode(chunk_tokens))
                start += (self.token_chunk_size - self.token_overlap)
                
            for j, t_chunk in enumerate(token_chunks):
                chunk_id = f"{source_passage_id}_token_{j}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": t_chunk,
                    "chunk_strategy": "token",
                    "metadata": {**base_metadata, "chunk_id": chunk_id, "token_index": j}
                })

        return chunks

if __name__ == "__main__":
    # Test script locally
    sample_record = {
        "query_id": "123",
        "target_lang": "hi",
        "query": "भारत की राजधानी क्या है?",
        "Answer": "भारत की राजधानी नई दिल्ली है।",
        "passages": {
            "Translated_passages": [
                "नई दिल्ली भारत की राजधानी है। यह यमुना नदी के किनारे स्थित है। यहाँ कई ऐतिहासिक इमारतें हैं।",
                "मुंबई भारत की आर्थिक राजधानी है। यह महाराष्ट्र राज्य में है।"
            ],
            "is_selected": [1, 0]
        }
    }
    
    pipeline = ChunkingPipeline(token_chunk_size=10, token_overlap=2)
    result_chunks = pipeline.process_record(sample_record)
    
    for c in result_chunks:
        print(f"Strategy: {c['chunk_strategy']:<15} ID: {c['chunk_id']:<20} Length: {len(c['text'])} chars")
        # print(c['text'])
        # print("---")
