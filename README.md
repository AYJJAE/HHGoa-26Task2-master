# HH Goa 2026 — Task 2: Multilingual Voice-Enabled RAG

A production-grade, voice-first Retrieval-Augmented Generation (RAG) system built for **Hacker House Goa 2026**. Features hybrid dense-sparse vector search, calibrated soft-gating relevance verification, grounded multi-turn response generation, and sub-second primary/fallback Speech-to-Text streaming.

---

## System Architecture

```
User Query (Voice / Text)
       │
       ├──► Voice Pipeline (ElevenLabs Primary STT / Sarvam Fallback)
       │
       ▼
Query Router & Language Classifier (Indic Morpheme & Script Detection: EN, HI, MR)
       │
       ▼
Hybrid Search Pipeline (BGE-M3 Dense Embedding + Qdrant Sparse BM25 + Reciprocal Rank Fusion)
       │
       ▼
Context Gating & Relevance Scoring (Calibrated Multi-Signal Attribute Alignment)
       │
       ▼
Answer Generation (Grounded Gemini 2.5 Flash with Dynamic Language Alignment)
       │
       ▼
Grounding & Semantic Verification (Strict Factual Verification & Hallucination Prevention)
       │
       ▼
Structured Output (Answer, Sources, Confidence, Latency Percentiles: P50/P70/P100)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Lightweight service health verification |
| `GET` | `/health/ready` | Readiness check (validates FAISS & Sparse vector counts) |
| `POST` | `/api/retrieve` | Isolated hybrid retrieval candidate extraction |
| `POST` | `/api/ask` | Full text RAG pipeline with grounding verification |
| `POST` | `/api/voice_ask` | End-to-end voice-in / grounded answer response |
| `POST` | `/api/transcribe` | Dedicated low-latency STT endpoint |

---

## Local Evaluation

To evaluate this repository against the official **`rag-local-eval-loop`** evaluation harness:

### 1. Start backend
```bash
cd backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

### 2. Ensure `/api/ask` is reachable
```bash
curl -X POST http://127.0.0.1:8080/api/ask \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the capital of India?\"}"
```

### 3. Ensure `/api/retrieve` is reachable
```bash
curl -X POST http://127.0.0.1:8080/api/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the capital of India?\", \"top_k\": 5}"
```

### 4. Configure the evaluator's HTTP target
Use the provided target configuration file at [`eval/target_config.json`](eval/target_config.json) or point your HTTP runner to `http://127.0.0.1:8080`.

### 5. Set the evaluator's own API keys (local environment only)
```bash
export GEMINI_API_KEY="your-gemini-api-key"
# or if using external evaluators:
export OPENAI_API_KEY="your-evaluator-key"
```

### 6. Run smoke test (3 answerable + 3 unanswerable queries)
```bash
python eval/smoke_test.py
```

### 7. Run full evaluation (50 answerable + 50 unanswerable queries)
```bash
python eval/run_eval.py http://127.0.0.1:8080 50 50
```

The evaluator will calculate and report the 5 official dimensions:
- **Retrieval**: Recall@1, Recall@3, Recall@5, MRR
- **Faithfulness**: Hallucination rate against retrieved context
- **Correctness**: Semantic accuracy against gold MSMARCO-XI references
- **Reliability**: Unanswerable refusal rate and fabrication rate
- **Latency**: P50, P70, P100 from real requests

---

## Security & Secrets
- Never commit `.env` files or API keys.
- All evaluation credentials and benchmark artifacts remain strictly local.
