# HH Goa 2026 — Task 2: Multilingual Voice-Enabled RAG

A production-grade, voice-first Retrieval-Augmented Generation (RAG) system built for **Hacker House Goa 2026 (Task 2)** by **Team Horizon Labs**. Features hybrid dense-sparse retrieval, calibrated relevance gating, grounded multi-turn response generation, and sub-second primary/fallback Speech-to-Text streaming.

---

## Key Features

- 🎙️ **Voice-Enabled Multilingual RAG:** Real-time microphone audio capture with acoustic script & morpheme classification (English, Hindi, Marathi).
- ⚡ **Dual-Tier STT Streaming:** Primary ElevenLabs Scribe v2 streaming with instant automatic fallback to Sarvam AI Saaras v3.
- 🔍 **Hybrid Dense + Sparse Search:** Dense BGE-M3 semantic embeddings combined with Sparse BM25 lexical search fused via Reciprocal Rank Fusion (RRF).
- 🧩 **Multi-Strategy Chunking:** Sentence-level, paragraph, fixed-token, and semantic boundary chunking tailored to Indic and English corpora.
- 🛡️ **Grounded Hallucination Guardrails:** Two-stage calibrated verification (Pre-generation Soft Gating + Post-generation Semantic Verification).
- 🚫 **Reliable Refusal Policy:** Immediately refuses unanswerable, out-of-domain, or speculative queries without fabricating facts.
- 📊 **Real Latency Percentiles:** Complete instrumentation capturing real P50, P70, and P100 metrics across embedding, search, generation, and total end-to-end stages.
- 🧪 **Official Evaluation Suite:** Ready for `rag-local-eval-loop` with decoupled retrieval candidate inspection and standardized HTTP adapters.

---

## Architecture

### Voice Query Pipeline
```
User Audio (Microphone)
       │
       ├──► ElevenLabs Primary STT (Fallback: Sarvam AI Saaras v3)
       │
       ▼
Query Router & Intent Classifier (Language, Intent, Complexity)
       │
       ▼
Hybrid Search (Dense FAISS Vector Search + Sparse BM25 Index)
       │
       ▼
Reciprocal Rank Fusion (RRF Score Aggregation)
       │
       ▼
Context Gating & Relevance Scoring (Attribute Alignment & Conflict Detection)
       │
       ▼
Grounded Generation (Gemini 2.5 Flash with Enforced Target Language Parity)
       │
       ▼
Semantic Verifier & Grounding Gate (Hallucination Prevention)
       │
       ▼
Grounded Response (Answer, Sources, Confidence, Latency Percentiles: P50/P70/P100)
```

### Text Query Pipeline
```
Text Query ──► Query Router ──► Hybrid Search ──► Context Gate ──► Generation ──► Grounding ──► Answer
```

---

## Project Structure

```text
├── backend/
│   ├── api/                    # FastAPI routes (/api/ask, /api/retrieve, /api/voice_ask, STT)
│   ├── app/                    # Configuration, retriever wrappers, and latency benchmarks
│   ├── benchmark/              # Calibration, chunking, and correctness test suites
│   ├── data/                   # MSMARCO-XI dataset and schema definitions
│   ├── pipeline/               # Core RAG: embeddings, FAISS vector store, routing, generation, grounding
│   ├── tests/                  # Pytest unit, integration, STT, and CORS test suites
│   ├── Dockerfile              # Container definition for Railway/Cloud deployments
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── public/                 # Static assets, logos, and fonts
│   ├── src/
│   │   └── app/
│   │       ├── components/     # UI components (AnswerCard, TelemetryPanel, VoiceRecorder, etc.)
│   │       ├── goa-assistant/  # Goa Companion island exploration page
│   │       ├── rag/            # Main Multilingual Voice RAG workspace
│   │       └── page.tsx        # Landing page
│   ├── package.json            # Next.js dependencies
│   └── next.config.ts          # Turbopack & asset configuration
│
├── eval/
│   ├── target_config.json      # Official rag-local-eval-loop HTTP target configuration
│   ├── http_target.py          # HTTP adapter for the live service
│   ├── smoke_test.py           # 3 answerable + 3 unanswerable query smoke test
│   └── run_eval.py             # 50+50 query local evaluation harness
│
├── .env.example                # Clean environment variables template
├── .gitignore                  # Git repository exclusion rules
├── docker-compose.yml          # Local multi-service orchestration
└── README.md                   # Project documentation
```

---

## Local Setup

### 1. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Fill in your API keys in backend/.env

# Run FastAPI server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080
```

### 2. Frontend Setup

```bash
cd frontend

# Install Node packages
npm install

# Run Next.js development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Environment Variables

Only configure variable names; never commit actual secrets to version control.

| Variable | Description | Default / Example |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini LLM API Key | Required |
| `GEMINI_MODEL` | Target Gemini Generation Model | `gemini-2.5-flash` |
| `ELEVENLABS_API_KEY` | ElevenLabs STT API Key | Required |
| `ELEVENLABS_STT_MODEL` | ElevenLabs Scribe Model | `scribe_v2` |
| `SARVAM_API_KEY` | Sarvam AI Fallback STT API Key | Required |
| `SARVAM_STT_MODEL` | Sarvam Saaras Model | `saaras:v3` |
| `GROUNDING_THRESHOLD` | Minimum Grounding Confidence | `0.55` |
| `RELEVANCE_MIN_SCORE` | Pre-generation Soft Gating Threshold | `0.35` |
| `PORT` | Backend Port Binding | `8080` |
| `ALLOWED_ORIGINS` | CORS Allowed Origins | `*` |
| `NEXT_PUBLIC_API_URL` | Frontend Backend Base URL | `http://127.0.0.1:8080` |

---

## Running Benchmarks & Tests

### 1. Correctness & Grounding Tests
```bash
python -m pytest backend/benchmark/test_correctness.py
```

### 2. Comprehensive Backend Test Suite
```bash
python -m pytest backend/tests/
```

### 3. Latency Benchmark (P50 / P70 / P100)
```bash
python -m app.benchmark 50
```

### 4. Frontend Production Build Check
```bash
cd frontend && npm run build
```

---

## Official Evaluation Loop (`rag-local-eval-loop`)

The service is fully configured for evaluation in HTTP service mode:

1. **Start backend:** `python -m uvicorn api.main:app --host 127.0.0.1 --port 8080`
2. **Verify readiness:** `curl http://127.0.0.1:8080/health/ready`
3. **Run Smoke Test (3 answerable + 3 unanswerable queries):**
   ```bash
   python eval/smoke_test.py
   ```
4. **Run Full Evaluation Loop (50 answerable + 50 unanswerable queries):**
   ```bash
   python eval/run_eval.py http://127.0.0.1:8080 50 50
   ```

---

## Deployment

- **Frontend:** Deployed on **Vercel** (`Next.js 16.3 Turbopack`).
- **Backend:** Deployed on **Railway** with resident FAISS vector indexing and heavy asset streaming.

---

## License

This project is licensed under the terms included in [LICENSE.txt](LICENSE.txt).
