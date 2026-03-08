# College Buddy RAG System

This document describes the Retrieval-Augmented Generation (RAG) pipeline powering the College Buddy chatbot. The system retrieves accurate, up-to-date information from the TKRCET knowledge base and generates natural responses using a local Ollama LLM.

---

## Overview

The chatbot combines three systems:

| System | Role |
|---|---|
| **UltraRAG** | Hybrid FAISS+BM25 vector retrieval |
| **Knowledge Base** | Direct lookup for facts, FAQs, and key personnel |
| **SQL System** | Student-specific data (grades, attendance, results) |

A **Query Router** (`query_router.py`) decides which system(s) to use based on the query intent.

---

## RAG Pipeline (UltraRAG)

### 1. Query Routing

`query_router.py` classifies the incoming query:

- **`student_info`** → SQL system (attendance, CGPA, results)
- **`knowledge_base`** → Direct FAQ lookup (NAAC grade, key personnel, etc.)
- **`rag`** → UltraRAG hybrid retrieval (general college questions)

### 2. Retrieval (`ultra_rag.py`)

A **hybrid search** is performed in parallel:

- **Dense (FAISS):** Query is embedded using `all-MiniLM-L6-v2` → top-K nearest neighbors found in the FAISS index
- **Sparse (BM25):** Keyword-based search using the BM25 index (`.pkl` file)
- Results are merged and deduplicated

### 3. Reranking

A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores every candidate chunk against the original query. This separates genuinely relevant chunks from superficially similar ones.

### 4. Generation (`generator.py`)

The top reranked chunks are injected into a structured prompt and sent to **Ollama** (`llama3.2:3b`) for generation.

- Temperature: `0` (deterministic, factual answers)
- Context window: up to 15,000 chars of retrieved text
- Response format: plain conversational text

---

## Response Chain (`chain.py`)

```
User Query
    ↓
Intent Detector  ←─ greeting / out-of-scope fast-path
    ↓
Query Router
    ├── student_info  →  SQL System  →  Generator
    ├── knowledge_base  →  KB Lookup  →  Generator
    └── rag  →  UltraRAG (retrieve → rerank)  →  Generator
                                                      ↓
                                                  Response
```

---

## Key Files

| File | Purpose |
|---|---|
| `app/services/ultra_rag.py` | Hybrid FAISS+BM25 retrieval + reranking |
| `app/services/retriever.py` | ChromaDB fallback retriever |
| `app/services/generator.py` | Groq LLM prompt builder and response generator |
| `app/services/query_router.py` | Routes queries to correct system |
| `app/services/intent_detector.py` | Detects greetings, off-topic queries |
| `app/services/chain.py` | Orchestrates the full pipeline |
| `app/services/knowledge_base.py` | Direct fact lookup (NAAC, HODs, etc.) |
| `app/services/sql_system.py` | Student data queries via SQLite |

---

## Technologies

| Component | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Dense Index | FAISS (`faiss-cpu`) |
| Sparse Index | BM25 (`rank_bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Ollama → `llama3.2:3b` (Groq used for Scraping) |
| Student DB | SQLite (`students.db`) |
| API Server | FastAPI + Uvicorn |

---

## Running the System

```powershell
# One-time: build vector indices
python scripts/ingest.py

# Start the backend server
python backend.py
# → Server runs at http://127.0.0.1:8000

# Chat in terminal
python terminal_chat.py
```
