# Data Storage & Embedding Strategy

The chatbot uses **three different storage systems** depending on the type of data. Each is chosen for accuracy and retrieval speed.

---

## 1. Student Dataset — SQLite Database

**File:** `app/database/students.db`

Student academic records are stored in **SQL** because:
- Data is **structured and relational** (Roll No, CGPA, Credits, Results, Placements)
- SQL guarantees **100% accuracy** — no hallucinations possible
- Supports fast indexed queries by roll number or name
- Industry-standard for transactional academic data

**Example queries handled by SQL:**
- "What is my CGPA?"
- "How many credits have I completed?"
- "What are my internal marks for OS?"
- "Am I placed?"

**Managed by:** `app/services/sql_system.py`

---

## 2. FAQ Dataset — Vector Embeddings

**Source:** `data/rawdata/faq_rows.json`  
**Stored in:** `data/chunks/unified_vectors.json`

300+ curated Q&A pairs are embedded for **semantic search**. This allows the chatbot to match user questions even when phrased differently from the stored FAQ.

#### Embedding Model
`all-MiniLM-L6-v2` (SentenceTransformers)

#### Technique
- Each FAQ (question + answer) is embedded as a single dense vector (384 dimensions)
- Vectors are normalized for stable cosine similarity scoring
- Stored in FAISS index for fast nearest-neighbor lookup

---

## 3. Web-Scraped College Data — Vector Embeddings

**Source:** `data/scraped_data/outputs/*.json` (~97 files)  
**Stored in:** `data/chunks/unified_vectors.json`

Long-form college website content (departments, facilities, committees, etc.) is chunked and embedded for knowledge retrieval.

#### Embedding Model
`all-MiniLM-L6-v2` (same as FAQ — ensures both live in the same vector space)

#### Technique
- Content split into 500-char chunks with 100-char overlap
- Each chunk embedded to a 384-dim vector
- Retrieved via **hybrid FAISS + BM25** search

---

## Summary

| Data Type | Storage | Why |
|---|---|---|
| Student Records | SQLite (`students.db`) | Structured, accuracy-critical |
| FAQ Pairs | Vector Embeddings (FAISS) | Semantic question matching |
| Website Content | Vector Embeddings (FAISS+BM25) | Hybrid contextual retrieval |

**All embeddings use `all-MiniLM-L6-v2`** — ensuring FAQ and web data share the same vector space for uniform similarity scoring.
