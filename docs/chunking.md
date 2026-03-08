# Data Pipeline — Scraping to Vector DB

This document explains how raw website content flows from scraping all the way into the vector database.

---

## Pipeline Overview

```
tkrcet_links.txt  (130 URLs)
        ↓
tkrcet_scraper.py  (StealthyFetcher + Groq JSON structuring)
        ↓
data/scraped_data/outputs/*.json  (~97 per-URL JSON files)
        ↓
scripts/ingest.py  (chunking + embedding + indexing)
        ↓
data/chunks/unified_vectors.json   ← source of truth
app/database/vectordb/*.index/.pkl ← runtime search indices
```

---

## Stage 1 — Scraping (`tkrcet_scraper.py`)

Each URL from `tkrcet_links.txt` is fetched using **Scrapling's StealthyFetcher** (Playwright-based, bypasses Cloudflare).

The visible page text is then sent to **Groq** (`llama-3.1-8b-instant`) in JSON mode, which structures it into:

```json
{
  "sections": [
    { "title": "Department Overview", "content": "..." },
    { "title": "Vision & Mission",    "content": "..." }
  ]
}
```

Each URL gets its own `.json` file saved to `data/scraped_data/outputs/`.

> **Teammates:** You do NOT need to run the scraper. All output JSONs are committed to Git.  
> To re-scrape a single failed URL: `python scripts/rescrape_single.py "<url>"`

---

## Stage 2 — Ingestion & Chunking (`scripts/ingest.py`)

`ingest.py` reads all source data and builds the vector indices:

### Chunking Strategy
- **Chunk size:** 500 characters
- **Overlap:** 100 characters (sliding window for context continuity)
- **Metadata stored:** `source_url`, `section_title`, `data_type`

### Sources Processed
| Source | Type | Strategy |
|---|---|---|
| `data/scraped_data/outputs/*.json` | Web content | Chunked (500 chars, 100 overlap) |
| `data/rawdata/faq_rows.json` | FAQ pairs | One chunk per Q&A (no split) |
| `data/knowledge_base.json` | Key facts | One chunk per fact |

### Output Files
| File | Description |
|---|---|
| `data/chunks/unified_vectors.json` | All chunks + embeddings (committed to Git) |
| `data/chunks/corpus_ultrarag.jsonl` | UltraRAG-format corpus (auto-generated) |
| `app/database/vectordb/ultrarag_faiss.index` | FAISS dense index (auto-generated) |
| `app/database/vectordb/ultrarag_bm25.pkl` | BM25 sparse index (auto-generated) |

---

## Running Ingestion

```powershell
# Run once after cloning (or after re-scraping)
python scripts/ingest.py
```

> Estimated time: ~2–5 minutes depending on machine.

---

## Re-scraping a Single URL

If a specific page needs to be refreshed:

```powershell
python scripts/rescrape_single.py "https://tkrcet.ac.in/computer-science-and-engineering/"
```

Then re-run `python scripts/ingest.py` to rebuild the indices.