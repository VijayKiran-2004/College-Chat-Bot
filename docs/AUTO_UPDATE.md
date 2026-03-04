# Data Refresh Guide

This document explains how to update the chatbot's knowledge base when the TKRCET website content changes.

---

## Overview

The chatbot's knowledge is stored in two places:

| Source | How to Update |
|---|---|
| Website content (97 pages) | Re-run `tkrcet_scraper.py` |
| Vector database (FAISS+BM25) | Re-run `ingest.py` |

> **Normal operation:** You don't need to refresh anything. The scraped data is committed to Git and teammates get it automatically on `git pull`.

---

## When to Refresh

Refresh the data when:
- The college website has significant new content (new events, new HOD, fee changes, etc.)
- You want to add new FAQ entries to `data/rawdata/faq_rows.json`
- You manually edit the knowledge base (`data/knowledge_base.json`)

---

## Full Refresh (Re-scrape + Rebuild)

```powershell
# Step 1: Activate virtual environment
.venv\Scripts\activate

# Step 2: Re-scrape all 130 URLs (~25-40 mins, requires Groq API)
python scripts/tkrcet_scraper.py

# Step 3: Rebuild the vector database
python scripts/ingest.py

# Step 4: Restart backend
python backend.py
```

---

## Partial Refresh (Single URL)

To refresh just one page (e.g., CSE department page was updated):

```powershell
python scripts/rescrape_single.py "https://tkrcet.ac.in/computer-science-and-engineering/"

# Then rebuild indices
python scripts/ingest.py
```

---

## Updating Just the Vector DB (No Re-scraping)

If you edited `knowledge_base.json` or `faq_rows.json` but didn't re-scrape:

```powershell
python scripts/ingest.py
```

This is fast (~2-5 mins) and rebuilds the FAISS and BM25 indices from existing data.

---

## Troubleshooting

### Groq 403 Error During Scraping
- **Cause:** ISP or mobile hotspot IP blocked by Groq
- **Fix:** Enable a VPN (e.g., Cloudflare WARP — free) then re-run the scraper

### Scraper Gets 403 From Website
- **Cause:** Cloudflare or MalCare firewall
- **Fix:** The scraper automatically falls back to `undetected-chromedriver`. If that also fails, the URL is skipped and saved as `{"sections": []}`.

### Ingest Fails With Missing File Error
- **Cause:** A required source file is missing
- **Fix:** Ensure `data/scraped_data/outputs/`, `data/rawdata/faq_rows.json`, and `data/knowledge_base.json` all exist before running `ingest.py`

---

## Logs

Runtime query logs are stored in:
```
logs/response_log.xlsx      ← Current session log
logs/response_log_*.xlsx    ← Archived logs (timestamped)
```

To reset logs (archive current and start fresh):
```powershell
python tools/refresh_logs.py
```
