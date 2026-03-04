# Quick Start Guide - College Buddy v3.6

---

## 🆕 Fresh Clone Setup (For Teammates)

Follow these steps after cloning the repository for the first time:

```powershell
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set up your API key
copy .env.example .env
# Then open .env and fill in: GROQ_API_KEY=your_key_here
# Get a free key at: https://console.groq.com/keys

# 4. Build the vector database (one-time, ~2-5 mins)
python scripts/ingest.py

# 5. Start the server
python backend.py
# → Server runs at http://127.0.0.1:8000
# → Open frontend/index.html in your browser
```

> **No scraping needed!** All 97 scraped data files are already committed to the repo.

---

## 🚀 Already Set Up? Quick Start

```powershell
.venv\Scripts\activate
python backend.py
```
*Then open `frontend/index.html` in your browser.*

---

## ✅ Test the Chatbot

Try these queries to verify everything works:

### Valid College Queries (Should Answer)
- "how r u?"
- "where is college located?"
- "who is the principal?"
- "what are the college timings?"
- "what courses are offered?"
- "tell me about facilities"

### Invalid Queries (Should Reject)
- "(a+b)^2"
- "solve 2+2"
- "what is photosynthesis?"
- "capital of France?"

**Expected Rejection Message:**
```
I'm sorry, I can only answer questions about TKRCET college. 
Please ask me about admissions, courses, facilities, timings, 
faculty, or other college-related topics.
```

---



---

## 🔧 Troubleshooting

### Backend won't start
- Ensure `.env` file exists with a valid `GROQ_API_KEY`
- Run `python scripts/ingest.py` first if vector indices are missing

### Groq API 403 Error
- Your ISP/hotspot may be blocked by Groq
- **Fix:** Enable Cloudflare WARP (free VPN) → [1.1.1.1](https://1.1.1.1/)

### Wrong answers or stale data
```powershell
# Rebuild vector database from existing scraped data
python scripts/ingest.py
python backend.py
```

### Import errors
```powershell
pip install -r requirements.txt --force-reinstall
```

### Ingest fails with missing file
- Ensure `data/scraped_data/outputs/` has JSON files (do a `git pull`)
- Ensure `data/rawdata/faq_rows.json` and `data/knowledge_base.json` exist

---

## 📊 Evaluation & Logs

### Reset the log file (new session / fresh start)
```bash
python tools/refresh_logs.py
```
*Archives the current `logs/response_log.xlsx` with a timestamp and creates a clean one.*

### Run the evaluation suite
```bash
python tests/prompt_test.py
```
*Runs all test prompts, scores them across 8 quality metrics, and saves results incrementally to the Evaluation sheet. You can safely Ctrl+C — results up to that point are not lost.*

### Inspect logs
```bash
python inspect_logs.py
```
*Prints both the Production sheet (live query log) and the Evaluation sheet (test results) to the terminal.*

### Generate the Metrics Reference PDF
```bash
.venv\Scripts\python.exe tools\generate_metrics_pdf.py
```
*Creates/updates `logs/Metrics_Reference.pdf` — a polished guide explaining every metric, its formula, and how to interpret the score.*

### Log sheets explained
| Sheet | Updated by | Key columns |
|---|---|---|
| Production | Every `/query` request | Timestamp, Query, Response, Time Taken (s), Source |
| Evaluation | `prompt_test.py` | Latency (s), Server Time (s), Faithfulness %, Relevance %, Accuracy % + more |

---

## 📲 What's New


### 📋 Logging & Evaluation Refinement — v3.6 (March 2026)
- ✅ Production sheet simplified to 6 focused columns (Timestamp, Query, Response, Time Taken, Session, Source)
- ✅ Evaluation sheet now records both **Latency (s)** (client round-trip) and **Server Time (s)** (pure backend time) side-by-side
- ✅ `prompt_test.py` saves each result **immediately** after scoring — no data loss on interruption
- ✅ New `tools/generate_metrics_pdf.py` generates a polished `logs/Metrics_Reference.pdf`
- ✅ Greetings now bypass the full RAG pipeline for near-instant responses
- ✅ `inspect_logs.py` fixed to display both Production and Evaluation sheets

### 🔗 Navigation Links (February 2026)
- ✅ Clickable link chips below responses linking to relevant TKRCET website pages
- ✅ Topic keywords mapped to real TKRCET URLs
- ✅ AI-powered scraper (Groq + Playwright) for high-quality link discovery

### 🧠 Intelligence Injection (February 2026)
- ✅ Automated Scraping Pipeline: `tkrcet_scraper.py`
- ✅ Groq LPU Integration: Instant data extraction and structuring

### ✨ Scope Validation (January 2026)
- ✅ Filters out non-college queries (math, science, general knowledge)
- ✅ Enhanced greeting detection ("how r u?", "what's up", etc.)
- ✅ Explicit rejection messages for off-topic questions
- ✅ 89% test success rate

### ⚡ Performance Improvements
- ✅ Switched to Llama 3.2:3b
- ✅ Reduced context window (40% speed boost)
- ✅ Optimized prompt engineering
- ✅ Cached indices for instant startup

---

## 📚 Available Commands

Once the chatbot is running:

- `help` - Show available commands
- `clear` - Clear screen
- `status` - Show system status
- `exit` or `quit` - Exit chatbot

---

## 💡 Pro Tips

1. **First run takes longer**: FAISS and BM25 indices are built on first run (~2 min)
2. **Subsequent runs are instant**: Indices are cached on disk
3. **Use specific queries**: More specific questions get better answers
4. **Re-scrape a single page:** `python scripts/rescrape_single.py "<url>"` then re-run `ingest.py`

---

## 📊 System Status

Check if everything is working:

```bash
# In the chatbot, type:
status
```

**Expected Output (at** `http://127.0.0.1:8000/health`**):**
```json
{
  "status": "healthy",
  "router": "ready",
  "sessions_active": 0
}
```

---

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Check `docs/` folder for architecture guides
3. Verify dependencies: `pip list`
4. Check Groq API status: [status.groq.com](https://status.groq.com)

---

## 🎯 Next Steps

1. ✅ Test the chatbot with various queries
2. ✅ Verify scope validation is working
3. ✅ Explore the knowledge base
4. ✅ Customize for your needs (optional)

---

**That's it! Your chatbot is ready to use! 🎉**

**Version**: 3.6.0  
**Last Updated**: March 2026 (Cleanup & Docs Refresh)
