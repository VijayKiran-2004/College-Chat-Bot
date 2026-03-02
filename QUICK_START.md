# Quick Start Guide - College Buddy v3.6

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Activate virtual environment (if not already active)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### Step 2: Start Ollama

```bash
# Pull all models (first time only)
ollama pull llama3.2:3b
ollama pull gemma3:1b

# Ollama should auto-start, verify with:
ollama ps
```

### Step 3: Run the Chatbot

**Option A: Web Interface (Recommended)**
```bash
python backend.py
```
*Then open `frontend/index.html` in your browser. Server runs at `http://127.0.0.1:8000`*

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

### Chatbot won't start
```bash
# Check if Ollama is running
ollama ps

# If not, start it
ollama serve
```

### Wrong answers or off-topic responses
```bash
# Re-run ingestion to refresh database
python scripts/ingest.py

# Restart chatbot
python backend.py
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Out of memory
- Close other applications
- The llama3.2:3b model uses ~2GB RAM
- Consider using a smaller model if needed

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

1. **First run takes longer**: FAISS and BM25 indices are built on first run
2. **Subsequent runs are instant**: Indices are cached
3. **Keep Ollama running**: Chatbot needs Ollama service active
5. **Use specific queries**: More specific questions get better answers

---

## 📊 System Status

Check if everything is working:

```bash
# In the chatbot, type:
status
```

**Expected Output:**
```
✓ Knowledge Base: 2029 documents loaded
✓ Embedding Model: all-MiniLM-L6-v2
✓ LLM Model: Llama 3.2:3b (via Ollama)
✓ Knowledge Base: 2029 documents loaded
✓ Embedding Model: all-MiniLM-L6-v2
✓ LLM Model: Llama 3.2:3b (via Ollama)
✓ Retrieval: ChromaDB (Vector Search)
✓ Navigation Links: 28 topic mappings active
✓ System: UltraRAG v3.1
```

---

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Check Ollama status: `ollama ps`
4. Verify dependencies: `pip list`

---

## 🎯 Next Steps

1. ✅ Test the chatbot with various queries
2. ✅ Verify scope validation is working
3. ✅ Explore the knowledge base
4. ✅ Customize for your needs (optional)

---

**That's it! Your chatbot is ready to use! 🎉**

**Version**: 3.6.0  
**Last Updated**: March 2026 (Logging & Evaluation Refinement)
