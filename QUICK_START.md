# Quick Start Guide - College Buddy v3.5

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

**Option B: Terminal Interface**
```bash
python terminal_chat.py
```

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

## 📁 What's New in v3.1

### 🔗 Navigation Links (February 2026)
- ✅ Clickable link chips below responses linking to relevant TKRCET website pages
- ✅ Topic keywords mapped to real TKRCET URLs
- ✅ AI-powered scraper (Groq + Playwright) for high-quality link discovery

### 🧠 Intelligence Injection (February 2026)
- ✅ Automated Scraping Pipeline: `tkrcet_scraper.py`
- ✅ Groq LPU Integration: Instant data extraction and structuring
- ✅ Enhanced Metrics: Automated Accuracy/Faithfulness scoring in backend

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

### 🛡️ Reliability Enhancements
- ✅ Strict LLM prompt enforcement
- ✅ Multi-layer scope validation
- ✅ Comprehensive test suite
- ✅ Better error handling

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

**Version**: 3.5.0
**Last Updated**: February 2026 (Scraping & Ingestion Update)
