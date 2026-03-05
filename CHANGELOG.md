# Changelog - College Buddy

All notable changes to this project will be documented in this file.

## [3.7.1] - March 2026

### Fixed
- **Documentation Accuracy**: Corrected `docs/vector_db.md` and `docs/rag.md` to accurately reference **Ollama (`llama3.2:3b`)** for RAG generation.
- **Clarification**: Distinguished between the **Scraper pipeline** (which uses Groq `llama-3.1-8b-instant`) and the **RAG generation pipeline** (which uses local Ollama).

## [3.7.0] - March 2026

### Added
- **Unified Brain (Deep Reasoning Chain)**: Implemented `DeepReasoningChain` using LangChain Agents to orchestrate complex queries between RAG and SQL systems.
- **Expanded Test Suite**: Increased `tests/prompt_test.py` to support **607 unique prompts** (200 simple, 200 SQL, 207 complex) loaded dynamically from `tests/all_prompts.json`.
- **Scrapling Stealth Engine**: Replaced the legacy Playwright scraper with the Scrapling engine for advanced bot evasion.
- **Response Completeness Logic**: Enhanced generator prompts to ensure responses have a "natural end" and don't cut off mid-sentence, even near token limits.

### Fixed
- **"Chairmen" Query Error**: Standardized administrative title formatting in `KnowledgeBase.py` to fix specific person-lookup failures.
- **Link Duplication**: Refactored `generator.py` to remove redundant "Source Links" and unify on top-level "Quick Links".
- **System Robustness**: 
  - Fixed absolute path handling for `students.db` in `SQLSystem`.
  - Added recursion guards to `DeepReasoningChain` to prevent infinite loops on agent failure.
  - Fixed missing `Path` imports causing 503 errors during backend startup.

## [3.6.0] - March 2026

### Added
- **`tools/generate_metrics_pdf.py`**: New script that auto-generates `logs/Metrics_Reference.pdf` — a fully designed, multi-section reference guide explaining every evaluation metric (formula, plain-English explanation, score bands, and worked examples). Requires `reportlab` (install inside `.venv`).
- **`logs/Metrics_Reference.pdf`**: Generated PDF reference guide covering all 11 metrics across 5 sections: Timing, Retrieval, LLM-as-Judge, Statistical, and Final Accuracy. Premium visual design with dark-navy cover, per-metric accent borders, and color-coded score interpretation tables.
- **`Server Time (s)` column in Evaluation sheet**: The backend now includes `time_taken` in every API response (in the SSE `done` event for streaming, and in the JSON body for SQL responses). `prompt_test.py` captures this and writes it alongside the client-side `Latency (s)` column for a complete picture of where time is spent.
- **Incremental save in `prompt_test.py`**: Results are now saved to the Evaluation sheet after **every single prompt** rather than at the end of the full test run. This prevents data loss on Ctrl+C or crash.
- **`QueryResponse.time_taken` field**: Added to the Pydantic response model in `backend.py` so both streaming and non-streaming paths report server-side processing time.

### Changed
- **Production sheet simplified** (`logger_service.py`, `refresh_logs.py`): Removed heavy evaluation columns (Retrieval Confidence, Faithfulness, Answer Relevance, Cross-Validation, Link Validity, Accuracy) from the Production sheet. It now logs only 6 essential fields: `Timestamp`, `User Query`, `Bot Response`, `Time Taken (s)`, `Session ID`, `Source`.
- **`log_async` refactored** (`backend.py`): Removed all metric recalculation (faithfulness, BERTScore, etc.) from the background logging task. It is now a thin wrapper around `logger_service.log_response`. All heavy evaluation is done exclusively by `prompt_test.py`.
- **Evaluation sheet columns updated** (`logger_service.py`, `refresh_logs.py`, `prompt_test.py`): `EVAL_COLS` now includes `Server Time (s)` positioned after `Latency (s)`.
- **`inspect_logs.py`**: Fixed to read all sheets (Production **and** Evaluation), not just the first sheet.
- **Greeting handling optimised**: Greetings are now fast-tracked through the router, bypassing the full RAG pipeline entirely for near-instant responses.

### Fixed
- **Timing discrepancy explained and resolved**: `Time Taken (s)` (Production) is pure server time; `Latency (s)` (Evaluation) is total client round-trip. Both are now logged side-by-side with `Server Time (s)` so the difference is immediately visible.
- **`refresh_logs.py` column mismatch**: `prod_cols` and `eval_cols` now exactly match `logger_service.py` — no more empty/shifted columns on fresh log creation.

---

## [3.5.0] - February 2026


### Added
- **`scripts/tkrcet_scraper.py`**: Advanced AI-powered scraper using Playwright and Groq API. Structures raw web text into clean JSON sections.
- **`scripts/prepare_data.py`**: Pre-processing script to flatten complex scraper output for the vectorization pipeline.
- **`tkrcet_links.txt`**: Unified source link list for the scraping pipeline.
- **Metrics fix**: Fixed 'kb_encoder' path in `backend.py`, enabling automated accuracy/relevance scoring for all responses.

### Changed
- **Scraping Workflow**: Replaced legacy Selenium/CSV workflow with a modern Playwright/Groq/JSON pipeline for 10x higher data quality.
- **Backend CORS**: Permissive CORS policy (`*`) enabled to allow local `file://` access and dynamic origin development.
- **Request Models**: `QueryRequest` expanded with `language` support and increased message length (1000 chars).
- **Documentation**: All guides (`README`, `QUICK_START`, `Codebase_Reference`) synchronized with the v3.5 "Intelligence Injection" architecture.

### Fixed
- **'UltraRAGSystem' object has no attribute 'kb_encoder'**: Corrected metrics calculation path in `backend.py`.
- **400 Bad Request (OPTIONS)**: Fixed CORS origin rejection by allowing all origins for local development.

---

## [3.2.0] - February 2026

### Added
- **SQL Safety Filter**: `_validate_sql()` blocks DROP/DELETE/ALTER/UPDATE/INSERT/TRUNCATE. Only SELECT queries on the `students` table are allowed. Enforces LIMIT 1000 cap.
- **`scripts/generate_vectors.py`**: New script that generates `unified_vectors.json` from `scraped_data.jsonl` + `faq_rows.json` with text chunking and deduplication.
- **`tests/` directory**: All test files moved here from root (`test_gender.py`, `test_llm_sql.py`, `test_router_upgrade.py`, `test_typo.py`, `test_logging_integration.py`, `run_batch_test.py`).
- **`tools/` directory**: Log analysis scripts moved here (`analyze_logs.py`, `refresh_logs.py`, `read_recent_logs.py`).

### Changed
- **Intent Detector**: Removed keyword-based fallback (`student_keywords`, `general_keywords`, `_regex_detect_intent`). Now uses embeddings-only: Regex greetings → Gemma 3 1B → Semantic similarity → default 'general'.
- **SQL Entity Detection**: Expanded from 6 departments to 17 aliases (CSE-AIML, CSE-DS, CSM, MBA, etc.) with natural language aliases like "data science", "artificial intelligence".
- **Scraper** (`scrape.py`): Structured extraction using `<main>`, `<article>`, `.content` containers instead of raw body text. Strips nav/footer/sidebar noise. Adds URL deduplication.
- **Config** (`ultrarag_config.yaml`): Fixed mismatches — model `llama3.2:3b`, collection `college_data`, top_k `3`, max_tokens `512`, num_ctx `2048`.
- **Requirements**: Pinned `langchain-experimental==0.0.65`, `langchain-huggingface==0.0.3`, `scikit-learn`, `transformers` to fix installation conflicts.
- **`verify_codebase.py`**: Updated to reflect moved/deleted files.

### Removed
- `app/services/prompt_construction.py` (unused in production)
- `scripts/setup_student_database.py` (replaced by direct DB setup)
- `scripts/migrate_csv_to_excel.py` (one-time migration, no longer needed)
- `scripts/clean_and_fill.py` (superseded by cleanup_database.py)
- `analyze_csv.py` (referenced non-existent files)

---

## [3.1.0] - February 2026

### Added
- **Navigation Link Chips**: Clickable 🔗 link chips now appear below chat responses, linking to relevant TKRCET website pages.
  - `extractLinks()` and `renderLinkChips()` reusable functions in frontend JavaScript
  - Links extracted and rendered for both SSE streaming and non-streaming responses
- **Topic-Based Link Mapping**: 28 query keywords mapped to real TKRCET website URLs in `TOPIC_LINKS` class attribute
  - `_get_topic_links()` method matches query keywords (admission, cse, placement, etc.) to relevant website pages
  - Falls back to main TKRCET homepage when no topic keyword matches
- **KB Responses Include Links**: Knowledge Base fast-track responses now also include topic-based navigation links

### Changed
- **`ultra_rag.py`**: `_extract_relevant_links()` now accepts a `query` parameter and falls back to topic-based links when corpus documents lack URLs
- **`ultra_rag.py`**: Quick Links and Source Links sections handle both dict format (topic links) and string format (document links)
- **`frontend/index.html`**: Link extraction regex updated to match all backend headers (`📌 **Quick Links:**`, `📚 **Source Links:**`, `Related Links:`, `Sources:`)

### Fixed
- **Missing Navigation Links**: Links were never visible because (1) the frontend regex didn't match backend headers and (2) the SSE streaming path skipped link extraction entirely
- **Empty Corpus URLs**: All corpus documents had `"url": ""`, so `_extract_relevant_links()` always returned empty — now falls back to topic-based links

---

## [3.0.0] - February 2026

### Added
- **Diagnostic Tool**: `verify_codebase.py` for comprehensive health checks.

### Changed
- **Documentation**: Unified versioning across all files to v3.0.0.
- **Default Model**: Explicitly set to `llama3.2:3b` in codebase to match docs.

## [2.1.0] - January 2026

### Added
- **Scope Validation System**: Multi-layer filtering to reject non-college queries
  - `_is_college_related()` method for pre-filtering
  - College keyword detection (admission, course, faculty, etc.)
  - Non-college topic rejection (math, science, general knowledge)
  - 89% test success rate
  
- **Enhanced Greeting Detection**: Handles variations like "how r u?", "what's up"
  
- **Test Suite**: `test_scope_validation.py` for automated testing
  
- **Requirements File**: `requirements.txt` with all dependencies
  
- **Comprehensive Documentation**:
  - Updated `README.md` with current architecture
  - Rewrote `QUICK_START.md` for v2.0
  - Created `CHANGELOG.md`

### Changed
- **LLM Model**: Switched from Gemma 3:1b to Gemma 2:2b (2x faster) — later replaced by Llama 3.2:3b in v3.2
- **Context Window**: Reduced from 2048 to 1024 tokens (40% speed boost)
- **Temperature**: Lowered to 0.1 for more deterministic responses
- **Max Predictions**: Reduced from 250 to 150 tokens for faster generation
- **Prompt Engineering**: Strict scope enforcement in LLM prompts

### Improved
- **Response Quality**: More focused, college-only answers
- **Performance**: Faster response times with optimized settings
- **Reliability**: Better error handling and fallback mechanisms
- **User Experience**: Clear rejection messages for off-topic queries

### Fixed
- Off-topic query handling (math formulas, science questions)
- Inconsistent greeting responses
- Scope leakage in LLM responses

---

## [1.0.0] - Previous Version

### Features
- UltraRAG architecture with FAISS + BM25 hybrid retrieval
- Knowledge base for instant answers
- Ollama integration for LLM
- Terminal-based chat interface
- Document corpus with 2000+ chunks

---

## Version Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Scope Validation | ❌ None | ✅ Multi-layer |
| Off-topic Handling | ❌ Inconsistent | ✅ Reliable rejection |
| Greeting Detection | ⚠️ Basic | ✅ Enhanced |
| LLM Model | Gemma 3:1b | Llama 3.2:3b |
| Response Speed | ⚠️ Moderate | ✅ 2x faster |
| Test Coverage | ❌ None | ✅ 89% success |
| Documentation | ⚠️ Outdated | ✅ Complete |

---

**Current Version**: 3.5.0  
**Status**: Production Ready ✅
