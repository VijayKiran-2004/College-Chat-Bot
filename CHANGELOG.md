# Changelog - College Buddy

All notable changes to this project will be documented in this file.

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
- **LLM Model**: Switched from Gemma 3:1b to Gemma 2:2b (2x faster)
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
| LLM Model | Gemma 3:1b | Gemma 2:2b |
| Response Speed | ⚠️ Moderate | ✅ 2x faster |
| Test Coverage | ❌ None | ✅ 89% success |
| Documentation | ⚠️ Outdated | ✅ Complete |

---

**Current Version**: 3.1.0  
**Status**: Production Ready ✅
