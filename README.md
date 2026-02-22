# College Buddy - AI Powered Campus Assistant

## Overview
College Buddy is an intelligent conversational AI designed to assist students, faculty, and visitors of TKRCET. It features a advanced **Hybrid RAG-SQL System** that intelligently routes queries between a semantic search engine (for general college info) and a structured SQL database (for student analytics).

## Key Features
- 🧠 **Hybrid Architecture**: Intelligently routes queries to either **UltraRAG** (general info) or **SQL System** (tabular data).
- 📊 **Student Data Analysis**: Can answer complex queries about student performance, placements, and departments (e.g., "List top 5 companies", "How many students got > 8.5 CGPA?").
- 🤖 **Efficient LLM**: Powered by **Llama 3.2:3b** via Ollama, optimized for speed on local hardware.
- 🛡️ **Scope Validation**: Built-in filtering to reject non-college queries.
-  **Knowledge Base**: Instant answers for critical facts (personnel, timings, location).
- ⚡ **Fast & Lightweight**: Runs efficiently on local hardware with minimal memory footprint.
- 🗣️ **Natural Conversations**: Varied, friendly responses for common queries to avoid robotic answers.
- 🔗 **Navigation Links**: Responses include clickable link chips to relevant TKRCET website pages (admissions, departments, placements, etc.).
- 🌐 **Web Frontend**: Browser-based chat interface with SSE streaming, dark mode, voice I/O, and multi-language support.

## Tech Stack
- **Language**: Python 3.8+
- **LLM**: Llama 3.2:3b (via Ollama)
- **Embeddings**: all-MiniLM-L6-v2
- **Vector DB**: ChromaDB (Semantic Search)
- **Structured DB**: SQLite + Pandas (for Student Data)
- **Routing**: Gemma 3 1B + Regex-based Intent Detection
- **Backend**: FastAPI with SSE streaming
- **Frontend**: HTML/CSS/JS with Tailwind CSS, Marked.js

## Prerequisites
- **OS**: Windows, Linux, or macOS
- **Python**: 3.9 - 3.11
- **RAM**: Minimum 8GB recommended
- **Software**: 
  - [Ollama](https://ollama.com/) (Required)
  - [Git](https://git-scm.com/)
  - **Google Chrome** or **Chromium** (Required for Data Scraping)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/VijayKiran-2004/College-Chat-Bot.git
cd College-Chat-Bot
```

### 2. Create Virtual Environment
> [!IMPORTANT]
> You **MUST** go into the project folder first!
> ```bash
> cd College-Chat-Bot
> ```

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```
*Note: If you see a security error in PowerShell, run `Set-ExecutionPolicy Unrestricted -Scope Process` first.*
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Ollama (Critical)
1. Install [Ollama](https://ollama.com/).
2. Pull the model:
   ```bash
   ollama pull llama3.2:3b
   ```
3. Keep Ollama running in the background (`ollama serve`).

### 5. Data Setup (Crucial Step)
Since raw data is not checked into git, you must generate it locally:

**A. Create Student Database (SQL)**
```bash
python scripts/setup_student_database.py
```
*Creates `app/database/students.db` from Excel data.*

**B. Scrape College Data (Web)**
```bash
python scripts/scrape.py
```
*Scrapes 90+ pages from the college website to `app/database/vectordb/scraped_data.jsonl`. Requires Chrome.*

### 6. Ingest Data (Vector DB)
Now, process the scraped data into the Vector Database:
```bash
python scripts/ingest.py
```
*(This populates the `college_data` ChromaDB collection)*

**Generate Knowledge Corpus:**
```bash
python scripts/corpus_converter.py
```
*(Specifices the JSONL corpus required for UltraRAG)*

### 7. Run the Chatbot

**Option A: Backend API Server (Recommended)**
Start the REST API server to power frontend applications:
```bash
python backend.py
```
*Server runs at `http://127.0.0.1:8000`*

**Option B: Terminal Interface**
For quick testing in the command line:
```bash
python terminal_chat.py
```

## Usage Examples

### 🏢 General Queries (RAG)
- "Who is the principal?"
- "What are the college timings?"
- "Tell me about CSE department facilities."
- "How do I apply for admission?"

### 🎓 Student Data Queries (SQL)
- "List all students in CSE department"
- "Show students with CGPA > 8.5"
- "Who are the top recruiters?"
- "How many students got placed in TCS?"
- "What is the average CGPA of ECE students?"

### 🔄 Hybrid Queries
- "Show students with > 9.0 CGPA and tell me about the placement cell."

## Architecture

### System Architecture
The College Buddy system follows a hybrid architecture combining Retrieval-Augmented Generation (RAG) with Structured Query Language (SQL) capabilities.

```mermaid
graph TD
    User([User]) <--> Interface[Terminal Interface]
    Interface <--> Router[Query Router]
    Router --> Detector[Intent Detector]
    
    Detector -->|General| RAG[UltraRAG System]
    Detector -->|Student| SQL[SQL System]
    Detector -->|Hybrid| Fusion[Result Fusion]
    
    Fusion --> RAG
    Fusion --> SQL
    
    RAG <--> KB[Knowledge Base]
    RAG <--> HybridSearch{Hybrid Search}
    HybridSearch <--> VectorDB[(Vector DB)]
    RAG --> LLM[Ollama Llama Model]
    
    SQL <--> QueryBuilder[Query Builder]
    QueryBuilder <--> DB[(Student DB)]
    SQL --> Analysis[Data Analysis]
```

### Data Flow
How a user query is processed and answered:

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant I as Intent Detector
    participant RS as RAG System
    participant SS as SQL System
    
    U->>R: "How many CSE students got > 8.5 CGPA?"
    R->>I: Analyze Query
    I-->>R: Intent: "Student"
    
    R->>SS: Execute Query
    SS->>SS: Extract Entities
    SS->>SS: Build SQL
    SS->>SS: Generate Summary
    SS-->>R: Return Response
    
    R-->>U: "Summary: 45 students found..."
    
    Note over U,R: Hybrid Query Example
    U->>R: "Show toppers and tell me about placements"
    R->>I: Analyze Query
    I-->>R: Intent: "Hybrid"
    
    par Parallel Processing
        R->>SS: Get Toppers
        R->>RS: Get Placement Info
    end
    
    R->>R: Combine Results
    R-->>U: "Here are the toppers... and placement info..."
```

## Project Structure
```
college-buddy/
├── app/
│   ├── services/
│   │   ├── ultra_rag.py           # General Info Engine (RAG) + Topic Links
│   │   ├── sql_system.py          # Student Data Engine (SQL)
│   │   ├── intent_detector.py     # Router Logic
│   │   ├── query_router.py        # Main Controller
│   │   └── chain.py               # Chain-of-Thought handling
│   ├── database/
│   │   ├── students.db            # SQLite Student DB
│   │   └── vectordb/              # ChromaDB Storage + Corpus
│
├── frontend/
│   └── index.html                 # Web Chat UI (SSE + Link Chips)
├── backend.py                     # FastAPI Server (REST API + SSE)
├── terminal_chat.py               # CLI Chat Interface
├── verify_codebase.py             # Diagnostic Tool
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## Privacy & Security
- **Student Data**: The SQL system is designed to provide *aggregate* summaries for general queries (e.g., "count of students") to protect privacy. Individual records are only shown for specific ID lookups.
- **Local Processing**: All data stays on your local machine.

## Team
- **Vijay Kiran**: System Architecture
- **Sanjana**: Data Pipeline
- **Subhash Chandra**: Database
- **Sathish**: Vector Optimization
- **Mokshagna**: LLM Integration
- **Vishnuvardhan**: Prompt Engineering
- **Praneetha**: Testing

---
**Version**: 3.1.0 (Hybrid Edition)
**Status**: Production Ready ✅
