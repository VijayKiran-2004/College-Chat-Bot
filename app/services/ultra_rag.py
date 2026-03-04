"""
UltraRAG System - Modern MCP-based RAG implementation
Refactored to orchestrate independent components (KnowledgeBase, LinkManager, Retriever, Generator)
"""

import sys
import os
import re
import json
from pathlib import Path

# Fix Windows encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Fix for WinError 1114 (DLL Initialization Failed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from app.services.knowledge_base import KnowledgeBase
from app.services.link_manager import LinkManager
from app.services.retriever import Retriever
from app.services.generator import Generator
from app.services.embedding_service import get_embedding_model

# ---------------------------------------------------------------------------
# Query Expansion: College-specific synonyms to improve retrieval coverage
# ---------------------------------------------------------------------------
_QUERY_SYNONYMS = {
    "result":     ["result", "marks", "grade", "exam result", "score"],
    "results":    ["results", "marks", "grades", "exam results", "scores"],
    "attendance": ["attendance", "present", "absent", "shortage"],
    "fee":        ["fee", "fees", "cost", "payment", "charges"],
    "hostel":     ["hostel", "accommodation", "room", "residence"],
    "bus":        ["bus", "transport", "route", "vehicle"],
    "placement":  ["placement", "placed", "job", "company", "recruit"],
    "library":    ["library", "books", "reading room", "digital library"],
    "scholarship":["scholarship", "stipend", "financial aid", "eamcet rank"],
    "sports":     ["sports", "games", "ground", "athletic"],
    "exam":       ["exam", "examination", "test", "internal", "external"],
    "internship": ["internship", "training", "industrial visit", "in-plant"],
}

def _expand_query(query: str) -> str:
    """Expand query with domain synonyms to improve semantic retrieval."""
    query_lower = query.lower()
    extras = []
    for term, synonyms in _QUERY_SYNONYMS.items():
        if term in query_lower:
            # Add synonyms not already in the query
            for syn in synonyms:
                if syn not in query_lower:
                    extras.append(syn)
            break  # Only expand the first matched term to avoid noise
    if extras:
        expanded = query + " " + " ".join(extras[:3])  # max 3 extras
        return expanded
    return query

class UltraRAGSystem:
    """
    UltraRAG-based RAG system for college-buddy chatbot
    Refactored into a facade composing KnowledgeBase, Retriever, Generator, and LinkManager.
    """
    
    def __init__(
        self,
        corpus_path=None,
        ollama_model=None,
        ollama_url=None,
        semantic_model=None
    ):
        self.ollama_model = ollama_model or os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')
        self.ollama_url = ollama_url or os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.corpus_path = corpus_path or str(self.project_root / 'data/chunks/corpus_ultrarag.jsonl')
        self.kb_path = str(self.project_root / 'data/knowledge_base.json')
        self._cache_path = str(self.project_root / 'data/chunks/response_cache.json')
        
        # Initialize sub-components
        if semantic_model is None:
            print("Fetching shared embedding model...")
            semantic_model = get_embedding_model('all-MiniLM-L6-v2')
            
        self.kb = KnowledgeBase(self.kb_path, semantic_model)
        self.kb.load_sql_stats()
        
        self.link_manager = LinkManager()
        self.retriever = Retriever(self.project_root, self.corpus_path)
        self.generator = Generator(self.ollama_model, self.ollama_url)
        
        # Persistent cache: load from disk if available
        self.response_cache = self._load_cache()
        print(f"  [Cache] Loaded {len(self.response_cache)} cached responses from disk")
        
        # Context carry-forward: last retrieved context snippets for follow-ups
        self._last_context: list = []
        self._last_docs: list = []
        
        print("✓ UltraRAGSystem ready!\n")

    def _load_cache(self) -> dict:
        """Load persistent cache from disk."""
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_cache(self):
        """Save cache to disk (fire-and-forget, errors are non-fatal)."""
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [Cache] Could not save cache: {e}")

    def _is_followup(self, query: str) -> bool:
        """Detect if a short query is likely a follow-up to the previous one."""
        words = query.strip().split()
        if len(words) > 7:
            return False
        followup_starters = [
            "what about", "and", "also", "what is his", "what is her",
            "tell me more", "more details", "elaborate", "explain more",
            "how about", "who is he", "who is she", "what else"
        ]
        q_lower = query.lower().strip()
        return any(q_lower.startswith(s) for s in followup_starters)

    def _stream_call(self, query, is_greeting=False, language='en', temperature=0.2):
        """Streaming generator entry point for queries"""
        if not query:
            yield "Please enter a question."
            return
            
        if is_greeting:
            # Skip retrieval for greetings
            yield {"type": "metadata", "source": "Greeting Fast-track", "confidence": 100}
            for chunk in self.generator.generate(
                query=query,
                docs=[],
                kb_context="",
                kb_fact=None,
                links=[],
                language=language,
                stream=True,
                temperature=temperature,
                is_greeting=True
            ):
                yield chunk
            return

        # Check Knowledge Base for raw facts
        kb_fact = self.kb.check(query)
        metadata = {"type": "metadata", "source": "RAG"}
        if kb_fact:
            metadata["source"] = "Knowledge Base"
            
        # Query expansion for better retrieval
        search_query = _expand_query(query)
        
        # Context carry-forward for short follow-up queries
        if self._is_followup(query) and self._last_docs:
            print("  [Context] Detected follow-up — reusing previous context")
            docs = self._last_docs
        else:
            docs = self.retriever.retrieve(search_query, top_k=4)
            if docs:
                self._last_docs = docs
        
        confidence = 0
        if docs:
            avg_score = sum(d.get('relevance_score', 0) for d in docs) / len(docs)
            confidence = max(0, min(100, int(avg_score * 100)))
            metadata["confidence"] = confidence
            
        context_snippets = [d.get("contents", "") for d in docs[:3]]
        metadata["context"] = context_snippets
        
        yield metadata
        
        links = self.link_manager.extract_relevant_links(docs, query)
        kb_context = self.kb.format_context()
        
        # Stream Generation (Integrating KB Fact as a primary source if found)
        for chunk in self.generator.generate(
            query=query, 
            docs=docs[:2],  # Send top 2 to LLM (best after re-ranking)
            kb_context=kb_context, 
            kb_fact=kb_fact,
            links=links, 
            language=language, 
            stream=True, 
            temperature=temperature
        ):
            yield chunk

    def __call__(self, query, is_greeting=False, language='en', stream=False, return_dict=False, temperature=0.2):
        """Main entry point for queries"""
        query = query.strip()
        
        if stream:
            return self._stream_call(query, is_greeting, language, temperature)
            
        if not query:
            return {"response": "Please enter a question.", "source": "System"} if return_dict else "Please enter a question."
            
        if is_greeting:
            # Fast-track for greetings
            response = self.generator.generate(
                query=query,
                docs=[],
                kb_context="",
                kb_fact=None,
                links=[],
                language=language,
                stream=False,
                temperature=temperature,
                is_greeting=True
            )
            if return_dict:
                return {"response": response, "source": "Greeting Fast-track", "confidence": 100}
            return response

        # Check Cache
        query_key = query.lower().strip()
        if query_key in self.response_cache:
            print("  [Cache Hit] Returning cached response")
            return {"response": self.response_cache[query_key], "source": "Cache"} if return_dict else self.response_cache[query_key]
            
        # Check Knowledge Base for raw facts
        kb_fact = self.kb.check(query)
        source = "Knowledge Base" if kb_fact else "RAG"
        
        # Query expansion for better retrieval
        search_query = _expand_query(query)
        
        # Context carry-forward for short follow-up queries
        if self._is_followup(query) and self._last_docs:
            print("  [Context] Detected follow-up — reusing previous context")
            docs = self._last_docs
        else:
            docs = self.retriever.retrieve(search_query, top_k=4)
            if docs:
                self._last_docs = docs
        
        confidence = 0
        if docs:
            avg_score = sum(d.get('relevance_score', 0) for d in docs) / len(docs)
            confidence = max(0, min(100, int(avg_score * 100)))
            
        links = self.link_manager.extract_relevant_links(docs, query)
        kb_context = self.kb.format_context()
        
        # Generation — send top 2 docs (best after re-ranking)
        response = self.generator.generate(
            query=query, 
            docs=docs[:2],
            kb_context=kb_context, 
            kb_fact=kb_fact,
            links=links, 
            language=language, 
            stream=False, 
            temperature=temperature
        )
        
        # Persist cache to disk (trim if too large)
        self.response_cache[query_key] = response
        if len(self.response_cache) > 200:
            oldest_keys = list(self.response_cache.keys())[:50]
            for k in oldest_keys:
                del self.response_cache[k]
        self._save_cache()

        if return_dict:
            return {"response": response, "source": source, "confidence": confidence}
        return response


