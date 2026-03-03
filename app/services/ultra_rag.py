"""
UltraRAG System - Modern MCP-based RAG implementation
Refactored to orchestrate independent components (KnowledgeBase, LinkManager, Retriever, Generator)
"""

import sys
import os
import re
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
        self.corpus_path = corpus_path or str(self.project_root / 'app/database/vectordb/corpus_ultrarag.jsonl')
        self.kb_path = str(self.project_root / 'data/knowledge_base.json')
        
        # Initialize sub-components
        if semantic_model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading local SentenceTransformer model...")
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.kb = KnowledgeBase(self.kb_path, semantic_model)
        self.kb.load_sql_stats()
        
        self.link_manager = LinkManager()
        self.retriever = Retriever(self.project_root, self.corpus_path)
        self.generator = Generator(self.ollama_model, self.ollama_url)
        
        # Response cache
        self.response_cache = {}
        
        print("✓ UltraRAGSystem ready!\n")

    def _stream_call(self, query, language='en', temperature=0.2):
        """Streaming generator entry point for queries"""
        if not query:
            yield "Please enter a question."
            return
            
        # Greetings (Flexible Regex Matching)
        greetings_pattern = r"^(hi|hello|hey|greetings|how are you|how r u|how are u|whats up|what's up|how do you do|good morning|good afternoon|good evening)[\s\?\!\.]*$"
        if re.match(greetings_pattern, query.lower()):
            greeting_response = "Hello! I'm your TKRCET College Buddy! 😊 How can I help you today!"
            yield {"type": "metadata", "source": "Greeting"}
            yield greeting_response
            return
            
        # Check Knowledge Base
        kb_answer = self.kb.check(query)
        if kb_answer:
            topic_links = self.link_manager.get_topic_links(query)
            if topic_links:
                kb_answer += "\n\n📌 **Quick Links:**\n" + "".join([f"• [{tl['title']}]({tl['url']})\n" for tl in topic_links])
            yield {"type": "metadata", "source": "Knowledge Base"}
            yield kb_answer
            return
            
        # Retrieve Documents
        docs = self.retriever.retrieve(query, top_k=3)
        
        confidence = 0
        if docs:
            avg_score = sum(d.get('relevance_score', 0) for d in docs) / len(docs)
            confidence = max(0, min(100, int(avg_score * 100)))
            
        context_snippets = [d.get("contents", "") for d in docs[:3]]
        yield {"type": "metadata", "source": "RAG", "confidence": confidence, "context": context_snippets}
        
        links = self.link_manager.extract_relevant_links(docs, query)
        kb_context = self.kb.format_context()
        
        # Stream Generation
        for chunk in self.generator.generate(
            query=query, 
            docs=docs, 
            kb_context=kb_context, 
            links=links, 
            language=language, 
            stream=True, 
            temperature=temperature
        ):
            yield chunk

    def __call__(self, query, language='en', stream=False, return_dict=False, temperature=0.2):
        """Main entry point for queries"""
        query = query.strip()
        
        if stream:
            return self._stream_call(query, language, temperature)
            
        if not query:
            return {"response": "Please enter a question.", "source": "System"} if return_dict else "Please enter a question."
            
        # Greetings
        greetings_pattern = r"^(hi|hello|hey|greetings|how are you|how r u|how are u|whats up|what's up|how do you do|good morning|good afternoon|good evening)[\s\?\!\.]*$"
        if re.match(greetings_pattern, query.lower()):
            greeting_response = "Hello! I'm your TKRCET College Buddy! 😊 How can I help you today!"
            return {"response": greeting_response, "source": "Greeting"} if return_dict else greeting_response
            
        # Check Cache
        query_key = query.lower().strip()
        if query_key in self.response_cache:
            print("  [Cache Hit] Returning cached response")
            return {"response": self.response_cache[query_key], "source": "Cache"} if return_dict else self.response_cache[query_key]
            
        # Check Knowledge Base
        kb_answer = self.kb.check(query)
        if kb_answer:
            topic_links = self.link_manager.get_topic_links(query)
            if topic_links:
                kb_answer += "\n\n📌 **Quick Links:**\n" + "".join([f"• [{tl['title']}]({tl['url']})\n" for tl in topic_links])
            
            self.response_cache[query_key] = kb_answer
            if len(self.response_cache) > 50:
                self.response_cache.pop(next(iter(self.response_cache)))
            
            return {"response": kb_answer, "source": "Knowledge Base"} if return_dict else kb_answer
            
        # Retrieve Documents
        docs = self.retriever.retrieve(query, top_k=3)
        
        confidence = 0
        if docs:
            avg_score = sum(d.get('relevance_score', 0) for d in docs) / len(docs)
            confidence = max(0, min(100, int(avg_score * 100)))
            
        links = self.link_manager.extract_relevant_links(docs, query)
        kb_context = self.kb.format_context()
        
        # Generation
        response = self.generator.generate(
            query=query, 
            docs=docs, 
            kb_context=kb_context, 
            links=links, 
            language=language, 
            stream=False, 
            temperature=temperature
        )
        
        if return_dict:
            return {"response": response, "source": "RAG", "confidence": confidence}
        return response

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ULTRARAG SYSTEM TEST")
    print("=" * 70 + "\n")
    
    rag = UltraRAGSystem()
    test_queries = [
        "hi",
        "who is the principal?",
        "what are the facilities?",
        "college timings?",
    ]
    for query in test_queries:
        print(f"Q: {query}")
        print(f"A: {rag(query)}\n")
