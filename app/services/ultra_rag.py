"""
UltraRAG System - Modern MCP-based RAG implementation
Refactored to orchestrate independent components
(KnowledgeBase, LinkManager, Retriever, Generator)
"""

import io
import os
import sys
from pathlib import Path

from app.services.embedding_service import get_embedding_model
from app.services.generator import Generator
from app.services.knowledge_base import KnowledgeBase
from app.services.link_manager import LinkManager
from app.services.retriever import Retriever

# Fix Windows encoding
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
    )

# Fix for WinError 1114 (DLL Initialization Failed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class UltraRAGSystem:
    """
    UltraRAG-based RAG system for college-buddy chatbot.

    Facade composing KnowledgeBase, Retriever,
    Generator, and LinkManager.
    """

    def __init__(
        self,
        corpus_path=None,
        ollama_model=None,
        ollama_url=None,
        semantic_model=None,
    ):
        self.ollama_model = ollama_model or os.environ.get(
            "OLLAMA_MODEL",
            "llama3.2:3b",
        )

        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_URL",
            "http://127.0.0.1:11434/api/generate",
        )

        self.project_root = Path(__file__).resolve().parent.parent.parent

        self.corpus_path = corpus_path or str(
            self.project_root / "data/chunks/corpus_ultrarag.jsonl"
        )

        self.kb_path = str(
            self.project_root / "data/knowledge_base.json"
        )

        if semantic_model is None:
            print("Fetching shared embedding model...")
            semantic_model = get_embedding_model(
                "all-MiniLM-L6-v2"
            )

        self.kb = KnowledgeBase(self.kb_path, semantic_model)
        self.kb.load_sql_stats()

        self.link_manager = LinkManager()
        self.retriever = Retriever(
            self.project_root,
            self.corpus_path,
        )

        self.generator = Generator(
            self.ollama_model,
            self.ollama_url,
        )

        self.response_cache = {}

        print("✓ UltraRAGSystem ready!\n")

    def _stream_call(
        self,
        query,
        is_greeting=False,
        language="en",
        temperature=0.2,
    ):
        """Streaming generator entry point."""

        if not query:
            yield "Please enter a question."
            return

        if is_greeting:

            yield {
                "type": "metadata",
                "source": "Greeting Fast-track",
                "confidence": 100,
            }

            for chunk in self.generator.generate(
                query=query,
                docs=[],
                kb_context="",
                kb_fact=None,
                links=[],
                language=language,
                stream=True,
                temperature=temperature,
                is_greeting=True,
            ):
                yield chunk
            return

        kb_fact = self.kb.check(query)

        metadata = {"type": "metadata", "source": "RAG"}

        if kb_fact:
            metadata["source"] = "Knowledge Base"

        docs = self.retriever.retrieve(query, top_k=2)

        confidence = 0

        if docs:
            avg_score = sum(
                d.get("relevance_score", 0) for d in docs
            ) / len(docs)

            confidence = max(
                0,
                min(100, int(avg_score * 100)),
            )

            metadata["confidence"] = confidence

        context_snippets = [
            d.get("contents", "") for d in docs[:3]
        ]

        metadata["context"] = context_snippets

        yield metadata

        links = self.link_manager.extract_relevant_links(
            docs,
            query,
        )

        kb_context = self.kb.format_context()

        for chunk in self.generator.generate(
            query=query,
            docs=docs,
            kb_context=kb_context,
            kb_fact=kb_fact,
            links=links,
            language=language,
            stream=True,
            temperature=temperature,
        ):
            yield chunk

    def __call__(
        self,
        query,
        is_greeting=False,
        language="en",
        stream=False,
        return_dict=False,
        temperature=0.2,
    ):
        """Main entry point for queries."""

        query = query.strip()

        if stream:
            return self._stream_call(
                query,
                is_greeting,
                language,
                temperature,
            )

        if not query:
            if return_dict:
                return {
                    "response": "Please enter a question.",
                    "source": "System",
                }
            return "Please enter a question."

        if is_greeting:

            response = self.generator.generate(
                query=query,
                docs=[],
                kb_context="",
                kb_fact=None,
                links=[],
                language=language,
                stream=False,
                temperature=temperature,
                is_greeting=True,
            )

            if return_dict:
                return {
                    "response": response,
                    "source": "Greeting Fast-track",
                    "confidence": 100,
                }

            return response

        query_key = query.lower().strip()

        if query_key in self.response_cache:

            print("  [Cache Hit] Returning cached response")

            if return_dict:
                return {
                    "response": self.response_cache[query_key],
                    "source": "Cache",
                }

            return self.response_cache[query_key]

        kb_fact = self.kb.check(query)

        source = "Knowledge Base" if kb_fact else "RAG"

        docs = self.retriever.retrieve(query, top_k=2)

        confidence = 0

        if docs:
            avg_score = sum(
                d.get("relevance_score", 0) for d in docs
            ) / len(docs)

            confidence = max(
                0,
                min(100, int(avg_score * 100)),
            )

        links = self.link_manager.extract_relevant_links(
            docs,
            query,
        )

        kb_context = self.kb.format_context()

        response = self.generator.generate(
            query=query,
            docs=docs,
            kb_context=kb_context,
            kb_fact=kb_fact,
            links=links,
            language=language,
            stream=False,
            temperature=temperature,
        )

        self.response_cache[query_key] = response

        if len(self.response_cache) > 50:
            self.response_cache.pop(
                next(iter(self.response_cache))
            )

        if return_dict:
            return {
                "response": response,
                "source": source,
                "confidence": confidence,
            }

        return response
