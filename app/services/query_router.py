"""
Query Router - Intelligent routing between RAG and SQL systems
"""

from app.services.chain import DeepReasoningChain
from app.services.intent_detector import IntentDetector
from app.services.sql_system import SQLSystem
from app.services.ultra_rag import UltraRAGSystem


class QueryRouter:
    def __init__(self):
        """Initialize query router with all systems."""
        print("Initializing Query Router...")

        from sentence_transformers import SentenceTransformer

        print("Loading shared SentenceTransformer model...")
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Shared semantic model loaded")

        self.intent_detector = IntentDetector(
            semantic_model=self.semantic_model
        )
        print("✓ Intent Detector loaded")

        self.rag_system = UltraRAGSystem(
            semantic_model=self.semantic_model
        )
        print("✓ UltraRAG System loaded")

        self.sql_system = SQLSystem()
        print("✓ SQL System loaded")

        self.reasoning_chain = DeepReasoningChain(
            rag_system=self.rag_system,
            sql_system=self.sql_system,
        )
        print("✓ Deep Reasoning Chain loaded")

        print("✓ Query Router ready!\n")

    def route_query(self, query, chat_history=None):
        """
        Route query to appropriate system(s).

        Args:
            query: Natural language query
            chat_history: Optional conversation history

        Returns:
            dict with response, source and accuracy
        """

        intent = self.intent_detector.detect_intent(query)

        general_keywords = [
            "fest",
            "tournament",
            "sports",
            "event",
            "club",
            "campus",
            "hostel",
            "bus",
            "transport",
        ]

        if any(keyword in query.lower() for keyword in general_keywords):
            intent = "general"

        if intent == "greeting":

            result = self.rag_system(
                query,
                is_greeting=True,
                return_dict=True,
            )

            if isinstance(result, dict):
                response = result.get("response")
                source = result.get(
                    "source",
                    "Greeting Fast-track",
                )
                confidence = result.get("confidence", 100)
            else:
                response = result
                source = "Greeting Fast-track"
                confidence = 100

            return {
                "response": response,
                "source": source,
                "accuracy": f"{confidence}%",
            }

        elif intent == "general":

            result = self.rag_system(
                query,
                return_dict=True,
            )

            if isinstance(result, dict):
                response = result.get("response")
                source = result.get(
                    "source",
                    "RAG/Knowledge Base",
                )
                confidence = result.get("confidence", 0)
            else:
                response = result
                source = "RAG/Knowledge Base"
                confidence = 0

            return {
                "response": response,
                "source": source,
                "accuracy": f"{confidence}%",
            }

        elif intent == "student":

            response = self.sql_system.query_students(
                query,
                chat_history=chat_history,
            )

            return {
                "response": response,
                "source": "SQL Database",
                "accuracy": "100%",
            }

        elif intent == "hybrid":

            print("  -> Routing to Agent (Deep Reasoning Chain)")
            response = self.reasoning_chain.run(query)

            return {
                "response": response,
                "source": "Deep Reasoning Agent",
                "accuracy": "N/A",
            }

        else:

            print("  -> Intent unclear, routing to Agent")
            response = self.reasoning_chain.run(query)

            return {
                "response": response,
                "source": "Deep Reasoning Agent (Fallback)",
                "accuracy": "N/A",
            }

    def __call__(self, query, chat_history=None):
        """Make router callable."""
        result = self.route_query(query, chat_history=chat_history)

        if isinstance(result, dict):
            return result["response"]

        return result

    def close(self):
        """Close all connections."""
        self.sql_system.close()


if __name__ == "__main__":
    router = QueryRouter()

    print("=" * 70)
    print("TESTING QUERY ROUTER")
    print("=" * 70 + "\n")

    test_queries = [
        "Who is the principal?",
        "What are college timings?",
        "List all CSE students",
        "Show students with CGPA > 8.5",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 70)
        response = router(query)
        print(f"Response: {response}")
        print("\n")

    router.close()
