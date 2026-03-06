"""
Query Router - Intelligent routing between RAG and SQL systems
"""
from app.services.intent_detector import IntentDetector
from app.services.ultra_rag import UltraRAGSystem
from app.services.sql_system import SQLSystem
from app.services.chain import DeepReasoningChain

class QueryRouter:
    def __init__(self):
        """Initialize query router with all systems"""
        print("Initializing Query Router...")
        
        from sentence_transformers import SentenceTransformer
        print("Loading shared SentenceTransformer model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Shared semantic model loaded")
        
        self.intent_detector = IntentDetector(semantic_model=self.semantic_model)
        print("✓ Intent Detector loaded")
        
        self.rag_system = UltraRAGSystem(semantic_model=self.semantic_model)  # Using new UltraRAG system
        print("✓ UltraRAG System loaded")
        
        self.sql_system = SQLSystem()
        print("✓ SQL System loaded")
        
        # Initialize Deep Reasoning Chain with existing systems (Unified Brain)
        self.reasoning_chain = DeepReasoningChain(
            rag_system=self.rag_system,
            sql_system=self.sql_system
        )
        print("✓ Deep Reasoning Chain loaded")
        
        print("✓ Query Router ready!\n")
    
    def route_query(self, query, chat_history=None):
        """
        Route query to appropriate system(s)
        
        Args:
            query: Natural language query
            chat_history: Optional list of previous conversation turns for context
        
        Returns:
            dict: {
                "response": str,
                "source": str,
                "accuracy": str  # Real confidence score
            }
        """
        # Detect intent
        intent = self.intent_detector.detect_intent(query)
        
        # Override for specific keywords that might be misclassified as student queries
        general_keywords = [
            'fest', 'tournament', 'sports', 'event', 'club', 'campus',
            'hostel', 'bus', 'transport',
            # Procedural / result queries go to RAG (not SQL)
            'result', 'results', 'marks', 'attendance',
            # Personnel / info queries should NOT hit SQL
            'head of', 'who is head', 'who runs', 'in charge',
        ]
        if any(keyword in query.lower() for keyword in general_keywords):
           intent = 'general'
        
        # Also force 'general' if the query starts with 'who is' without student identifiers
        query_lower = query.lower().strip()
        student_ids = ['placed', 'topper', 'highest cgpa', 'highest package', 'roll']
        if query_lower.startswith('who is') and not any(s in query_lower for s in student_ids):
            intent = 'general'
        
        if intent == 'greeting':
            # Fast-track greetings: Skip RAG retrieval
            result = self.rag_system(query, is_greeting=True, return_dict=True)
            if isinstance(result, dict):
                 response = result.get('response')
                 source = result.get('source', 'Greeting Fast-track')
                 confidence = result.get('confidence', 100)
                 context = result.get('context', [])
            else:
                 response = result
                 source = "Greeting Fast-track"
                 confidence = 100
                 context = []
            
            return {
                "response": response,
                "source": source,
                "accuracy": f"{confidence}%",
                "context": context
            }

        elif intent == 'general':
            # Use RAG system normally
            result = self.rag_system(query, return_dict=True)
            if isinstance(result, dict):
                 response = result.get('response')
                 source = result.get('source', 'RAG/Knowledge Base')
                 confidence = result.get('confidence', 0)
                 context = result.get('context', [])
            else:
                 response = result
                 source = "RAG/Knowledge Base"
                 confidence = 0
                 context = []
            
            return {
                "response": response,
                "source": source,
                "accuracy": f"{confidence}%",
                "context": context
            }
        
        elif intent == 'student':
            # Use SQL system only (with context for follow-up questions)
            response = self.sql_system.query_students(query, chat_history=chat_history)
            return {
                "response": response,
                "source": "SQL Database",
                "accuracy": "100%",
                "context": [f"SQL Result: {response}"]
            }
        
        elif intent == 'hybrid':
            # Use Deep Reasoning Chain for complex queries
            print("  -> Routing to Agent (Deep Reasoning Chain)")
            response = self.reasoning_chain.run(query)
            return {
                "response": response,
                "source": "Deep Reasoning Agent",
                "accuracy": "N/A",
                "context": [f"Agent reasoning for: {query}"]
            }
        
        else:
            # Fallback to Agent if intent is unclear (Safety Net)
            print("  -> Intent unclear, routing to Agent")
            response = self.reasoning_chain.run(query)
            return {
                "response": response,
                "source": "Deep Reasoning Agent (Fallback)",
                "accuracy": "N/A",
                "context": [f"Agent reasoning for: {query}"]
            }

    
    def __call__(self, query, chat_history=None):
        """Make class callable"""
        result = self.route_query(query, chat_history=chat_history)
        if isinstance(result, dict):
            return result['response'] # For simple string compatibility
        return result
    
    def close(self):
        """Close all connections"""
        self.sql_system.close()

if __name__ == "__main__":
    # Test query router
    router = QueryRouter()
    
    print("="*70)
    print("TESTING QUERY ROUTER")
    print("="*70 + "\n")
    
    test_queries = [
        "Who is the principal?",  # general -> RAG
        "What are college timings?",  # general -> RAG
        "List all CSE students",  # student -> SQL
        "Show students with CGPA > 8.5",  # student -> SQL
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-"*70)
        response = router(query)
        print(f"Response: {response}")
        print("\n")
    
    router.close()
