"""
Intent Detector - Smart routing using Gemma 3 1B (via Ollama) + Semantic Similarity fallback
Uses google/gemma-3-1b-it through Ollama for AI-powered intent classification.
Falls back to semantic similarity (all-MiniLM-L6-v2) if Ollama is unavailable.
"""
import re
import os
import sys
import io
import json
import requests
import numpy as np

# Fix Windows encoding (only if not already wrapped)
if sys.platform.startswith('win') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class IntentDetector:
    def __init__(self, semantic_model=None):
        """Initialize intent detector with Gemma 3 1B (Ollama) + semantic fallback"""
        
        self.ollama_url = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        self.router_model = 'gemma3:1b'
        self.use_gemma_routing = False
        self.use_semantic_fallback = False
        self.semantic_model = None
        self.intent_embeddings = {}
        
        # System prompt for Gemma 3 routing
        self.router_prompt = """Classify this college chatbot query into ONE category. Reply with ONLY the category name.

GREETING: Simple greetings (hi, hello, hey, good morning, how are you)

STUDENT: Queries requesting student DATA, numbers, lists, or statistics. Examples:
- "top 5 students gpa" -> STUDENT
- "how many students were placed" -> STUDENT  
- "list CSE students" -> STUDENT
- "CGPA of student 12345" -> STUDENT
- "how many students are not placed" -> STUDENT
- "show placement statistics" -> STUDENT

GENERAL: Queries about college INFORMATION, processes, people, or facilities. Examples:
- "who is the principal" -> GENERAL
- "how to apply for bonafide" -> GENERAL
- "college timings" -> GENERAL
- "how many students pass out every year" -> GENERAL
- "what is the fee structure" -> GENERAL
- "exam timetable" -> GENERAL

Reply ONLY: GREETING, STUDENT, or GENERAL"""


        # Load semantic fallback (always, as backup)
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.semantic_model = semantic_model if semantic_model is not None else SentenceTransformer('all-MiniLM-L6-v2')
            
            intent_examples = {
                'greeting': ["hi", "hello", "hey", "good morning", "how are you"],
                'student': [
                    "show me student cgpa scores", "list students with high gpa",
                    "which students are placed", "top 5 students by gpa",
                    "how many students were placed", "student attendance details",
                    "number of placed students", "students with cgpa above 8",
                    "average gpa of students", "branch wise student count",
                    "student placement statistics", "how many students are not placed"
                ],
                'general': [
                    "who is the principal", "what are college timings",
                    "how to apply for bonafide certificate", "what courses are offered",
                    "fee structure details", "hostel facilities", "transport routes",
                    "campus life and events", "exam schedule and timetable",
                    "college history", "how to apply for scholarship",
                    "college location and address", "how many students pass out every year",
                    "what is the college ranking", "tell me about admissions"
                ]
            }
            
            self.intent_embeddings = {}
            for intent, examples in intent_examples.items():
                self.intent_embeddings[intent] = self.semantic_model.encode(
                    examples, show_progress_bar=False
                )
            
            self.use_semantic_fallback = True
            if not self.use_gemma_routing:
                print(f"  ✓ Semantic fallback ready (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"  ⚠ Semantic fallback not available ({e})")
        

    
    def detect_intent(self, query):
        """
        Detect query intent. Priority: Regex greetings → Gemma 3 → Semantic
        
        Returns: 'greeting', 'student', 'general', or 'hybrid'
        """
        query_lower = query.lower().strip()
        
        # Always check greetings with regex first (instant)
        greetings_pattern = r"^(hi|hello|hey|greetings|how are you|how r u|how are u|whats up|what's up|how do you do|good morning|good afternoon|good evening)[\s\?\!\.]*$"
        if re.match(greetings_pattern, query_lower):
            return 'greeting'
        
        # Try Gemma 3 1B routing via Ollama
        if self.use_gemma_routing:
            try:
                return self._gemma_detect_intent(query)
            except Exception as e:
                print(f"  ⚠ Gemma routing failed ({e}), using semantic fallback")
        
        # Semantic similarity fallback (embeddings-based)
        if self.use_semantic_fallback:
            try:
                return self._semantic_detect_intent(query)
            except Exception as e:
                print(f"  ⚠ Semantic routing failed ({e}), defaulting to general")
        
        # Default to 'general' when no detection method is available
        return 'general'
    
    def _gemma_detect_intent(self, query):
        """Use Gemma 3 1B via Ollama for intent classification"""
        prompt = f"{self.router_prompt}\n\nUser query: \"{query}\"\n\nCategory:"
        
        resp = requests.post(
            self.ollama_url,
            json={
                "model": self.router_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "temperature": 0,
                    "top_k": 1
                }
            },
            timeout=15
        )
        
        if resp.status_code != 200:
            raise Exception(f"Ollama returned {resp.status_code}")
        
        result = resp.json().get('response', '').strip().upper()
        
        # Parse the response
        if 'GREETING' in result:
            intent = 'greeting'
        elif 'STUDENT' in result:
            intent = 'student'
        elif 'GENERAL' in result:
            intent = 'general'
        else:
            # Gemma returned something unexpected, use semantic fallback
            print(f"  [Gemma Router] Unexpected: '{result}', using fallback")
            if self.use_semantic_fallback:
                return self._semantic_detect_intent(query)
            return 'general'
        
        print(f"  [Gemma Router] '{query}' -> {intent}")
        return intent
    
    def _semantic_detect_intent(self, query):
        """Use semantic similarity as fallback"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embedding = self.semantic_model.encode([query], show_progress_bar=False)
        
        scores = {}
        for intent, embeddings in self.intent_embeddings.items():
            sims = cosine_similarity(query_embedding, embeddings)[0]
            scores[intent] = float(np.max(sims))
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        if best_score < 0.3:
            # Low confidence — default to 'general' (safest fallback)
            return 'general'
        
        # Check for hybrid
        if scores.get('student', 0) > 0.45 and scores.get('general', 0) > 0.45:
            return 'hybrid'
        
        return best_intent
    
    def extract_entities(self, query):
        """Extract entities from query (always uses regex)"""
        query_lower = query.lower()
        entities = {}
        
        student_id_match = re.search(r'\b(\d{5,10})\b', query)
        if student_id_match:
            entities['student_id'] = student_id_match.group(1)
        
        departments = ['cse', 'ece', 'eee', 'civil', 'mechanical', 'it', 'mba']
        for dept in departments:
            if dept in query_lower:
                entities['department'] = dept.upper()
                break
        
        name_match = re.search(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', query)
        if name_match:
            entities['name'] = name_match.group(1)
        
        return entities


if __name__ == "__main__":
    detector = IntentDetector()
    
    test_queries = [
        ("hi", "greeting"),
        ("Who is the principal?", "general"),
        ("What is the CGPA of student 12345?", "student"),
        ("List CSE students", "student"),
        ("What are college timings?", "general"),
        ("how do i apply for the bonofide", "general"),
        ("how many students pass out every year?", "general"),
        ("how can i apply for the passport?", "general"),
        ("what is the place of tkr college?", "general"),
        ("top 5 students gpa details", "student"),
        ("what is my exam time table", "general"),
        ("how many students were placed?", "student"),
    ]
    
    mode = "Gemma 3 1B (Ollama)" if detector.use_gemma_routing else (
        "Semantic (MiniLM)" if detector.use_semantic_fallback else "Regex")
    print(f"\nMode: {mode}")
    print("="*70 + "\n")
    
    correct = 0
    for query, expected in test_queries:
        result = detector.detect_intent(query)
        match = "PASS" if result == expected else "FAIL"
        if result == expected:
            correct += 1
        print(f"  [{match}] '{query}' -> {result} (expected: {expected})\n")
    
    print(f"Score: {correct}/{len(test_queries)} ({int(correct/len(test_queries)*100)}%)")
