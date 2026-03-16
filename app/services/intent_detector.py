"""
Intent Detector - Smart routing using Semantic Similarity (all-MiniLM-L6-v2)
Uses embeddings-based cosine similarity for fast, local intent classification.
Falls back to 'general' if model is unavailable.
"""

import re
import sys
import io
import numpy as np

# Fix Windows encoding (only if not already wrapped)
if sys.platform.startswith("win") and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class IntentDetector:
    def __init__(self, semantic_model=None):
        """Initialize intent detector with semantic similarity (all-MiniLM-L6-v2)"""
        self.use_semantic_fallback = False
        self.semantic_model = None
        self.intent_embeddings = {}

        # Load semantic model
        try:
            from sentence_transformers import SentenceTransformer

            self.semantic_model = (
                semantic_model
                if semantic_model is not None
                else SentenceTransformer("all-MiniLM-L6-v2")
            )

            intent_examples = {
                "greeting": ["hi", "hello", "hey", "good morning", "how are you"],
                "student": [
                    "show me student cgpa scores",
                    "list students with high gpa",
                    "which students are placed",
                    "top 5 students by gpa",
                    "how many students were placed",
                    "student attendance details",
                    "number of placed students",
                    "students with cgpa above 8",
                    "average gpa of students",
                    "branch wise student count",
                    "student placement statistics",
                    "how many students are not placed",
                ],
                "general": [
                    "who is the principal",
                    "what are college timings",
                    "how to apply for bonafide certificate",
                    "what courses are offered",
                    "fee structure details",
                    "hostel facilities",
                    "transport routes",
                    "campus life and events",
                    "exam schedule and timetable",
                    "college history",
                    "how to apply for scholarship",
                    "college location and address",
                    "how many students pass out every year",
                    "what is the college ranking",
                    "tell me about admissions",
                ],
            }

            self.intent_embeddings = {}
            for intent, examples in intent_examples.items():
                self.intent_embeddings[intent] = self.semantic_model.encode(
                    examples, show_progress_bar=False
                )

            self.use_semantic_fallback = True
            print("  ✓ Semantic routing ready (all-MiniLM-L6-v2)")
        except Exception as e:
            print(
                f"  ⚠ Semantic model not available ({e}),"
                " defaulting all queries to 'general'"
            )

    def detect_intent(self, query):
        """
        Detect query intent. Priority: Regex greetings → Semantic similarity

        Returns: 'greeting', 'student', 'general', or 'hybrid'
        """
        query_lower = query.lower().strip()

        # Always check greetings with regex first (instant)
        greetings_pattern = (
            r"^(hi|hello|hey|greetings|how are you|how r u|how are u|"
            r"whats up|what's up|how do you do|good morning|good afternoon|"
            r"good evening)[\s\?\!\.]*$"
        )
        if re.match(greetings_pattern, query_lower):
            return "greeting"

        # Semantic similarity routing
        if self.use_semantic_fallback:
            try:
                return self._semantic_detect_intent(query)
            except Exception as e:
                print(f"  ⚠ Semantic routing failed ({e}), defaulting to general")

        # Default to 'general' when no detection method is available
        return "general"

    def _semantic_detect_intent(self, query):
        """Use semantic similarity for intent classification"""
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
            return "general"

        # Check for hybrid
        if scores.get("student", 0) > 0.45 and scores.get("general", 0) > 0.45:
            return "hybrid"

        return best_intent

    def extract_entities(self, query):
        """Extract entities from query (always uses regex)"""
        query_lower = query.lower()
        entities = {}

        student_id_match = re.search(r"\b(\d{5,10})\b", query)
        if student_id_match:
            entities["student_id"] = student_id_match.group(1)

        departments = ["cse", "ece", "eee", "civil", "mechanical", "it", "mba"]
        for dept in departments:
            if dept in query_lower:
                entities["department"] = dept.upper()
                break

        name_match = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", query)
        if name_match:
            entities["name"] = name_match.group(1)

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

    mode = "Semantic (MiniLM)" if detector.use_semantic_fallback else "Regex Default"
    print(f"\nMode: {mode}")
    print("=" * 70 + "\n")

    correct = 0
    for query, expected in test_queries:
        result = detector.detect_intent(query)
        match = "PASS" if result == expected else "FAIL"
        if result == expected:
            correct += 1
        print(f"  [{match}] '{query}' -> {result} (expected: {expected})\n")

    print(
        f"Score: {correct}/{len(test_queries)} ({int(correct/len(test_queries)*100)}%)"
    )
