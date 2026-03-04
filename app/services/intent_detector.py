"""
Intent Detector - Smart routing using Gemma 3 1B (via Ollama)
+ Semantic Similarity fallback.

Uses google/gemma-3-1b-it through Ollama for AI-powered
intent classification.

Falls back to semantic similarity (all-MiniLM-L6-v2)
if Ollama is unavailable.
"""

import io
import os
import re
import sys

import numpy as np
import requests

# Fix Windows encoding (only if not already wrapped)
if sys.platform.startswith("win") and not isinstance(
    sys.stdout, io.TextIOWrapper
):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8"
    )


class IntentDetector:
    def __init__(self, semantic_model=None):
        """Initialize intent detector."""

        self.ollama_url = os.environ.get(
            "OLLAMA_URL",
            "http://127.0.0.1:11434/api/generate",
        )
        self.router_model = "gemma3:1b"
        self.use_gemma_routing = False
        self.use_semantic_fallback = False
        self.semantic_model = None
        self.intent_embeddings = {}

        self.router_prompt = (
            "Classify this college chatbot query into ONE category.\n"
            "Reply with ONLY the category name.\n\n"
            "GREETING: Simple greetings.\n\n"
            "STUDENT: Queries requesting student data or statistics.\n\n"
            "GENERAL: Queries about college information.\n\n"
            "Reply ONLY: GREETING, STUDENT, or GENERAL"
        )

        try:
            from sentence_transformers import SentenceTransformer

            self.semantic_model = (
                semantic_model
                if semantic_model is not None
                else SentenceTransformer("all-MiniLM-L6-v2")
            )

            intent_examples = {
                "greeting": [
                    "hi", "hello", "hey",
                    "good morning", "how are you",
                ],
                "student": [
                    "show me student cgpa scores",
                    "list students with high gpa",
                    "which students are placed",
                    "top 5 students by gpa",
                    "how many students were placed",
                ],
                "general": [
                    "who is the principal",
                    "what are college timings",
                    "how to apply for bonafide certificate",
                    "fee structure details",
                    "college location and address",
                ],
            }

            self.intent_embeddings = {}
            for intent, examples in intent_examples.items():
                self.intent_embeddings[intent] = (
                    self.semantic_model.encode(
                        examples,
                        show_progress_bar=False,
                    )
                )

            self.use_semantic_fallback = True
            print("  ✓ Semantic fallback ready")

        except Exception as exc:
            print(f"  ⚠ Semantic fallback not available ({exc})")

    def detect_intent(self, query):
        """
        Detect query intent.

        Priority:
        Regex greetings → Gemma 3 → Semantic.
        """

        query_lower = query.lower().strip()

        greetings_pattern = (
            r"^(hi|hello|hey|greetings|how are you|"
            r"how r u|how are u|whats up|what's up|"
            r"good morning|good afternoon|"
            r"good evening)[\s\?\!\.]*$"
        )

        if re.match(greetings_pattern, query_lower):
            return "greeting"

        if self.use_gemma_routing:
            try:
                return self._gemma_detect_intent(query)
            except Exception as exc:
                print(
                    f"  ⚠ Gemma routing failed ({exc}), "
                    "using semantic fallback"
                )

        if self.use_semantic_fallback:
            try:
                return self._semantic_detect_intent(query)
            except Exception as exc:
                print(
                    f"  ⚠ Semantic routing failed ({exc}), "
                    "defaulting to general"
                )

        return "general"

    def _gemma_detect_intent(self, query):
        """Use Gemma via Ollama for intent classification."""

        prompt = (
            f"{self.router_prompt}\n\n"
            f"User query: \"{query}\"\n\n"
            "Category:"
        )

        resp = requests.post(
            self.ollama_url,
            json={
                "model": self.router_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "temperature": 0,
                    "top_k": 1,
                },
            },
            timeout=15,
        )

        if resp.status_code != 200:
            raise Exception(
                f"Ollama returned {resp.status_code}"
            )

        result = resp.json().get("response", "").strip().upper()

        if "GREETING" in result:
            intent = "greeting"
        elif "STUDENT" in result:
            intent = "student"
        elif "GENERAL" in result:
            intent = "general"
        else:
            if self.use_semantic_fallback:
                return self._semantic_detect_intent(query)
            return "general"

        return intent

    def _semantic_detect_intent(self, query):
        """Use semantic similarity as fallback."""

        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.semantic_model.encode(
            [query],
            show_progress_bar=False,
        )

        scores = {}
        for intent, embeddings in self.intent_embeddings.items():
            sims = cosine_similarity(
                query_embedding,
                embeddings,
            )[0]
            scores[intent] = float(np.max(sims))

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score < 0.3:
            return "general"

        if (
            scores.get("student", 0) > 0.45
            and scores.get("general", 0) > 0.45
        ):
            return "hybrid"

        return best_intent

    def extract_entities(self, query):
        """Extract entities using regex."""

        query_lower = query.lower()
        entities = {}

        student_id_match = re.search(
            r"\b(\d{5,10})\b",
            query,
        )
        if student_id_match:
            entities["student_id"] = (
                student_id_match.group(1)
            )

        departments = [
            "cse", "ece", "eee",
            "civil", "mechanical",
            "it", "mba",
        ]

        for dept in departments:
            if dept in query_lower:
                entities["department"] = dept.upper()
                break

        name_match = re.search(
            r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
            query,
        )
        if name_match:
            entities["name"] = name_match.group(1)

        return entities


if __name__ == "__main__":
    detector = IntentDetector()
    print("Intent detector ready.")
