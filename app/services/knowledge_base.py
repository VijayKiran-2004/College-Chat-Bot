import json

import numpy as np


class KnowledgeBase:
    """
    Handles fast knowledge base retrieval using exact matching
    and semantic fallback.
    """

    def __init__(self, kb_path, semantic_model):
        self.data = self._load_data(kb_path)
        self.kb_encoder = semantic_model

        self.kb_entries = []
        self.kb_embeddings = []

        print("Building KB semantic index...")
        self._build_kb_index()
        print("✓ KB semantic index ready")

    def _load_data(self, kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            print(f"⚠ Could not load KB from {kb_path}: {exc}")
            return {}

    def _build_kb_index(self):
        """Pre-compute embeddings for semantic matching."""

        if not self.data:
            return

        def flatten_kb(data, category="", parent_key=""):
            for key, value in data.items():
                current_key = (
                    f"{parent_key}.{key}" if parent_key else key
                )

                if isinstance(value, dict):
                    flatten_kb(value, category or key, current_key)

                elif isinstance(value, list):
                    text_value = ", ".join(str(item) for item in value)
                    search_text = (
                        f"{category} {key} {text_value}"
                    )
                    self.kb_entries.append(
                        {
                            "category": category or key,
                            "key": key,
                            "value": text_value,
                            "search_text": search_text,
                        }
                    )

                elif isinstance(value, str):
                    search_text = f"{category} {key} {value}"
                    self.kb_entries.append(
                        {
                            "category": category or key,
                            "key": key,
                            "value": value,
                            "search_text": search_text,
                        }
                    )

        flatten_kb(self.data)

        search_texts = [
            entry["search_text"]
            for entry in self.kb_entries
        ]

        self.kb_embeddings = self.kb_encoder.encode(
            search_texts,
            show_progress_bar=False,
        )

        self.kb_embeddings = np.array(self.kb_embeddings)

    def check(self, query):
        """
        KB matching with keyword fallback + semantic matching.
        Returns raw fact or None.
        """

        from sklearn.metrics.pairwise import cosine_similarity

        if not self.kb_entries:
            return None

        query_lower = query.lower()

        if "principal" in query_lower:
            return f"Principal: {self.data['personnel']['principal']}"

        query_embedding = self.kb_encoder.encode(
            [query],
            show_progress_bar=False,
        )

        similarities = cosine_similarity(
            query_embedding,
            self.kb_embeddings,
        )[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        CONFIDENCE_THRESHOLD = 0.75

        if best_score < CONFIDENCE_THRESHOLD:
            return None

        matched_entry = self.kb_entries[best_idx]
        return matched_entry["value"]

    def format_context(self):
        """Format knowledge base as context."""

        if not self.data:
            return ""

        lines = [
            f"Principal: {self.data['personnel']['principal']}",
            f"Vice Principal: "
            f"{self.data['personnel']['vice_principal']}",
            f"Timings: {self.data['timings']['working_hours']}",
            f"Founded: {self.data['history']['established']}",
            f"Affiliation: {self.data['history']['affiliation']}",
        ]

        if "statistics" in self.data:
            stats = self.data["statistics"]
            lines.append("\nFAST FACTS:")
            lines.append(
                f"• Total Students: {stats['total_students']}"
            )
            lines.append(
                f"• Placed Students: "
                f"{stats['placed_students']} "
                f"(Rate: {stats['placement_rate']})"
            )
            lines.append(
                f"• Top Recruiters: {stats['top_recruiters']}"
            )

        return "\n".join(lines)
