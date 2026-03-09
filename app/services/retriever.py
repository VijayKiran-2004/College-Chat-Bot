import json
from pathlib import Path
from app.services.embedding_service import get_embedding_model


class Retriever:
    """Handles hybrid document retrieval using ChromaDB"""

    def __init__(self, project_root, corpus_path):
        self.project_root = Path(project_root)
        self.corpus_path = corpus_path
        self.documents = self._load_corpus()
        print(f"✓ Loaded {len(self.documents)} documents")
        self.collection = self._init_chroma()

    def _load_corpus(self):
        documents = []
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    documents.append(doc)
        except Exception as e:
            print(f"⚠ Could not load corpus: {e}")
        return documents

    def _init_chroma(self):
        try:
            import chromadb

            print("Connecting to ChromaDB...")
            chroma_db_path = self.project_root / "app/database/vectordb/chroma"
            chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))

            collection = chroma_client.get_collection(name="college_data")
            print("✓ Connected to ChromaDB")
            return collection
        except Exception as e:
            print(f"⚠ Error initializing retrieval: {e}")
            print("Run 'python scripts/ingest.py' to populate the database.")
            return None

    def retrieve(self, query, top_k=5):
        """Retrieve using ChromaDB (Semantic Search) with relevance filtering
        and re-ranking"""
        if not self.collection:
            print("⚠ Database not initialized, skipping retrieval")
            return []

        try:
            # Use shared model to encode query manually (avoids function conflict)
            shared_model = get_embedding_model()
            query_embeddings = shared_model.encode([query]).tolist()

            # Retrieve broader set (top_k+2) then re-rank down to top_k
            fetch_k = max(top_k + 2, 4)
            results = self.collection.query(
                query_embeddings=query_embeddings, n_results=fetch_k
            )

            docs = []
            if not results["documents"]:
                return []

            # Get distances if available for filtering
            distances = (
                results.get("distances", [[]])[0] if results.get("distances") else []
            )

            for i in range(len(results["documents"][0])):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = distances[i] if i < len(distances) else 0

                # Filter out very low-relevance documents
                if distance > 1.0:
                    print(
                        f"  [Filtering] Skipping low-relevance doc (distance: "
                        f"{distance:.3f})"
                    )
                    continue

                docs.append(
                    {
                        "contents": content,
                        "metadata": metadata,
                        "relevance_score": 1 - distance,
                    }
                )

            # If we filtered out everything, return top 2 anyway (better than nothing)
            if not docs and len(results["documents"][0]) > 0:
                print("  [Warning] All docs filtered, returning top 2 anyway")
                for i in range(min(2, len(results["documents"][0]))):
                    content = results["documents"][0][i]
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    distance = distances[i] if i < len(distances) else 0
                    docs.append(
                        {
                            "contents": content,
                            "metadata": metadata,
                            "relevance_score": 1 - distance,
                        }
                    )

            # --- Re-Ranker: score each doc against query using all-MiniLM-L6-v2 ---
            if len(docs) > 1:
                from sklearn.metrics.pairwise import cosine_similarity

                query_emb = shared_model.encode([query], show_progress_bar=False)
                doc_texts = [
                    d["contents"][:500] for d in docs
                ]  # truncate to save compute
                doc_embs = shared_model.encode(doc_texts, show_progress_bar=False)
                rerank_scores = cosine_similarity(query_emb, doc_embs)[0]
                for i, doc in enumerate(docs):
                    doc["relevance_score"] = float(rerank_scores[i])
                # Sort by re-rank score (highest first) and trim to top_k
                docs = sorted(docs, key=lambda d: d["relevance_score"], reverse=True)[
                    :top_k
                ]
                print(f"  [Re-ranker] Best score: {docs[0]['relevance_score']:.3f}")

            return docs

        except Exception as e:
            print(f"⚠ ChromaDB retrieval error: {e}")
            return []
