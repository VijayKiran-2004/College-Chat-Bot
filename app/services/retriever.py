import os
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
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    documents.append(doc)
        except Exception as e:
            print(f"⚠ Could not load corpus: {e}")
        return documents

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            print("Connecting to ChromaDB...")
            chroma_db_path = self.project_root / 'app/database/vectordb/chroma'
            chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
            
            collection = chroma_client.get_collection(
                name="college_data"
            )
            print("✓ Connected to ChromaDB")
            return collection
        except Exception as e:
            print(f"⚠ Error initializing retrieval: {e}")
            print("Run 'python scripts/ingest.py' to populate the database.")
            return None

    def retrieve(self, query, top_k=5):
        """Retrieve using ChromaDB (Semantic Search) with relevance filtering"""
        if not self.collection:
            print("⚠ Database not initialized, skipping retrieval")
            return []
            
        try:
            # Use shared model to encode query manually (avoids function conflict)
            shared_model = get_embedding_model()
            query_embeddings = shared_model.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k
            )
            
            docs = []
            if not results['documents']:
                return []
            
            # Get distances if available for filtering
            distances = results.get('distances', [[]])[0] if results.get('distances') else []
                
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = distances[i] if i < len(distances) else 0
                
                # Filter out low-relevance documents
                if distance > 1.0:
                    print(f"  [Filtering] Skipping low-relevance doc (distance: {distance:.3f})")
                    continue
                
                docs.append({
                    "contents": content,
                    "metadata": metadata,
                    "relevance_score": 1 - distance
                })
            
            # If we filtered out everything, return top 2 anyway (better than nothing)
            if not docs and len(results['documents'][0]) > 0:
                print("  [Warning] All docs filtered, returning top 2 anyway")
                for i in range(min(2, len(results['documents'][0]))):
                    content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = distances[i] if i < len(distances) else 0
                    docs.append({
                        "contents": content,
                        "metadata": metadata,
                        "relevance_score": 1 - distance
                    })
            
            return docs
            
        except Exception as e:
            print(f"⚠ ChromaDB retrieval error: {e}")
            return []
