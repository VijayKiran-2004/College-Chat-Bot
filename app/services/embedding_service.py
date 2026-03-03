import threading
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, model_name='all-MiniLM-L6-v2'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
                    print(f"Loading shared model: {model_name}...")
                    cls._model = SentenceTransformer(model_name)
                    print(f"✓ Shared model '{model_name}' loaded successfully.")
        return cls._instance

    @property
    def model(self):
        return self._model

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Helper function to get the shared model instance"""
    return EmbeddingService(model_name).model
