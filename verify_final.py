
import chromadb
from chromadb.utils import embedding_functions
import re
import os

# 1. Verify Regex Fix
print("--- Testing Greeting Regex ---")
greetings_pattern = r"^(hi|hello|hey|greetings|how are you|how r u|how are u|whats up|what's up|how do you do|good morning|good afternoon|good evening)[\s\?\!\.]*$"
test_queries = ["hi", "hello", "hi!", "how r u?"]
for q in test_queries:
    match = re.match(greetings_pattern, q.lower())
    print(f"  Query '{q}': {'Matched' if match else 'Failed'}")

# 2. Verify ChromaDB
print("\n--- Testing ChromaDB Collection ---")
CHROMA_PATH = "app/database/vectordb/chroma"
COLLECTION_NAME = "college_data"

if os.path.exists(CHROMA_PATH):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"✓ Collection '{COLLECTION_NAME}' exists with {count} chunks.")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("✗ ChromaDB path not found.")
