
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(os.getcwd())))

# Fix for WinError 1114 (DLL initialization failed)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

from app.services.query_router import QueryRouter

def test_issue():
    print("Initializing QueryRouter...")
    router = QueryRouter()
    
    conversation = [
        "how many students got placed?",
        "how many boys got placed ?",
        "what was the average gpa?"
    ]
    
    print("\n--- Starting Conversation Replay ---")
    for query in conversation:
        print(f"\nUser: {query}")
        try:
            response = router.route_query(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")
            
    router.close()

if __name__ == "__main__":
    test_issue()
