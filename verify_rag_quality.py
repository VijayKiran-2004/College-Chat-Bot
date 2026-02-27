
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(os.getcwd())))

from app.services.ultra_rag import UltraRAGSystem

def test_rag():
    print("Initializing UltraRAGSystem...")
    rag = UltraRAGSystem()
    
    queries = [
        "hi",
        "who is the head of the transportation",
        "what is the process of the bonofide applying",
        "how can i apply for the passport?"
    ]
    
    print("\n" + "="*50)
    print("BACKEND RESPONSE VERIFICATION")
    print("="*50)
    
    for q in queries:
        print(f"\nUser Query: {q}")
        print("-" * 30)
        # Use return_dict=True to see metadata/links
        result = rag(q, return_dict=True)
        
        # In a real run, this might be a generator if stream=True
        # But here rag() returns a dict or string
        if isinstance(result, dict):
            print(f"Source: {result.get('source', 'Unknown')}")
            print(f"Response:\n{result.get('response', '')}")
            # Check for links in the response text (UltraRAG injects them into the string)
            if "Source Links" in result.get('response', ''):
                print("✓ Source Links found in response text")
            else:
                print("✗ No Source Links found in response text")
        else:
            print(f"Response:\n{result}")

if __name__ == "__main__":
    test_rag()
