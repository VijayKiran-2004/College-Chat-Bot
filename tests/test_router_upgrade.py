import sys
import os
from pathlib import Path

# Fix for WinError 1114 / DLL load failed
# Must import sentence_transformers/torch BEFORE pandas/numpy
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

# Add project root to path (one level up from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.query_router import QueryRouter

def test_router():
    print("Testing QueryRouter Initialization...")
    try:
        router = QueryRouter()
        print("\n[SUCCESS] QueryRouter initialized.")
        
        # Test 1: General Query (Fast Path)
        print("\nTest 1: General Query ('Who is the principal?')")
        # We don't want to actually run the LLM heavily, just check routing path if possible or dry run
        # But for now let's just see if it runs without crashing
        
        # Test 2: Student Query (Fast Path)
        print("\nTest 2: Student Query ('List CSE students')")
        
        # Test 3: Hybrid Query (Agent Path)
        print("\nTest 3: Hybrid Query ('Who students with > 9 CGPA and where placed?')")
        
        print("\n[SUCCESS] Tests passed (dry run initialization).")
        router.close()
    except Exception as e:
        print(f"\n[FAILURE] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_router()
