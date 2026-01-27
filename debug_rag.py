
import sys
import os

# Add app directory to path
sys.path.insert(0, os.getcwd())

try:
    from app.services.ultra_rag import UltraRAGSystem
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Initializing UltraRAGSystem...")
try:
    rag = UltraRAGSystem()
    print("Success!")
except Exception as e:
    print(f"Initialization failed: {e}")
    import traceback
    traceback.print_exc()
