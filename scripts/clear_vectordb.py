"""
Script to clear the Vector Database (ChromaDB) and related index files.
This resets the vector search capability but preserves the SQL database and source
corpus.
"""

import os
import shutil
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform.startswith("win"):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def clear_vector_db():
    print("=" * 70)
    print("CLEARING VECTOR DATABASE")
    print("=" * 70 + "\n")

    # Define paths to clear
    # We use relative paths assuming the script is run from project root or scripts dir
    # But to be safe, we'll locate based on this file's position
    project_root = Path(__file__).resolve().parent.parent
    vectordb_path = project_root / "app/database/vectordb"

    if not vectordb_path.exists():
        print(f"⚠ Vector DB directory not found at: {vectordb_path}")
        return

    targets = [
        # ChromaDB Directory
        {
            "path": vectordb_path / "chroma",
            "type": "dir",
            "desc": "ChromaDB Persistence Directory",
        },
        # FAISS Index
        {
            "path": vectordb_path / "ultrarag_faiss.index",
            "type": "file",
            "desc": "FAISS Index File",
        },
        {
            "path": vectordb_path / "faiss_index.bin",
            "type": "file",
            "desc": "Legacy FAISS Index File",
        },
        # BM25 Cache
        {
            "path": vectordb_path / "ultrarag_bm25.pkl",
            "type": "file",
            "desc": "BM25 Cache File",
        },
        {
            "path": vectordb_path / "bm25_index.pkl",
            "type": "file",
            "desc": "Legacy BM25 Cache File",
        },
    ]

    cleared_count = 0

    for target in targets:
        path = target["path"]
        if path.exists():
            try:
                if target["type"] == "dir":
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"✓ Deleted {target['desc']}: {path.name}")
                cleared_count += 1
            except Exception as e:
                print(f"❌ Error deleting {path.name}: {e}")
        else:
            print(f"• Skipped {target['desc']} (Not found)")

    print("\n" + "=" * 70)
    if cleared_count > 0:
        print(f"✓ Successfully cleared {cleared_count} items.")
        print("The Vector Database has been reset.")
        print("To rebuild it, run: python scripts/ingest.py")
    else:
        print("Database was already clean.")
    print("=" * 70)


if __name__ == "__main__":
    clear_vector_db()
