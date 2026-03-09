"""
ChromaDB Ingestion Script
Processes scraped data and stores it in ChromaDB with Semantic Chunking.
"""

import json
import os
import sys
import uuid

import chromadb
from chromadb.utils import embedding_functions
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


# Fix Windows encoding
if sys.platform.startswith("win"):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# Configuration
INPUT_FILE = "data/chunks/corpus_ultrarag.jsonl"
CHROMA_PATH = "app/database/vectordb/chroma"
COLLECTION_NAME = "college_data"


def normalize_text(text):
    """Basic text normalization."""
    return " ".join(text.split())


def main():
    print("=" * 70)
    print("CHROMADB INGESTION (Contextual Chunking)")
    print("=" * 70 + "\n")

    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        print("Run 'python scripts/scrape.py' first.")
        sys.exit(1)

    documents = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    print(f"Loaded {len(documents)} documents")

    # 2. Initialize ChromaDB
    print(f"Initializing ChromaDB at {CHROMA_PATH}...")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print("Collection created")

    # 3. LLM Cleaning Setup
    print("Skipping LLM cleaning prompt...")

    cleaning_prompt = PromptTemplate.from_template(
        """
You are a Data Cleaner for a College Chatbot.

Below is raw text scraped from a webpage titled '{title}'.

Task:
Rewrite the content preserving ALL factual details such as
dates, names, rules, fees, phone numbers, and addresses.

Remove:
- navigation menus
- copyright footers
- advertisements
- contact forms

If the text is empty or irrelevant return 'SKIP'.

Raw Content:
{raw_text}

Cleaned Content:
"""
    )

    # 4. Semantic Chunking
    print("Processing and Chunking documents...")

    chunker_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    text_splitter = SemanticChunker(
        embeddings=chunker_embeddings,
        breakpoint_threshold_type="percentile",
    )

    ids = []
    metadatas = []
    final_texts = []

    total_chunks = 0

    for idx, doc in enumerate(documents):

        raw_text = doc.get("contents", "")
        title = doc.get("title", "Unknown Title")
        source = doc.get("source", "")

        processed_text = raw_text

        print(f"[{idx + 1}/{len(documents)}] Processing: " f"{title[:30]}...")

        chunks = text_splitter.split_text(processed_text)

        for i, chunk in enumerate(chunks):

            contextual_text = f"Title: {title}\n" f"Source: {source}\n\n" f"{chunk}"

            ids.append(str(uuid.uuid4()))
            final_texts.append(contextual_text)

            metadatas.append(
                {
                    "source": source,
                    "title": title,
                    "original_chunk_index": i,
                }
            )

            total_chunks += 1

    print(f"Generated {total_chunks} contextual chunks")

    # 5. Upsert to ChromaDB
    print("Upserting to ChromaDB...")

    BATCH_SIZE = 100

    if ids:
        for i in range(0, len(ids), BATCH_SIZE):

            batch_ids = ids[i : i + BATCH_SIZE]
            batch_texts = final_texts[i : i + BATCH_SIZE]
            batch_metadatas = metadatas[i : i + BATCH_SIZE]

            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )

            print(
                "Processed batch "
                f"{i // BATCH_SIZE + 1}/"
                f"{(len(ids) - 1) // BATCH_SIZE + 1}"
            )

    else:
        print("No suitable content found to ingest.")

    print(f"\nSuccessfully stored {total_chunks} chunks in ChromaDB")

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
