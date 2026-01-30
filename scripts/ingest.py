"""
ChromaDB Ingestion Script
Processes scraped data and stores it in ChromaDB with "Contextual Chunking".
"""

import json
import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# Fix Windows encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
INPUT_FILE = "app/database/vectordb/scraped_data.jsonl"
CHROMA_PATH = "app/database/vectordb/chroma"
COLLECTION_NAME = "college_data"

def normalize_text(text):
    """Basic text normalization"""
    return " ".join(text.split())

def main():
    print("="*70)
    print("CHROMADB INGESTION (Contextual Chunking)")
    print("="*70 + "\n")

    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found: {INPUT_FILE}")
        print("Run 'python scripts/scrape.py' first.")
        sys.exit(1)
        
    documents = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    
    print(f"✓ Loaded {len(documents)} documents")

    # 2. Initialize ChromaDB
    print(f"Initializing ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Reset collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  - Deleted existing collection")
    except Exception:
        pass
    
    # Use default all-MiniLM-L6-v2 embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    print("✓ Collection created")

    # 3. LLM Initialization (Smart Cleaning)
    print("Initializing LLM for Data Cleaning...")
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.prompts import PromptTemplate
        
        llm = None 
        # llm = ChatOllama(
        #     model="gemma2:2b",
        #     temperature=0,
        #     base_url="http://localhost:11434"
        # )
        
        cleaning_prompt = PromptTemplate.from_template("""
        You are a Data Cleaner for a College Chatbot.
        Below is raw text scraped from a webpage titled '{title}'.
        
        **Task**: Rewrite this content, preserving ALL factual details (dates, names, fees, rules, phone numbers, addresses) but removing navigation menus, copyright footers, ads, and contact forms.
        - If the text is empty or contains only error messages, return 'SKIP'.
        - Keep the output concise and fact-dense.
        
        Raw Content:
        {raw_text}
        
        Cleaned Content:
        """)
        
        print("✓ LLM Ready: gemma2:2b")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize LLM ({e}). Proceeding with raw text.")
        llm = None

    # 4. Contextual Chunking & Processing
    print("Processing and Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    ids = []
    embeddings_updates = []
    metadatas = []
    final_texts = []

    total_chunks = 0
    
    for idx, doc in enumerate(documents):
        raw_text = doc.get('contents', '')
        title = doc.get('title', 'Unknown Title')
        source = doc.get('source', '')
        
        # --- LLM CLEANING STEP ---
        processed_text = raw_text
        if llm:
            try:
                print(f"  [{idx+1}/{len(documents)}] Cleaning: {title[:30]}...", end='', flush=True)
                
                # Check length to avoid context overflow
                if len(raw_text) > 10000:
                    raw_text = raw_text[:10000] # Truncate for safety
                
                response = llm.invoke(cleaning_prompt.format(title=title, raw_text=raw_text))
                cleaned = response.content.strip()
                
                if cleaned == 'SKIP' or len(cleaned) < 50:
                    print(" -> SKIPPED (Empty/Irrelevant)")
                    continue
                
                processed_text = cleaned
                print(" -> Done")
            except Exception as e:
                print(f" -> Failed ({e}) - Using Raw")
        # -------------------------
        
        chunks = text_splitter.split_text(processed_text)
        
        for i, chunk in enumerate(chunks):
            # Contextual Enrichment: Prepend Title and Source
            contextual_text = f"Title: {title}\nSource: {source}\n\n{chunk}"
            
            ids.append(str(uuid.uuid4()))
            final_texts.append(contextual_text)
            metadatas.append({
                "source": source,
                "title": title,
                "original_chunk_index": i,
                "is_llm_cleaned": True if llm else False
            })
            total_chunks += 1

    print(f"✓ Generated {total_chunks} contextual chunks")

    # 5. Upsert to ChromaDB
    print("Upserting to ChromaDB (this may take a moment)...")
    
    # Batch processing to avoid memory issues
    BATCH_SIZE = 100
    if ids:
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_texts = final_texts[i:i+BATCH_SIZE]
            batch_metadatas = metadatas[i:i+BATCH_SIZE]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            print(f"  - Processed batch {i//BATCH_SIZE + 1}/{(len(ids)-1)//BATCH_SIZE + 1}")
    else:
        print("⚠ No suitable content found to ingest.")
        
    print(f"\n✓ Successfully stored {total_chunks} chunks in ChromaDB")
    print("\n" + "="*70)
    print("INGESTION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
