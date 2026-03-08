"""
Generate Unified Vectors JSON
Reads scraped_data.jsonl and faq_rows.json, chunks text content,
and outputs unified_vectors.json for corpus_converter.py to consume.

This script is idempotent — it overwrites the output file on each run.

Usage:
    python scripts/generate_vectors.py
"""

import json
import os
import sys
import re
from pathlib import Path

# Fix Windows encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Always anchor paths relative to THIS script, not the cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration — absolute paths
# Primary source (from prepare_data.py)
SCRAPED_JSONL = str(_PROJECT_ROOT / "app" / "database" / "vectordb" / "scraped_data.jsonl")
# Backup source (direct from scraper)
SCRAPED_MASTER = str(_PROJECT_ROOT / "data" / "scraped_data" / "all_results.json")

FAQ_DATA     = str(_PROJECT_ROOT / "data" / "rawdata" / "faq_rows.json")
OUTPUT_FILE  = str(_PROJECT_ROOT / "data" / "chunks" / "unified_vectors.json")

# Chunking parameters
MAX_CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 150    # overlap for context continuity


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common nav/footer patterns
    text = re.sub(r'(Skip to content|Menu|Search|Copyright ©.*|All Rights Reserved.*)', '', text, flags=re.IGNORECASE)
    return text.strip()


def chunk_text(text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks at sentence boundaries"""
    if len(text) <= max_size:
        return [text]
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of previous chunk
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap//5:]) if len(words) > overlap//5 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_size]]


def load_scraped_data():
    """Load documents from scraped_data.jsonl"""
    # 1. Load Scraped Data
    processed_docs = []
    
    if os.path.exists(SCRAPED_JSONL):
        print(f"Loading scraped data from JSONL...")
        with open(SCRAPED_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        processed_docs.append(doc)
                    except json.JSONDecodeError:
                        continue
    elif os.path.exists(SCRAPED_MASTER):
        print(f"Loading scraped data directly from Master JSON...")
        try:
            with open(SCRAPED_MASTER, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
                
            # Master data is a list of items: {"url": "...", "data": {"sections": [...]}}
            if isinstance(master_data, list):
                for item in master_data:
                    url = item.get("url", "")
                    data = item.get("data", {})
                    sections = data.get("sections", [])
                    
                    content = ""
                    for sec in sections:
                        heading = sec.get("title", "") or sec.get("heading", "")
                        raw_sec_content = sec.get("content", "")
                        
                        # Handle content being a list or string
                        if isinstance(raw_sec_content, list):
                            sec_text = "\n".join([str(x) for x in raw_sec_content])
                        else:
                            sec_text = str(raw_sec_content)
                            
                        content += f"## {heading}\n{sec_text}\n\n"
                    
                    if content.strip():
                        processed_docs.append({
                            "title": url,  # Use URL as title if no specific page title
                            "source": url,
                            "contents": content
                        })
            else:
                print(f"  ⚠ Unexpected format in {SCRAPED_MASTER} (expected list)")
        except Exception as e:
            print(f"  ⚠ Error reading {SCRAPED_MASTER}: {e}")
    else:
        print(f"  ⚠ Scraped data not found in {SCRAPED_JSONL} or {SCRAPED_MASTER}")
    
    print(f"  ✓ Loaded {len(processed_docs)} scraped documents")
    return processed_docs


def load_faq_data():
    """Load FAQ data from faq_rows.json"""
    documents = []
    if not os.path.exists(FAQ_DATA):
        print(f"  ⚠ FAQ data not found: {FAQ_DATA}")
        return documents
    
    with open(FAQ_DATA, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    if isinstance(faq_data, list):
        for item in faq_data:
            if isinstance(item, dict):
                # Combine question and answer if present
                question = item.get('question', item.get('Q', ''))
                answer = item.get('answer', item.get('A', ''))
                text = f"Q: {question}\nA: {answer}" if question and answer else str(item)
                documents.append({
                    'contents': text,
                    'title': question[:80] if question else 'FAQ',
                    'source': 'faq_rows.json',
                    'type': 'faq'
                })
    
    print(f"  ✓ Loaded {len(documents)} FAQ entries")
    return documents


def main():
    print("=" * 70)
    print("GENERATE UNIFIED VECTORS")
    print("=" * 70 + "\n")
    
    # 1. Load all source data
    print("Loading source data...")
    scraped_docs = load_scraped_data()
    faq_docs = load_faq_data()
    
    all_docs = scraped_docs + faq_docs
    print(f"\n  Total source documents: {len(all_docs)}")
    
    if not all_docs:
        print("❌ No source data found. Run 'python scripts/scrape.py' first.")
        sys.exit(1)
    
    # 2. Process and chunk documents
    print("\nProcessing and chunking...")
    unified_vectors = []
    seen_texts = set()  # Deduplication
    
    for doc in all_docs:
        raw_text = doc.get('contents', '')
        title = doc.get('title', 'Unknown')
        source = doc.get('source', doc.get('id', 'unknown'))
        url = doc.get('url', source if source.startswith('http') else '')
        
        cleaned = clean_text(raw_text)
        if len(cleaned) < 30:  # Skip empty/tiny content
            continue
        
        chunks = chunk_text(cleaned)
        
        for chunk in chunks:
            # Deduplicate
            chunk_hash = chunk[:100].lower()
            if chunk_hash in seen_texts:
                continue
            seen_texts.add(chunk_hash)
            
            # --- Title-Anchored Chunk ---
            # Prepend a short title tag so the embedding captures page context
            short_title = title[:60] if title and title != url else ""
            if short_title and short_title.lower() not in chunk[:80].lower():
                anchored_chunk = f"[TKRCET | {short_title}]\n{chunk}"
            elif url:
                anchored_chunk = f"[TKRCET | {url}]\n{chunk}"
            else:
                anchored_chunk = chunk
            
            unified_vectors.append({
                "text": anchored_chunk,
                "source": source,
                "url": url,
                "metadata": {
                    "title": title,
                    "source": source,
                    "url": url
                }
            })
    
    print(f"  ✓ Generated {len(unified_vectors)} chunks")
    
    # 3. Write output (overwrites existing file — idempotent)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unified_vectors, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print(f"  Next step: python scripts/corpus_converter.py")
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
